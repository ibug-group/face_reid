import os
import cv2
import sys
import multiprocessing
import ibug_face_tracker
if sys.platform != "win32":
    import ffmpeg_videoreader


class VideoFileReaderFFMPEG:
    use_bgr_colour_model = False

    def __init__(self, input_video_file):
        self.input_video = ffmpeg_videoreader.VideoReader(input_video_file)
        self.fps = max(round(self.input_video.average_FPS), 1)
        self.num_frames = self.input_video.number_of_frames
        self.frame_number = 0

    def grab_frame(self):
        frame, _ = self.input_video.get_nth_frame(self.frame_number)
        self.frame_number = self.frame_number + 1
        return frame

    def close(self):
        pass


class VideoReaderOpenCV:
    use_bgr_colour_model = True

    def __init__(self, input_video_file):
        self.input_video = cv2.VideoCapture(input_video_file)
        self.fps = self.input_video.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    def grab_frame(self):
        ret, frame = self.input_video.read()
        if ret:
            return frame
        else:
            raise ValueError()

    def close(self):
        self.input_video.release()


def face_tracker_worker(job_queue, print_lock, worker_id, plot_landmarks, minimum_face_size,
                        face_detection_scale, minimum_face_detection_gap):
    try:
        # Initialise the tracker
        with print_lock:
            print("Initialising worker #%d..." % worker_id)
        model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        tracker = ibug_face_tracker.FaceTracker(os.path.join(model_folder, "new3_68_pts_UAD_1_tr_6_cas_15.dat"),
                                                os.path.join(model_folder, "additional_svrs.model"))
        tracker.minimum_face_size = minimum_face_size
        tracker.face_detection_scale = face_detection_scale
        tracker.minimum_face_detection_gap = minimum_face_detection_gap
        with print_lock:
            print("Worker #%d has been initialised." % worker_id)

        # Process the videos
        while True:
            job = job_queue.get_nowait()
            try:
                input_video = VideoFileReaderFFMPEG(job)
            except:
                try:
                    input_video = VideoReaderOpenCV(job)
                except:
                    input_video = None
            if input_video is not None:
                with print_lock:
                    print("Worker #%d: Now tracking \"%s\"..." % (worker_id, job))

                # Create output directory
                output_folder = os.path.join(os.path.dirname(os.path.realpath(job)),
                                             os.path.splitext(os.path.basename(job))[0])
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                landmarks_folder = os.path.join(output_folder, "Facial_Landmarks_alt")
                if not os.path.exists(landmarks_folder):
                    os.mkdir(landmarks_folder)

                # Process the video
                output_video = cv2.VideoWriter()
                for idx in range(input_video.num_frames):
                    try:
                        frame = input_video.grab_frame()
                        if not input_video.use_bgr_colour_model:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        tracker.track(frame)
                        tracker.plot_current_result(frame)
                        if not output_video.isOpened():
                            output_video.open(landmarks_folder + ".avi",
                                              fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                              frameSize=(frame.shape[1], frame.shape[0]),
                                              fps=input_video.fps)
                        output_video.write(frame)
                        with open(os.path.join(landmarks_folder, "%06d.txt" % (idx + 1)), 'w') as result_file:
                            result_file.write(tracker.serialise_current_result())
                    except:
                        pass
                if output_video.isOpened():
                    output_video.release()

                with print_lock:
                    print("Worker #%d: \"%s\" has been tracked." % (worker_id, job))
            else:
                with print_lock:
                    print("Worker #%d: Failed to open \"%s\"." % (worker_id, job))
    except:
        pass
    with print_lock:
        print("Worker #%d has stopped." % worker_id)


def main():
    if len(sys.argv) < 2:
        print("Correct usage: python ibug_face_tracker_test.py [Job_Spec] [Num_Workers=1] "
              "[Plot_Landmarks=1] [Min_Face_Size=1] [Face_Detection_Sale=1.0] [Min_Face_Detection_Gap=0]")
    else:
        # Parse command-line parameters
        job_specification = sys.argv[1]
        if len(sys.argv) >= 3:
            number_of_workers = max(1, int(sys.argv[2]))
        else:
            number_of_workers = 1
        if len(sys.argv) >= 4:
            plot_landmarks = int(sys.argv[3]) != 0
        else:
            plot_landmarks = True
        if len(sys.argv) >= 5:
            minimum_face_size = int(sys.argv[4])
        else:
            minimum_face_size = 1
        if len(sys.argv) >= 6:
            face_detection_scale = float(sys.argv[5])
        else:
            face_detection_scale = 1.0
        if len(sys.argv) >= 7:
            minimum_face_detection_gap = int(sys.argv[6])
        else:
            minimum_face_detection_gap = 0

        # Parse the jobs
        print("Parsing jobs specified in \"" + job_specification + "\"... ")
        with open(job_specification, 'r') as job_specification_file:
            jobs = [job for job in [job.strip() for job in
                                    job_specification_file.readlines()] if len(job) > 0]
        print("Done, %d job(s) found." % len(jobs))

        # Track the videos
        if len(jobs) > 0:
            number_of_workers = min(number_of_workers, len(jobs))
            job_queue = multiprocessing.Queue()
            for job in jobs:
                job_queue.put_nowait(job)
            print_lock = multiprocessing.RLock()
            workers = [multiprocessing.Process(target=face_tracker_worker,
                                               args=(job_queue, print_lock, index + 1, plot_landmarks,
                                                     minimum_face_size, face_detection_scale,
                                                     minimum_face_detection_gap))
                       for index in range(number_of_workers)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        print('All done.')


if __name__ == "__main__":
    main()
