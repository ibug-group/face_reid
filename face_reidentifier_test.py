import os
import cv2
import sys
import math
import numpy as np
import ibug.face_reid
import ibug.face_tracking
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser


def read_path(config, working_folder, section, key):
    path = config.get(section, key).replace('\'', '').replace('\"', '')
    if os.path.isabs(path):
        return path
    else:
        return os.path.realpath(os.path.join(working_folder, path))


def main():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    if len(sys.argv) < 2:
        print("Correct usage: " + script_name + ".py [webcam_index] [config_file=" + script_name + ".ini]")
    else:
        webcam_index = int(sys.argv[1])
        if len(sys.argv) < 3:
            config_file = os.path.splitext(os.path.realpath(__file__))[0] + ".ini"
        else:
            config_file = sys.argv[2]
        config_folder = os.path.dirname(config_file)

        # Parse the INI file
        config = ConfigParser()
        config.read(config_file)

        # Create the multi-face tracker
        tracker_section_name = "ibug.face_tracking.MultiFaceTracker"
        ert_model_path = read_path(config, config_folder, tracker_section_name, "ert_model_path")
        auxiliary_model_path = read_path(config, config_folder, tracker_section_name, "auxiliary_model_path")
        faces_to_track = config.getint(tracker_section_name, "faces_to_track", fallback=1)
        print("Creating multi-face tracker with the following parameters...\n"
              "ert_model_path = \"" + ert_model_path + "\"\n"
              "auxiliary_model_path = \"" + auxiliary_model_path + "\"\n"
              "faces_to_track = %d" % faces_to_track)
        tracker = ibug.face_tracking.MultiFaceTracker(ert_model_path, auxiliary_model_path, faces_to_track)
        print("Multi-face tracker created.")

        # Configure the tracker
        tracker.face_detection_interval = config.getint(tracker_section_name, "face_detection_interval",
                                                        fallback=tracker.face_detection_interval)
        tracker.face_detection_scale = config.getfloat(tracker_section_name, "face_detection_scale",
                                                       fallback=tracker.face_detection_scale)
        tracker.minimum_face_size = config.getint(tracker_section_name, "minimum_face_size",
                                                  fallback=tracker.minimum_face_size)
        tracker.overlap_threshold = config.getfloat(tracker_section_name, "overlap_threshold",
                                                    fallback=tracker.overlap_threshold)
        tracker.estimate_head_pose = config.getboolean(tracker_section_name, "estimate_head_pose",
                                                       fallback=tracker.estimate_head_pose)
        tracker.eye_iterations = config.getint(tracker_section_name, "eye_iterations",
                                               fallback=tracker.eye_iterations)
        tracker.failure_detection_interval = config.getint(tracker_section_name, "failure_detection_interval",
                                                           fallback=tracker.failure_detection_interval)
        tracker.hard_failure_threshold = config.getfloat(tracker_section_name, "hard_failure_threshold",
                                                         fallback=tracker.hard_failure_threshold)
        tracker.soft_failure_threshold = config.getfloat(tracker_section_name, "soft_failure_threshold",
                                                         fallback=tracker.soft_failure_threshold)
        tracker.maximum_number_of_soft_failures = config.getint(tracker_section_name,
                                                                "maximum_number_of_soft_failures",
                                                                fallback=tracker.maximum_number_of_soft_failures)
        print("\nMulti-face tracker configured with the following settings: \n"
              "face_detection_interval = %d" % tracker.face_detection_interval + "\n"
              "face_detection_scale = %.6f" % tracker.face_detection_scale + "\n"
              "minimum_face_size = %d" % tracker.minimum_face_size + "\n"
              "overlap_threshold = %.6f" % tracker.overlap_threshold + "\n"
              "estimate_head_pose = %s" % str(tracker.estimate_head_pose) + "\n"
              "eye_iterations = %d" % tracker.eye_iterations + "\n"
              "failure_detection_interval = %d" % tracker.failure_detection_interval + "\n"
              "hard_failure_threshold = %.6f" % tracker.hard_failure_threshold + "\n"
              "soft_failure_threshold = %.6f" % tracker.soft_failure_threshold + "\n"
              "maximum_number_of_soft_failures = %d" % tracker.maximum_number_of_soft_failures)

        # Load settings for face image extraction
        extraction_section_name = "face_image_extraction"
        margin = eval(config.get(extraction_section_name, "margin", fallback="(0.225, 0.225, 0.225, 0.225)"))
        assert len(margin) == 4
        exclude_chin_points = config.getboolean(extraction_section_name, "exclude_chin_points", fallback=True)
        equalise_histogram = config.getboolean(extraction_section_name, "equalise_histogram", fallback=True)
        target_size = max(1, config.getint(extraction_section_name, "target_size", fallback=224))

        # Create the face reidentifier
        reidentifier_section_name = "ibug.face_reid.FaceReidentifier"
        vgg_model_path = read_path(config, config_folder, reidentifier_section_name, "model_path")
        try:
            gpu = config.getint(reidentifier_section_name, "gpu")
        except:
            gpu = None
        print("\nCreating face reidentifier with the following parameter...\n"
              "model_path = \"" + vgg_model_path + "\"")
        if gpu is None:
            print("gpu = None")
        else:
            print("gpu = %d" % gpu)
        reidentifier = ibug.face_reid.FaceReidentifier(vgg_model_path, gpu=gpu)
        if gpu is None or reidentifier.gpu == gpu:
            print("Face reidentifier created.")
        else:
            print("Face reidentifier created, but using CPU as fallback.")

        # Configure the reidentifier
        reidentifier.distance_threshold = config.getfloat(reidentifier_section_name, "distance_threshold",
                                                          fallback=reidentifier.distance_threshold)
        reidentifier.neighbour_count_threshold = config.getint(reidentifier_section_name,
                                                               "neighbour_count_threshold",
                                                               fallback=reidentifier.neighbour_count_threshold)
        reidentifier.quality_threshold = config.getfloat(reidentifier_section_name, "quality_threshold",
                                                         fallback=reidentifier.quality_threshold)
        reidentifier.database_capacity = config.getint(reidentifier_section_name, "database_capacity",
                                                       fallback=reidentifier.database_capacity)
        reidentifier.descriptor_list_capacity = config.getint(reidentifier_section_name, "descriptor_list_capacity",
                                                              fallback=reidentifier.descriptor_list_capacity)
        reidentifier.descriptor_update_rate = config.getfloat(reidentifier_section_name, "descriptor_update_rate",
                                                              fallback=reidentifier.descriptor_update_rate)
        mean_rgb = eval(config.get(reidentifier_section_name, "mean_rgb", fallback="()"))
        if len(mean_rgb) == 3:
            reidentifier.mean_rgb = mean_rgb
        distance_metric = config.get(reidentifier_section_name, "distance_metric",
                                     fallback=reidentifier.distance_metric)
        reidentifier.distance_metric = distance_metric.replace('\'', '').replace(
            '\"', '').replace(' ', '').replace('\t', '').lower()
        print("\nFace reidentifier configured with the following settings: \n"
              "distance_threshold = %.6f" % reidentifier.distance_threshold + "\n"
              "neighbour_count_threshold = %d" % reidentifier.neighbour_count_threshold + "\n"
              "quality_threshold = %.6f" % reidentifier.quality_threshold + "\n"
              "database_capacity = %d" % reidentifier.database_capacity + "\n"
              "descriptor_list_capacity = %d" % reidentifier.descriptor_list_capacity + "\n"
              "descriptor_update_rate = %.6f" % reidentifier.descriptor_update_rate + "\n"
              "mean_rgb = (%.6f, %.6f, %.6f)" % (reidentifier.mean_rgb[0],
                                                 reidentifier.mean_rgb[1],
                                                 reidentifier.mean_rgb[2]) + "\n"
              "distance_metric = %s" % reidentifier.distance_metric)

        # Load settings for the tracking context
        context_section_name = "tracking_context"
        face_reidentification_interval = max(1, config.getint(context_section_name, "face_reidentification_interval",
                                                              fallback=8))
        minimum_tracking_length = max(1, config.getint(context_section_name, "minimum_tracking_length", fallback=6))

        # Now open the webcam
        webcam_section_name = "cv2.VideoCapture"
        print("\nOpening webcam #%d..." % webcam_index)
        webcam = cv2.VideoCapture(webcam_index)
        if webcam.isOpened():
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, config.getint(webcam_section_name, "CAP_PROP_FRAME_WIDTH",
                                                               fallback=webcam.get(cv2.CAP_PROP_FRAME_WIDTH)))
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.getint(webcam_section_name, "CAP_PROP_FRAME_HEIGHT",
                                                                fallback=webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print("Webcam #%d opened." % webcam_index)
        else:
            print("Failed to open webcam #%d" % webcam_index)
            return

        # Start tracking!
        frame_number = 0
        tracking_context = {}
        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                   (0, 128, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255), (0, 255, 128), (255, 128, 0)]
        unidentified_face_colours = [(128, 128, 128), (192, 192, 192)]
        minimum_good_face_size = tracker.minimum_face_size / max(np.finfo(np.float).eps,
                                                                 margin[0] + 1.0 + margin[2],
                                                                 margin[1] + 1.0 + margin[3])
        print("\nFace tracking started, you may press \'Q\' to stop or \'R\' to reset...")
        while True:
            ret, frame = webcam.read()
            if ret:
                # Track the frame
                tracker.track(frame)
                tracklets = tracker.current_result

                # Update tracking context
                for tracklet_id in tracking_context.keys():
                    tracking_context[tracklet_id]['tracked'] = False
                for face in tracklets:
                    tracklet_id = face['id']
                    if tracklet_id in tracking_context:
                        tracking_context[tracklet_id]['tracking_length'] += 1
                    else:
                        tracking_context[tracklet_id] = {'tracking_length': 1, 'face_id': 0}
                    tracking_context[tracklet_id]['tracked'] = True
                    tracking_context[tracklet_id]['facial_landmarks'] = face['facial_landmarks']
                    if 'pitch' in face:
                        tracking_context[tracklet_id]['head_pose'] = (face['pitch'], face['yaw'], face['roll'])
                    else:
                        tracking_context[tracklet_id]['head_pose'] = None
                    if (face['facial_landmarks'][:, 0].min() <= 0.0 or
                            face['facial_landmarks'][:, 1].min() <= 0.0 or
                            face['facial_landmarks'][:, 0].max() >= frame.shape[1] or
                            face['facial_landmarks'][:, 1].max() >= frame.shape[0] or
                            max(face['facial_landmarks'][:, 0].max() - face['facial_landmarks'][:, 0].min(),
                                face['facial_landmarks'][:, 1].max() - face['facial_landmarks'][:, 1].min()) <
                            minimum_good_face_size):
                        tracking_context[tracklet_id]['quality'] = reidentifier.quality_threshold - 1.0
                    elif 'most_recent_fitting_scores' in face:
                        tracking_context[tracklet_id]['quality'] = np.max(face['most_recent_fitting_scores'])
                    else:
                        tracking_context[tracklet_id]['quality'] = reidentifier.quality_threshold
                for tracklet_id in list(tracking_context.keys()):
                    if not tracking_context[tracklet_id]['tracked']:
                        del tracking_context[tracklet_id]

                # Re-identify the faces
                frame_number += + 1
                if frame_number % face_reidentification_interval == 0:
                    tracklet_ids_to_be_identified = [x for x in tracking_context.keys() if
                                                     tracking_context[x]['tracking_length'] >=
                                                     minimum_tracking_length]
                    face_images = [ibug.face_reid.extract_face_image(frame,
                                                                     tracking_context[x]['facial_landmarks'],
                                                                     (target_size, target_size), margin,
                                                                     tracking_context[x]['head_pose'],
                                                                     exclude_chin_points=exclude_chin_points)[0]
                                   for x in tracklet_ids_to_be_identified]
                    qualities = [tracking_context[x]['quality'] for x in tracklet_ids_to_be_identified]
                    if equalise_histogram:
                        face_images = [ibug.face_reid.equalise_histogram(x) for x in face_images]
                    face_ids = reidentifier.reidentify_faces(face_images, tracklet_ids_to_be_identified, qualities)
                    for idx, tracklet_id in enumerate(tracklet_ids_to_be_identified):
                        tracking_context[tracklet_id]['face_id'] = face_ids[idx]
                tracked_faces = sorted([tracking_context[x]['face_id'] for x in tracking_context.keys() if
                                        tracking_context[x]['face_id'] > 0])
                if len(tracked_faces) == 0:
                    print("Frame #%d: no face is tracked." % frame_number)
                elif len(tracked_faces) == 1:
                    print("Frame #%d: face #%d is tracked." % (frame_number, tracked_faces[0]))
                else:
                    print("Frame #%d: face #" % frame_number +
                          ", #".join([str(face_id) for face_id in tracked_faces[0:-1]]) +
                          " and #%d" % tracked_faces[-1] + " are tracked.")

                # Plot the tracked faces
                for face in tracklets:
                    face_id = tracking_context[face['id']]['face_id']
                    if face_id > 0:
                        colour = colours[(face_id - 1) % len(colours)]
                        next_colour = colours[face_id % len(colours)]
                        ibug.face_tracking.FaceTracker.plot_landmark_connections(frame, face['facial_landmarks'],
                                                                                 colour=next_colour)
                        ibug.face_tracking.FaceTracker.plot_facial_landmarks(frame, face['facial_landmarks'],
                                                                             colour=colour)
                        if 'eye_landmarks' in face:
                            ibug.face_tracking.FaceTracker.plot_eye_landmarks(frame, face['eye_landmarks'],
                                                                              colour=colour)
                        if 'pitch' in face:
                            text_origin = (int(math.floor(min(face['facial_landmarks'][:, 0]))),
                                           int(math.ceil(max(face['facial_landmarks'][:, 1]))) + 16)
                            cv2.putText(frame, "Pose: [%.1f, %.1f, %.1f]" % (face['pitch'], face['yaw'], face['roll']),
                                        text_origin, cv2.FONT_HERSHEY_DUPLEX, 0.4, next_colour, lineType=cv2.LINE_AA)
                        text_origin = tuple(np.floor(np.min(face['facial_landmarks'], axis=0) + [2, -12]).astype(int))
                        cv2.putText(frame, "Face #%d" % face_id, text_origin, cv2.FONT_HERSHEY_DUPLEX,
                                    0.6, next_colour, lineType=cv2.LINE_AA)
                    else:
                        ibug.face_tracking.FaceTracker.plot_landmark_connections(frame, face['facial_landmarks'],
                                                                                 colour=unidentified_face_colours[1])
                        ibug.face_tracking.FaceTracker.plot_facial_landmarks(frame, face['facial_landmarks'],
                                                                             colour=unidentified_face_colours[0])
                        if 'eye_landmarks' in face:
                            ibug.face_tracking.FaceTracker.plot_eye_landmarks(frame, face['eye_landmarks'],
                                                                              colour=unidentified_face_colours[0])

                # Show the result
                cv2.imshow(script_name, frame)
                key = cv2.waitKey(1) % 2 ** 16
                if key == ord('q') or key == ord('Q'):
                    print("\'Q\' pressed, we are done here.")
                    break
                elif key == ord('r') or key == ord('R'):
                    print("\'R\' pressed, reset everything.")
                    reidentifier.reset()
                    tracker.reset()
            else:
                print("Failed to grab a new frame, we are done here.")
                break

        # Close the webcam
        webcam.release()


if __name__ == "__main__":
    main()
