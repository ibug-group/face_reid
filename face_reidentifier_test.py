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
        print("\nMulti-face tracker configured with the following settings:" +
              "\nface_detection_interval = %d" % tracker.face_detection_interval +
              "\nface_detection_scale = %.6f" % tracker.face_detection_scale +
              "\nminimum_face_size = %d" % tracker.minimum_face_size +
              "\noverlap_threshold = %.6f" % tracker.overlap_threshold +
              "\nestimate_head_pose = %r" % tracker.estimate_head_pose +
              "\neye_iterations = %d" % tracker.eye_iterations +
              "\nfailure_detection_interval = %d" % tracker.failure_detection_interval +
              "\nhard_failure_threshold = %.6f" % tracker.hard_failure_threshold +
              "\nsoft_failure_threshold = %.6f" % tracker.soft_failure_threshold +
              "\nmaximum_number_of_soft_failures = %d" % tracker.maximum_number_of_soft_failures)

        # Create the face reidentifier
        reidentifier_section_name = "ibug.face_reid.FaceReidentifierEx"
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
        reidentifier = ibug.face_reid.FaceReidentifierEx(model_path=vgg_model_path, gpu=gpu)
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
        reidentifier.face_margin = eval(config.get(reidentifier_section_name, "face_margin",
                                                   fallback="(0.225, 0.225, 0.225, 0.225)"))
        reidentifier.exclude_chin_points = config.getboolean(reidentifier_section_name,
                                                             "exclude_chin_points", fallback=True)
        reidentifier.equalise_histogram = config.getboolean(reidentifier_section_name,
                                                            "equalise_histogram", fallback=True)
        reidentifier.normalised_face_size = config.getint(reidentifier_section_name,
                                                          "normalised_face_size", fallback=224)
        reidentifier.reidentification_interval = config.getint(reidentifier_section_name,
                                                               "reidentification_interval", fallback=8)
        reidentifier.minimum_tracklet_length = config.getint(reidentifier_section_name,
                                                             "minimum_tracklet_length", fallback=6)
        reidentifier.minimum_face_size = (tracker.minimum_face_size /
                                          max(np.finfo(np.float).eps,
                                              reidentifier.face_margin[0] + 1.0 + reidentifier.face_margin[2],
                                              reidentifier.face_margin[1] + 1.0 + reidentifier.face_margin[3]))
        print("\nFace reidentifier configured with the following settings:"
              "\ndistance_threshold = %.6f" % reidentifier.distance_threshold +
              "\nneighbour_count_threshold = %d" % reidentifier.neighbour_count_threshold +
              "\nquality_threshold = %.6f" % reidentifier.quality_threshold +
              "\ndatabase_capacity = %d" % reidentifier.database_capacity +
              "\ndescriptor_list_capacity = %d" % reidentifier.descriptor_list_capacity +
              "\ndescriptor_update_rate = %.6f" % reidentifier.descriptor_update_rate +
              "\nmean_rgb = (%.6f, %.6f, %.6f)" % (reidentifier.mean_rgb[0],
                                                   reidentifier.mean_rgb[1],
                                                   reidentifier.mean_rgb[2]) +
              "\ndistance_metric = %s" % reidentifier.distance_metric +
              "\nface_margin = (%.3f, %.3f, %.3f, %.3f)" % (reidentifier.face_margin[0],
                                                            reidentifier.face_margin[1],
                                                            reidentifier.face_margin[2],
                                                            reidentifier.face_margin[3]) +
              "\nexclude_chin_points = %r" % reidentifier.exclude_chin_points +
              "\nequalise_histogram = %r" % reidentifier.equalise_histogram +
              "\nnormalised_face_size = %d" % reidentifier.normalised_face_size +
              "\nreidentification_interval = %d" % reidentifier.reidentification_interval +
              "\nminimum_tracklet_length = %d" % reidentifier.minimum_tracklet_length +
              "\nminimum_face_size = %.6f" % reidentifier.minimum_face_size)

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
        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                   (0, 128, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255), (0, 255, 128), (255, 128, 0)]
        unidentified_face_colours = [(128, 128, 128), (192, 192, 192)]
        print("\nFace tracking started, you may press \'Q\' to stop or \'R\' to reset...")
        while True:
            ret, frame = webcam.read()
            if ret:
                # Track the frame
                tracker.track(frame)
                tracklets = tracker.current_result

                # Reidentify the faces
                frame_number += 1
                identities = reidentifier.reidentify_tracked_faces(frame, tracklets)
                valid_face_ids = [identities[x]['face_id'] for x in identities.keys() if
                                  identities[x]['face_id'] > 0]
                if len(valid_face_ids) == 0:
                    print("Frame #%d: no face is tracked." % frame_number)
                elif len(valid_face_ids) == 1:
                    print("Frame #%d: face #%d is tracked." % (frame_number, valid_face_ids[0]))
                else:
                    print("Frame #%d: face #" % frame_number +
                          ", #".join([str(face_id) for face_id in valid_face_ids[0:-1]]) +
                          " and #%d" % valid_face_ids[-1] + " are tracked.")

                # Plot the tracked faces
                for face in tracklets:
                    face_id = identities[face['id']]['face_id']
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
                        if 'pitch' in face and 'yaw' in face and 'roll' in face:
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
