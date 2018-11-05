import os
import cv2
import sys
import math
import numpy as np
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def load_keras_w_tensorflow(cuda_devices=[], video_memory_utilisation=1.0):
    os.environ['KERAS_BACKEND'] = "tensorflow"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    if len(cuda_devices) == 0:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            import keras
        except:
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
            import keras
        sys.stderr.close()
        sys.stderr = original_stderr
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(["%d" % x for x in cuda_devices])
        import tensorflow as tf
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = max(0.0, min(video_memory_utilisation, 1.0))
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(config=tf_config)
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        import keras
        sys.stderr.close()
        sys.stderr = original_stderr
        keras.backend.get_session().close()
        keras.backend.set_session(session)


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

        # Parse the INI file
        config = ConfigParser()
        config.read(config_file)

        # Load keras with tensorflow
        misc_section_name = "misc"
        cuda_devices = config.get(misc_section_name, "cuda_visible_devices", fallback="").replace(
            '\'', '').replace('\"', '').replace(' ', '').replace('\t', '')
        cuda_devices = [int(x) for x in cuda_devices.split(',') if len(x) > 0]
        load_keras_w_tensorflow(cuda_devices, config.getfloat(misc_section_name,
                                                              "video_memory_utilisation",
                                                              fallback=1.0))

        # Import ibug_face_tracker
        tracker_section_name = "ibug_face_tracker.MultiFaceTracker"
        tracker_repository_path = config.get(tracker_section_name, "repository_path", fallback="").replace(
            '\'', '').replace('\"', '')
        if not os.path.isabs(tracker_repository_path):
            tracker_repository_path = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                                    tracker_repository_path))
        sys.path.append(tracker_repository_path)
        import ibug_face_tracker

        # Create the multi-face tracker
        ert_model_path = config.get(tracker_section_name, "ert_model_path").replace('\'', '').replace('\"', '')
        auxiliary_model_path = config.get(tracker_section_name, "auxiliary_model_path").replace(
            '\'', '').replace('\"', '')
        if not os.path.isabs(ert_model_path):
            ert_model_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ert_model_path))
        if not os.path.isabs(auxiliary_model_path):
            auxiliary_model_path = os.path.realpath(os.path.join(os.path.dirname(__file__), auxiliary_model_path))
        faces_to_track = config.getint(tracker_section_name, "faces_to_track", fallback=1)
        print("Creating multi-face tracker with the following parameters...\n"
              "ert_model_path = \"" + ert_model_path + "\"\n"
              "auxiliary_model_path = \"" + auxiliary_model_path + "\"\n"
              "faces_to_track = %d" % faces_to_track)
        tracker = ibug_face_tracker.MultiFaceTracker(ert_model_path, auxiliary_model_path, faces_to_track)
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
        margin = config.get(extraction_section_name, "margin", fallback="0.225,0.225,0.225,0.225,").replace(
            '\'', '').replace('\"', '').replace(' ', '').replace('\t', '')
        margin = tuple([float(x) for x in margin.split(',') if len(x) > 0])
        assert len(margin) == 4
        exclude_chin_points = config.getboolean(extraction_section_name, "exclude_chin_points", fallback=True)
        equalise_histogram = config.getboolean(extraction_section_name, "equalise_histogram", fallback=True)
        target_size = max(1, config.getint(extraction_section_name, "target_size", fallback=224))

        # Import face_reidentifier
        import face_reidentifier

        # Create the face reidentifier
        reidentifier_section_name = "face_reidentifier.FaceReidentifier"
        vgg_model_path = config.get(reidentifier_section_name, "model_path").replace('\'', '').replace('\"', '')
        print("\nCreating face reidentifier with the following parameter...\n"
              "model_path = \"" + vgg_model_path + "\"")
        reidentifier = face_reidentifier.FaceReidentifier(vgg_model_path)
        print("Face reidentifier created.")

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
        mean_rgb = config.get(reidentifier_section_name, "mean_rgb", fallback="").replace(
            '\'', '').replace('\"', '').replace(' ', '').replace('\t', '')
        mean_rgb = tuple([float(x) for x in mean_rgb.split(',') if len(x) > 0])
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
        frame_number = 1
        tracking_context = {}
        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                   (0, 128, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255), (0, 255, 128), (255, 128, 0)]
        unidentified_face_colours = [(128, 128, 128), (192, 192, 192)]
        print("\nFace tracking started, you may press \'Q\' to stop...")
        while True:
            ret, frame = webcam.read()
            if ret:
                # Track the frame
                tracker.track(frame)
                tracked_faces = tracker.current_result

                # Update tracking context
                for face_id in tracking_context.keys():
                    tracking_context[face_id]['tracked'] = False
                for face in tracked_faces:
                    if face['id'] in tracking_context:
                        tracking_context[face['id']]['tracking_length'] += 1
                    else:
                        tracking_context[face['id']] = {'tracking_length': 1, 'verified_id': 0}
                    tracking_context[face['id']]['tracked'] = True
                    tracking_context[face['id']]['facial_landmarks'] = face['facial_landmarks']
                    if 'pitch' in face:
                        tracking_context[face['id']]['head_pose'] = (face['pitch'], face['yaw'], face['roll'])
                    else:
                        tracking_context[face['id']]['head_pose'] = None
                    if 'most_recent_fitting_scores' in face:
                        if (face['facial_landmarks'][:, 0].min() <= 0.0 or
                                face['facial_landmarks'][:, 1].min() <= 0.0 or
                                face['facial_landmarks'][:, 0].max() >= frame.shape[1] or
                                face['facial_landmarks'][:, 1].max() >= frame.shape[0]):
                            tracking_context[face['id']]['quality'] = reidentifier.quality_threshold - 1.0
                        else:
                            tracking_context[face['id']]['quality'] = np.max(face['most_recent_fitting_scores'])
                for face_id in list(tracking_context.keys()):
                    if not tracking_context[face_id]['tracked']:
                        del tracking_context[face_id]

                # Re-identify the faces
                if frame_number % face_reidentification_interval == 0:
                    faces_to_be_identified = [x for x in tracking_context.keys() if
                                              tracking_context[x]['tracking_length'] >= minimum_tracking_length]
                    face_images = [face_reidentifier.extract_face_image(frame,
                                                                        tracking_context[x]['facial_landmarks'],
                                                                        (target_size, target_size), margin,
                                                                        tracking_context[x]['head_pose'],
                                                                        exclude_chin_points=exclude_chin_points)[0]
                                   for x in faces_to_be_identified]
                    if len(faces_to_be_identified) > 0 and 'quality' in tracking_context[faces_to_be_identified[0]]:
                        qualities = [tracking_context[x]['quality'] for x in faces_to_be_identified]
                    else:
                        qualities = None
                    if equalise_histogram:
                        face_images = [face_reidentifier.equalise_histogram(x) for x in face_images]
                    verified_face_ids = reidentifier.reidentify_faces(face_images, faces_to_be_identified, qualities)
                    for idx, face_id in enumerate(faces_to_be_identified):
                        tracking_context[face_id]['verified_id'] = verified_face_ids[idx]
                identified_faces = [tracking_context[x['id']]['verified_id'] for x in tracked_faces]
                identified_faces = [x for x in identified_faces if x > 0]
                if len(identified_faces) == 0:
                    print("Frame #%d: no face is tracked." % frame_number)
                elif len(identified_faces) == 1:
                    print("Frame #%d: face #%d is tracked." % (frame_number, identified_faces[0]))
                else:
                    print("Frame #%d: face #%d" % (frame_number, identified_faces[0]) +
                          ", #".join([str(face_id) for face_id in identified_faces[1:-2]]) +
                          " and #%d" % identified_faces[-1] + " are tracked.")
                frame_number = frame_number + 1

                # Plot the tracked faces
                for face in tracked_faces:
                    verified_id = tracking_context[face['id']]['verified_id']
                    if verified_id > 0:
                        colour = colours[(verified_id - 1) % len(colours)]
                        next_colour = colours[verified_id % len(colours)]
                        ibug_face_tracker.FaceTracker.plot_landmark_connections(frame, face['facial_landmarks'],
                                                                                colour=next_colour)
                        ibug_face_tracker.FaceTracker.plot_facial_landmarks(frame, face['facial_landmarks'],
                                                                            colour=colour)
                        if 'eye_landmarks' in face:
                            ibug_face_tracker.FaceTracker.plot_eye_landmarks(frame, face['eye_landmarks'],
                                                                             colour=colour)
                        if 'pitch' in face:
                            text_origin = (int(math.floor(min(face['facial_landmarks'][:, 0]))),
                                           int(math.ceil(max(face['facial_landmarks'][:, 1]))) + 16)
                            cv2.putText(frame, "Pose: [%.1f, %.1f, %.1f]" % (face['pitch'], face['yaw'], face['roll']),
                                        text_origin, cv2.FONT_HERSHEY_DUPLEX, 0.4, next_colour, lineType=cv2.LINE_AA)
                        text_origin = tuple(np.floor(np.min(face['facial_landmarks'], axis=0) + [2, -12]).astype(int))
                        cv2.putText(frame, "Face #%d" % verified_id, text_origin, cv2.FONT_HERSHEY_DUPLEX,
                                    0.6, next_colour, lineType=cv2.LINE_AA)
                    else:
                        ibug_face_tracker.FaceTracker.plot_landmark_connections(frame, face['facial_landmarks'],
                                                                                colour=unidentified_face_colours[1])
                        ibug_face_tracker.FaceTracker.plot_facial_landmarks(frame, face['facial_landmarks'],
                                                                            colour=unidentified_face_colours[0])
                        if 'eye_landmarks' in face:
                            ibug_face_tracker.FaceTracker.plot_eye_landmarks(frame, face['eye_landmarks'],
                                                                             colour=unidentified_face_colours[0])

                # Show the result
                cv2.imshow(script_name, frame)
                key = cv2.waitKey(1)
                if key == ord('q') or key == ord('Q'):
                    print("\'Q\' pressed, we are done here.")
                    break
            else:
                print("Failed to grab a new frame, we are done here.")
                break

        # Close the webcam
        webcam.release()


if __name__ == "__main__":
    main()
