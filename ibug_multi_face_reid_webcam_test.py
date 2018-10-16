import os
import cv2
import sys
import math
import numpy as np
import ibug_face_tracker
from face_reidentify.face_reid import FaceReID
try:
    from configparser import ConfigParser
except:
    from ConfigParser import ConfigParser


def main():
    script_name = os.path.splitext(os.path.basename('__file__'))[0]
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

        # Now open the webcam
        print("Opening webcam #%d..." % webcam_index)
        webcam = cv2.VideoCapture(webcam_index)
        if webcam.isOpened():
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, config.getint("cv2.VideoCapture", "CAP_PROP_FRAME_WIDTH",
                                                               fallback=webcam.get(cv2.CAP_PROP_FRAME_WIDTH)))
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.getint("cv2.VideoCapture", "CAP_PROP_FRAME_HEIGHT",
                                                                fallback=webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print("Webcam #%d opened." % webcam_index)
        else:
            print("Failed to open webcam #%d" % webcam_index)
            return

        # Create the multi-face tracker
        section_name = "ibug_face_tracker.MultiFaceTracker"
        ert_model_path = config.get(section_name, "ert_model_path")
        auxiliary_model_path = config.get(section_name, "auxiliary_model_path")
        if not os.path.isabs(ert_model_path):
            ert_model_path = os.path.realpath(os.path.join(os.path.dirname('__file__'), ert_model_path))
        if not os.path.isabs(auxiliary_model_path):
            auxiliary_model_path = os.path.realpath(os.path.join(os.path.dirname('__file__'), auxiliary_model_path))
        faces_to_track = config.getint(section_name, "faces_to_track", fallback=1)
        print("\nCreating multi-face tracker with the following parameters...\n"
              "ert_model_path = \"" + ert_model_path + "\"\n"
              "auxiliary_model_path = \"" + auxiliary_model_path + "\"\n"
              "faces_to_track = %d" % faces_to_track)
        tracker = ibug_face_tracker.MultiFaceTracker(ert_model_path, auxiliary_model_path, faces_to_track)
        print("Multi-face tracker created.")

        # Configure the tracker
        tracker.face_detection_interval = config.getint(section_name, "face_detection_interval",
                                                        fallback=tracker.face_detection_interval)
        tracker.face_detection_scale = config.getfloat(section_name, "face_detection_scale",
                                                       fallback=tracker.face_detection_scale)
        tracker.minimum_face_size = config.getint(section_name, "minimum_face_size",
                                                  fallback=tracker.minimum_face_size)
        tracker.overlap_threshold = config.getfloat(section_name, "overlap_threshold",
                                                    fallback=tracker.overlap_threshold)
        tracker.estimate_head_pose = config.getboolean(section_name, "estimate_head_pose",
                                                       fallback=tracker.estimate_head_pose)
        tracker.eye_iterations = config.getint(section_name, "eye_iterations",
                                               fallback=tracker.eye_iterations)
        tracker.failure_detection_interval = config.getint(section_name, "failure_detection_interval",
                                                           fallback=tracker.failure_detection_interval)
        tracker.hard_failure_threshold = config.getfloat(section_name, "hard_failure_threshold",
                                                         fallback=tracker.hard_failure_threshold)
        tracker.soft_failure_threshold = config.getfloat(section_name, "soft_failure_threshold",
                                                         fallback=tracker.soft_failure_threshold)
        tracker.maximum_number_of_soft_failures = config.getint(section_name, "maximum_number_of_soft_failures",
                                                                fallback=tracker.maximum_number_of_soft_failures)
        print("\nMulti-face tracker configured with the following settings: \n" +
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


        vggface_model_path='./models/rcmalli_vggface_tf_vgg16.h5'
        print('Starting face re-identifier ...')
        reidentifier=FaceReID(vggface_model_path)
        re_id_gap=5

        # Start tracking!
        frame_number = 1
        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                   (0, 128, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255), (0, 255, 128), (255, 128, 0)]
        print("\nFace tracking started, you may press \'Q\' to stop...")
        while True:
            ret, frame = webcam.read()
            if ret:
                # Track the frame
                tracker.track(frame)
                tracked_faces = tracker.current_result

                if np.mod(frame_number,re_id_gap)==0:
                    lm_list = [face['facial_landmarks'] for face in tracked_faces]
                    if len(lm_list)>0:
                        reidentifier.reidentify(frame,lm_list)
                        re_IDs=reidentifier.face_IDs  # the re-identified IDs

                if np.mod(frame_number,200)==0:
                    reidentifier.reset_db()


                frame_number = frame_number + 1

                # Plot the tracked faces
                for face in tracked_faces:
                    colour = colours[(face['id'] - 1) % len(colours)]
                    next_colour = colours[face['id'] % len(colours)]
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
                    frame=reidentifier.plot_reid(frame)

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
