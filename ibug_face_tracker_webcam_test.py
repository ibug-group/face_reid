import os
import cv2
import sys
import numpy as np
import ibug_face_tracker


def main():
    if len(sys.argv) >= 2:
        webcam_id = int(sys.argv[1])
    else:
        webcam_id = 0
    webcam = cv2.VideoCapture(webcam_id)
    if not webcam.isOpened():
        print("Failed to open the webcam #%d." % webcam_id)
    else:
        print("Webcam #%d opened." % webcam_id)
        model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        tracker = ibug_face_tracker.FaceTracker(os.path.join(model_folder, "new3_68_pts_UAD_1_tr_6_cas_15.dat"),
                                                os.path.join(model_folder, "additional_svrs.model"))
        tracker.minimum_face_size = 80
        tracker.minimum_face_detection_gap = 2
        print("Tracker initialised.")
        print(tracker)
        utility = ibug_face_tracker.AuxiliaryUtility(os.path.join(model_folder, "additional_svrs.model"))
        print("Auxiliary utility initialised.")
        while True:
            ret, frame = webcam.read()
            if ret:
                tracker.track(frame)
                if tracker.has_facial_landmarks:
                    facial_landmarks = tracker.facial_landmarks
                    eye_landmarks = tracker.eye_landmarks
                    pitch, yaw, roll = utility.estimate_head_pose(facial_landmarks)
                    print("Pitch: %f / %f, yaw: %f / %f, roll: %f / %f." % (pitch, tracker.pitch,
                                                                            yaw, tracker.yaw,
                                                                            roll, tracker.roll))
                    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    eye_points, fitting_scores = utility.apply_additional_svrs(
                        grayscale_frame, facial_landmarks, calculate_fitting_scores=tracker.fitting_scores_updated)
                    print("Left eye: %s / %s, right eye: %s / %s" % (
                        np.array2string(np.mean(eye_points[1:6, :], axis=0)),
                        np.array2string(np.mean(eye_landmarks[1:6, :], axis=0)),
                        np.array2string(np.mean(eye_points[8:14, :], axis=0)),
                        np.array2string(np.mean(eye_landmarks[8:14, :], axis=0))))
                    if fitting_scores is not None:
                        print("Fitting scores: %s / %s." % (np.array2string(fitting_scores),
                                                            np.array2string(tracker.most_recent_fitting_scores)))
                tracker.plot_current_result(frame)
                cv2.imshow("ibug_face_tracker_webcam_test", frame)
            key = cv2.waitKey(33)
            if key == ord('q') or key == ord('Q'):
                break
        webcam.release()


if __name__ == "__main__":
    main()
