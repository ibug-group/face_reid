import os
import cv2
import numpy as np


__all__ = ['RetinaFacePoseEstimator']


class RetinaFacePoseEstimator(object):
    def __init__(self, mean_shape_path=os.path.join(os.path.dirname(__file__),
                                                    'data', 'bfm_lms.npy')):
        # Load the 68-point mean shape derived from BFM
        mean_shape = np.load(mean_shape_path)

        # Calculate the 5-points mean shape for RetinaFace
        left_eye = mean_shape[[37, 38, 40, 41]].mean(axis=0)
        right_eye = mean_shape[[43, 44, 46, 47]].mean(axis=0)
        self._retina_mean_shape = np.vstack((left_eye, right_eye, mean_shape[[30, 48, 54]]))

        # Flip the y coordinates of the mean shape to match that of the image coordinate system
        self._retina_mean_shape[:, 1] = -self._retina_mean_shape[:, 1]

    def estimate_head_pose(self, landmarks, image_width, image_height):
        # Form the camera matrix
        camera_matrix = np.array([[image_width + image_height, 0, image_width / 2.0],
                                  [0, image_width + image_height, image_height / 2.0],
                                  [0, 0, 1]], dtype=float)

        # Prepare the landmarks
        if landmarks.shape[0] == 68:
            landmarks = landmarks[17:]
        if landmarks.shape[0] != 5:
            left_eye = landmarks[[20, 21, 23, 24]].mean(axis=0)
            right_eye = landmarks[[26, 27, 29, 30]].mean(axis=0)
            landmarks = np.vstack((left_eye, right_eye, landmarks[[13, 31, 37]]))

        # EPnP
        _, rotation, _ = cv2.solvePnP(self._retina_mean_shape, np.expand_dims(landmarks, axis=1),
                                      camera_matrix, None, flags=cv2.SOLVEPNP_EPNP)
        return tuple((rotation[:, 0] * [-1, 1, 1] / np.pi * 180.0).tolist())
