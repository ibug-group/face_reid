import os
import numpy as np


__all__ = ['HeadPoseEstimator']


class HeadPoseEstimator(object):
    def __init__(self, model_path=os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                                'data', 'head_pose.model'))):
        with open(model_path, 'r') as model_file:
            model_content = [float(x) for x in model_file.read().split(' ') if len(x) > 0]
            offset = 0
            self._coefficients, offset = self._parse_matrix(model_content, offset)
            self._mean_shape, offset = self._parse_matrix(model_content, offset)
            offset += 1
            mean, offset = self._parse_matrix(model_content, offset)
            _, offset = self._parse_matrix(model_content, offset)
            weights, offset = self._parse_matrix(model_content, offset)
            self._yaw_regressor = (mean, weights.reshape((1, -1), order='F')[0])
            mean, offset = self._parse_matrix(model_content, offset)
            _, offset = self._parse_matrix(model_content, offset)
            weights, offset = self._parse_matrix(model_content, offset)
            self._pitch_regressor = (mean, weights.reshape((1, -1), order='F')[0])

    def estimate_head_pose(self, landmarks):
        if landmarks.shape[0] == 68:
            landmarks = np.vstack((landmarks[17: 60, :], landmarks[61: 64, :], landmarks[65:, :]))
        elif landmarks.shape[0] == 51:
            landmarks = np.vstack((landmarks[: 43, :], landmarks[44: 47, :], landmarks[48:, :]))
        aligned_upper_face = self._apply_rigid_alignment_parameters(
            landmarks[0: 31, :], *self._compute_rigid_alignment_parameters(landmarks[0: 31, :],
                                                                           self._yaw_regressor[0]))
        yaw = (np.dot(aligned_upper_face.reshape((1, -1), order='F'), self._yaw_regressor[1][0: -1]) +
               self._yaw_regressor[1][-1])[0] * 1.25
        pitch = (np.dot(aligned_upper_face.reshape((1, -1), order='F'), self._pitch_regressor[1][0: -1]) +
                 self._pitch_regressor[1][-1])[0] * 1.25
        scos, ssin, _, _ = self._compute_rigid_alignment_parameters(self._mean_shape, landmarks)
        roll = np.degrees(np.arctan2(ssin, scos))
        return tuple(np.dot(np.array([pitch, yaw, roll]), self._coefficients).tolist())

    @staticmethod
    def _parse_matrix(model_content, offset):
        shape = (int(model_content[offset]), int(model_content[offset + 1]))
        offset += 2
        matrix = np.array(model_content[offset: offset + shape[0] * shape[1]]).reshape(shape, order='F')
        return matrix, offset + shape[0] * shape[1]

    @staticmethod
    def _compute_rigid_alignment_parameters(source, destination):
        a = np.zeros((4, 4), np.float)
        b = np.zeros((4, 1), np.float)
        a[0, 0] = np.square(source[:, 0]).sum() + np.square(source[:, 1]).sum()
        a[0, 2] = source[:, 0].sum()
        a[0, 3] = source[:, 1].sum()
        b[0] = np.sum(source[:, 0] * destination[:, 0] + source[:, 1] * destination[:, 1])
        b[1] = np.sum(source[:, 0] * destination[:, 1] - source[:, 1] * destination[:, 0])
        b[2] = destination[:, 0].sum()
        b[3] = destination[:, 1].sum()
        a[1, 1] = a[0, 0]
        a[3, 0] = a[0, 3]
        a[1, 2] = a[2, 1] = -a[0, 3]
        a[1, 3] = a[3, 1] = a[2, 0] = a[0, 2]
        a[2, 2] = a[3, 3] = source.shape[0]
        return tuple(np.dot(np.linalg.pinv(a), b).T[0].tolist())

    @staticmethod
    def _apply_rigid_alignment_parameters(source, scos, ssin, transx, transy):
        return np.dot(source, [[scos, ssin], [-ssin, scos]]) + [transx, transy]
