import numpy as np
from scipy.optimize import linear_sum_assignment


__all__ = ['NaiveFaceTracker']


class NaiveFaceTracker(object):
    def __init__(self, iou_threshold=0.4, minimum_face_size=0.0):
        self._iou_threshold = iou_threshold
        self._minimum_face_size = minimum_face_size
        self._tracklets = []
        self._tracklet_counter = 0

    @property
    def iou_threshold(self):
        return self._iou_threshold

    @iou_threshold.setter
    def iou_threshold(self, threshold):
        self._iou_threshold = threshold

    @property
    def minimum_face_size(self):
        return self._minimum_face_size

    @minimum_face_size.setter
    def minimum_face_size(self, face_size):
        self._minimum_face_size = face_size

    @staticmethod
    def get_iou(bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        Returns
        -------
        float
            in [0, 1]
        """

        # determine the coordinates of the intersection rectangle
        x_left = max(min(bb1['x1'], bb1['x2']), min(bb2['x1'], bb2['x2']))
        y_top = max(min(bb1['y1'], bb1['y2']), min(bb2['y1'], bb2['y2']))
        x_right = min(max(bb1['x2'], bb1['x1']), max(bb2['x2'], bb2['x1']))
        y_bottom = min(max(bb1['y2'], bb1['y1']), max(bb2['y2'], bb2['y1']))

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = abs((bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1']))
        bb2_area = abs((bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1']))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        return intersection_area / float(bb1_area + bb2_area - intersection_area)

    def track(self, faces):
        # Copy the list
        faces = faces[:]

        # Calculate area of the faces
        face_areas = []
        for face in faces:
            if 'bbox' in face:
                face_areas.append(abs((face['bbox']['x2'] - face['bbox']['x1']) *
                                      (face['bbox']['y2'] - face['bbox']['y1'])))
            else:
                face_areas.append(-1.0)

        # Prepare tracklets
        for tracklet in self._tracklets:
            tracklet['tracked'] = False

        # Calculate the distance matrix based on IOU
        iou_distance_threshold = np.clip(1.0 - self._iou_threshold, 0.0, 1.0)
        min_face_area = max(self._minimum_face_size ** 2, np.finfo(float).eps)
        distances = np.full(shape=(len(faces), len(self._tracklets)),
                            fill_value=2.0 * min(len(faces), len(self._tracklets)), dtype=float)
        for row, face in enumerate(faces):
            if face_areas[row] >= min_face_area:
                for col, tracklet in enumerate(self._tracklets):
                    distance = 1.0 - self.get_iou(face['bbox'], tracklet['bbox'])
                    if distance <= iou_distance_threshold:
                        distances[row, col] = distance

        # ID assignment
        for row, col in zip(*linear_sum_assignment(distances)):
            if distances[row, col] <= iou_distance_threshold:
                faces[row]['id'] = self._tracklets[col]['id']
                self._tracklets[col]['bbox'] = faces[row]['bbox']
                self._tracklets[col]['tracked'] = True

        # Remove expired tracklets
        self._tracklets = [x for x in self._tracklets if x['tracked']]

        # Register new faces
        for idx, face in enumerate(faces):
            if face_areas[idx] >= min_face_area and 'id' not in face:
                self._tracklet_counter += 1
                self._tracklets.append({'bbox': face['bbox'], 'id': self._tracklet_counter, 'tracked': True})
                face['id'] = self._tracklets[-1]['id']

        return faces

    def reset(self):
        self._tracklets = []
        self._tracklet_counter = 0
