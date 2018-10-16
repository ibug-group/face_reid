import cv2
import dlib
import numpy as np
from . import ibug_face_tracker


class MultiFaceTracker:
    def __init__(self, ert_model_path="", auxiliary_model_path="", faces_to_track=1,
                 facial_landmark_localiser=None, auxiliary_utility=None):
        if faces_to_track < 1:
            raise ValueError('faces_to_track must be set to a positive integer')
        else:
            self._face_trackers = []
            self._face_detector = dlib.get_frontal_face_detector()
            if len(ert_model_path) > 0:
                facial_landmark_localiser = ibug_face_tracker.FacialLandmarkLocaliser(ert_model_path)
            if len(auxiliary_model_path) > 0:
                auxiliary_utility = ibug_face_tracker.AuxiliaryUtility(auxiliary_model_path)
            for idx in range(faces_to_track):
                self._face_trackers.append(ibug_face_tracker.FaceTracker(facial_landmark_localiser,
                                                                         auxiliary_utility))
            self._minimum_face_size = self._face_trackers[0].minimum_face_size
            self._face_detection_scale = self._face_trackers[0].face_detection_scale
            self._face_detection_interval = self._face_trackers[0].minimum_face_detection_gap + 1
            self._face_ids = [0] * len(self._face_trackers)
            self._overlap_threshold = 0.5
            self._face_counter = 0
            self._face_detection_countdown = 0
            self._current_result = []

    @property
    def faces_to_track(self):
        return len(self._face_trackers)

    @property
    def overlap_threshold(self):
        return self._overlap_threshold

    @overlap_threshold.setter
    def overlap_threshold(self, overlap_threshold):
        self._overlap_threshold = overlap_threshold

    @property
    def facial_landmark_localiser(self):
        return self._face_trackers[0].facial_landmark_localiser

    @facial_landmark_localiser.setter
    def facial_landmark_localiser(self, facial_landmark_localiser):
        for face_tracker in self._face_trackers:
            face_tracker.facial_landmark_localiser = facial_landmark_localiser

    @property
    def auxiliary_utility(self):
        return self._face_trackers[0].auxiliary_utility

    @auxiliary_utility.setter
    def auxiliary_utility(self, auxiliary_utility):
        for face_tracker in self._face_trackers:
            face_tracker.auxiliary_utility = auxiliary_utility

    @property
    def minimum_face_size(self):
        return self._minimum_face_size

    @minimum_face_size.setter
    def minimum_face_size(self, minimum_face_size):
        for face_tracker in self._face_trackers:
            face_tracker.minimum_face_size = minimum_face_size
        self._minimum_face_size = self._face_trackers[0].minimum_face_size

    @property
    def eye_iterations(self):
        return self._face_trackers[0].eye_iterations

    @eye_iterations.setter
    def eye_iterations(self, eye_iterations):
        for face_tracker in self._face_trackers:
            face_tracker.eye_iterations = eye_iterations

    @property
    def estimate_head_pose(self):
        return self._face_trackers[0].estimate_head_pose

    @estimate_head_pose.setter
    def estimate_head_pose(self, estimate_head_pose):
        for face_tracker in self._face_trackers:
            face_tracker.estimate_head_pose = estimate_head_pose

    @property
    def face_detection_scale(self):
        return self._face_detection_scale

    @face_detection_scale.setter
    def face_detection_scale(self, face_detection_scale):
        for face_tracker in self._face_trackers:
            face_tracker.face_detection_scale = face_detection_scale
        self._face_detection_scale = self._face_trackers[0].face_detection_scale

    @property
    def soft_failure_threshold(self):
        return self._face_trackers[0].soft_failure_threshold

    @soft_failure_threshold.setter
    def soft_failure_threshold(self, soft_failure_threshold):
        for face_tracker in self._face_trackers:
            face_tracker.soft_failure_threshold = soft_failure_threshold

    @property
    def hard_failure_threshold(self):
        return self._face_trackers[0].hard_failure_threshold

    @hard_failure_threshold.setter
    def hard_failure_threshold(self, hard_failure_threshold):
        for face_tracker in self._face_trackers:
            face_tracker.hard_failure_threshold = hard_failure_threshold

    @property
    def face_detection_interval(self):
        return self._face_detection_interval

    @ face_detection_interval.setter
    def face_detection_interval(self, face_detection_interval):
        for face_tracker in self._face_trackers:
            face_tracker.minimum_face_detection_gap = face_detection_interval - 1
        self._face_detection_interval = self._face_trackers[0].minimum_face_detection_gap + 1

    @property
    def failure_detection_interval(self):
        return self._face_trackers[0].failure_detection_interval

    @failure_detection_interval.setter
    def failure_detection_interval(self, failure_detection_interval):
        for face_tracker in self._face_trackers:
            face_tracker.failure_detection_interval = failure_detection_interval

    @property
    def maximum_number_of_soft_failures(self):
        return self._face_trackers[0].maximum_number_of_soft_failures

    @maximum_number_of_soft_failures.setter
    def maximum_number_of_soft_failures(self, maximum_number_of_soft_failures):
        for face_tracker in self._face_trackers:
            face_tracker.maximum_number_of_soft_failures = maximum_number_of_soft_failures

    @property
    def number_of_landmarks(self):
        return self._face_trackers[0].number_of_landmarks

    @staticmethod
    def _face_box_from_landmarks(landmarks):
        top_left = np.floor(np.min(landmarks, axis=0))
        bottom_right = np.ceil(np.max(landmarks, axis=0))
        return dlib.rectangle(int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1]))

    @staticmethod
    def _get_overlap_ratio(box1, box2):
        if box1 is None or box2 is None or box1.is_empty() or box2.is_empty():
            return 0.0
        else:
            intersection_area = float(box1.intersect(box2).area())
            return max(intersection_area / box1.area(), intersection_area / box2.area())

    def track(self, frame, force_face_detection=False, use_bgr_colour_model=True):
        # Convert the frame to grayscale
        if frame.ndim == 3 and frame.shape[2] == 3:
            if use_bgr_colour_model:
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            grayscale_frame = frame

        # Tracking existing faces
        has_room_for_new_faces = False
        tracked_faces = [None] * len(self._face_trackers)
        for idx, face_tracker in enumerate(self._face_trackers):
            face_box = None
            if face_tracker.has_facial_landmarks:
                face_tracker.track(grayscale_frame, use_bgr_colour_model=use_bgr_colour_model)
                if face_tracker.has_facial_landmarks:
                    face_box = self._face_box_from_landmarks(face_tracker.facial_landmarks)
                    for idx2 in range(idx):
                        if self._get_overlap_ratio(face_box, tracked_faces[idx2]) > self._overlap_threshold:
                            # One of them must go
                            if face_tracker.most_recent_fitting_scores[0] > \
                                    self._face_trackers[idx2].most_recent_fitting_scores[0]:
                                self._face_trackers[idx2].reset()
                                tracked_faces[idx2] = None
                            else:
                                face_tracker.reset()
                                face_box = None
                                break
                    tracked_faces[idx] = face_box
            if face_box is None:
                has_room_for_new_faces = True

        # Detect faces if needed
        detected_faces = []
        if has_room_for_new_faces and self._face_detection_scale > 0.0 and \
                (force_face_detection or self._face_detection_countdown <= 0):
            frame_size = grayscale_frame.shape
            face_detection_frame_size = (int(max(round(frame_size[1] * self._face_detection_scale), 1)),
                                         int(max(round(frame_size[0] * self._face_detection_scale), 1)))
            if face_detection_frame_size == frame_size:
                face_detection_frame = grayscale_frame
            else:
                face_detection_frame = cv2.resize(grayscale_frame, face_detection_frame_size)
            new_faces = self._face_detector(face_detection_frame)
            detected_faces = sorted([dlib.rectangle(int(round(face_box.left() / self._face_detection_scale)),
                                                    int(round(face_box.top() / self._face_detection_scale)),
                                                    int(round(face_box.right() / self._face_detection_scale)),
                                                    int(round(face_box.bottom() / self._face_detection_scale)))
                                     for face_box in new_faces], key=dlib.rectangle.area, reverse=True)
            face_box_validity = [True] * len(detected_faces)
            for idx, face_box in enumerate(detected_faces):
                if face_box.width() < self._minimum_face_size or face_box.height() < self._minimum_face_size:
                    face_box_validity[idx] = False
                else:
                    for tracked_face in tracked_faces:
                        if self._get_overlap_ratio(face_box, tracked_face) > self._overlap_threshold:
                            face_box_validity[idx] = False
                            break
            detected_faces = [detected_faces[idx] for idx in range(len(detected_faces)) if face_box_validity[idx]]
            self._face_detection_countdown = self._face_detection_interval

        # Manage face detection countdown
        if self._face_detection_countdown > 0:
            self._face_detection_countdown = self._face_detection_countdown - 1

        # If applicable, track the new faces
        if len(detected_faces) > 0:
            for idx, face_tracker in enumerate(self._face_trackers):
                if not face_tracker.has_facial_landmarks:
                    while len(detected_faces) > 0:
                        face_tracker.track(grayscale_frame, (detected_faces[0].left(), detected_faces[0].top(),
                                                             detected_faces[0].width(), detected_faces[0].height()),
                                           use_bgr_colour_model=use_bgr_colour_model)
                        del detected_faces[0]
                        if face_tracker.has_facial_landmarks:
                            self._face_counter = self._face_counter + 1
                            self._face_ids[idx] = self._face_counter
                            break
                    if len(detected_faces) == 0:
                        break

        # Finally, construct result
        self._current_result = []
        for idx, face_tracker in enumerate(self._face_trackers):
            if face_tracker.has_facial_landmarks:
                face = {'id': self._face_ids[idx], 'facial_landmarks': face_tracker.facial_landmarks}
                if face_tracker.has_eye_landmarks:
                    face['eye_landmarks'] = face_tracker.eye_landmarks
                if face_tracker.has_head_pose:
                    face['pitch'] = face_tracker.pitch
                    face['yaw'] = face_tracker.yaw
                    face['roll'] = face_tracker.roll
                if face_tracker.has_fitting_scores:
                    face['most_recent_fitting_scores'] = face_tracker.most_recent_fitting_scores
                    face['fitting_scores_updated'] = face_tracker.fitting_scores_updated
                self._current_result.append(face)
        self._current_result.sort(key=lambda k: k['id'])

    def reset(self, reset_face_detection_countdown=True, reset_face_counter=False):
        for face_tracker in self._face_trackers:
            face_tracker.reset(reset_face_detection_countdown)
        if reset_face_detection_countdown:
            self._face_detection_countdown = 0
        if reset_face_counter:
            self._face_counter = 0

    @property
    def current_result(self):
        return self._current_result
