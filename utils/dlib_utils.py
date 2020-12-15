import cv2
import numpy as np


__all__ = ['detect_faces_and_key_points', 'plot_landmarks']


def detect_faces_and_key_points(face_detector, landmarker, image, use_retina_face_landmarks=False):
    faces = []
    for bbox in face_detector(image):
        landmarks = landmarker(image, bbox)
        faces.append({'bbox': {'x1': float(bbox.left()), 'y1': float(bbox.top()),
                               'x2': float(bbox.right()), 'y2': float(bbox.bottom())},
                     'facial_landmarks': np.array([[landmarks.part(idx).x, landmarks.part(idx).y]
                                                   for idx in range(landmarks.num_parts)], dtype=float)})
        if use_retina_face_landmarks:
            faces[-1]['facial_landmarks'] = np.vstack((faces[-1]['facial_landmarks'][[37, 38, 40, 41]].mean(axis=0),
                                                       faces[-1]['facial_landmarks'][[43, 44, 46, 47]].mean(axis=0),
                                                       faces[-1]['facial_landmarks'][[30, 48, 54]]))
    return faces


def plot_landmarks(frame, landmarks, connection_colour=(0, 255, 0), landmark_colour=(0, 0, 255)):
    if landmarks.shape[0] == 5:
        for landmark in landmarks:
            cv2.circle(frame, tuple(landmark.astype(int).tolist()), 3, landmark_colour, -1)
    else:
        for idx in range(len(landmarks) - 1):
            if (idx != 16 and idx != 21 and idx != 26 and idx != 30 and
                    idx != 35 and idx != 41 and idx != 47 and idx != 59):
                cv2.line(frame, tuple(landmarks[idx].astype(int).tolist()),
                         tuple(landmarks[idx + 1].astype(int).tolist()),
                         color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
            if idx == 30:
                cv2.line(frame, tuple(landmarks[30].astype(int).tolist()),
                         tuple(landmarks[33].astype(int).tolist()),
                         color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
            elif idx == 36:
                cv2.line(frame, tuple(landmarks[36].astype(int).tolist()),
                         tuple(landmarks[41].astype(int).tolist()),
                         color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
            elif idx == 42:
                cv2.line(frame, tuple(landmarks[42].astype(int).tolist()),
                         tuple(landmarks[47].astype(int).tolist()),
                         color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
            elif idx == 48:
                cv2.line(frame, tuple(landmarks[48].astype(int).tolist()),
                         tuple(landmarks[59].astype(int).tolist()),
                         color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
            elif idx == 60:
                cv2.line(frame, tuple(landmarks[60].astype(int).tolist()),
                         tuple(landmarks[67].astype(int).tolist()),
                         color=connection_colour, thickness=1, lineType=cv2.LINE_AA)
        for idx, landmark in enumerate(landmarks):
            if idx in [30, 48, 54]:
                cv2.circle(frame, tuple(landmark.astype(int).tolist()), 3, landmark_colour, -1)
            else:
                cv2.circle(frame, tuple(landmark.astype(int).tolist()), 1, landmark_colour, -1)
        cv2.circle(frame, tuple(landmarks[[37, 38, 40, 41]].mean(axis=0).astype(int).tolist()), 3, landmark_colour, -1)
        cv2.circle(frame, tuple(landmarks[[43, 44, 46, 47]].mean(axis=0).astype(int).tolist()), 3, landmark_colour, -1)
