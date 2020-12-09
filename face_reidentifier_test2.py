import os
import cv2
import dlib
import time
import imageio
import numpy as np
from argparse import ArgumentParser
from configparser import ConfigParser
from ibug.face_reid import FaceReidentifierEx
from utils import *


def main():
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', help='Input video path or webcam index', default=0)
    parser.add_argument('--output', '-o', help='Output file path', default=None)
    parser.add_argument('--config', '-c', help='Config file path',
                        default=os.path.splitext(os.path.realpath(__file__))[0] + '.ini')
    args = parser.parse_args()

    # Parse config
    config = ConfigParser()
    config.read(args.config)

    vid = None
    try:
        # Create dlib face detector (should be replaced by RetinaFace)
        face_detector = dlib.get_frontal_face_detector()
        print('Dlib face detector created.')

        # Create dlib key points detector (should be replaced by RetinaFace)
        landmarker_model_path = config.get('dlib.shape_predictor', 'model_path')
        if not os.path.isabs(landmarker_model_path):
            landmarker_model_path = os.path.realpath(os.path.join(os.path.dirname(args.config),
                                                                  landmarker_model_path))
        landmarker = dlib.shape_predictor(landmarker_model_path)
        print('Dlib key points detector created.')

        # Create the naive face tracker
        tracker = NaiveFaceTracker()
        tracker.iou_threshold = config.getfloat('NaiveFaceTracker', 'iou_threshold',
                                                fallback=tracker.iou_threshold)
        tracker.minimum_face_size = config.getfloat('NaiveFaceTracker', 'minimum_face_size',
                                                    fallback=tracker.minimum_face_size)
        print('Naive face tracker created.')

        # Create the face reidentifier
        # Note: All the parameters specified here are tightly linked to the embedding being used
        reidentifier_section_name = 'ibug.face_reid.FaceReidentifierEx'
        reidentifier = FaceReidentifierEx(distance_metric='euclidean', mean_rgb=(129.1863, 104.7624, 93.5940),
                                          std_rgb=(1.0, 1.0, 1.0), normalised_face_size=224, equalise_histogram=True,
                                          gpu=eval(config.get(reidentifier_section_name, 'gpu', fallback='None')))

        # Note: These parameters should be changed once we switch to RetinaFae, specifically:
        reidentifier.margin_dim = 0
        reidentifier.face_margin = (0.225, 0.225, 0.225, 0.225)
        reidentifier.exclude_chin_points = True

        # Note: These are the main parameters controlling the behaviour of the re-id algorithm
        reidentifier.distance_threshold = config.getfloat(reidentifier_section_name, 'distance_threshold',
                                                          fallback=reidentifier.distance_threshold)
        reidentifier.neighbour_count_threshold = config.getint(reidentifier_section_name,
                                                               'neighbour_count_threshold',
                                                               fallback=reidentifier.neighbour_count_threshold)
        reidentifier.quality_threshold = config.getfloat(reidentifier_section_name, 'quality_threshold',
                                                         fallback=reidentifier.quality_threshold)
        reidentifier.database_capacity = config.getint(reidentifier_section_name, 'database_capacity',
                                                       fallback=reidentifier.database_capacity)
        reidentifier.descriptor_list_capacity = config.getint(reidentifier_section_name, 'descriptor_list_capacity',
                                                              fallback=reidentifier.descriptor_list_capacity)
        reidentifier.descriptor_update_rate = config.getfloat(reidentifier_section_name, 'descriptor_update_rate',
                                                              fallback=reidentifier.descriptor_update_rate)
        reidentifier.reidentification_interval = config.getint(reidentifier_section_name, 'reidentification_interval',
                                                               fallback=reidentifier.reidentification_interval)
        reidentifier.minimum_tracklet_length = config.getint(reidentifier_section_name, 'minimum_tracklet_length',
                                                             fallback=reidentifier.minimum_tracklet_length)
        reidentifier.minimum_face_size = (tracker.minimum_face_size /
                                          max(np.finfo(np.float).eps,
                                              reidentifier.face_margin[0] + 1.0 + reidentifier.face_margin[2],
                                              reidentifier.face_margin[1] + 1.0 + reidentifier.face_margin[3]))
        print('Face reidentifier created.')

        # Open the input video
        using_webcam = not os.path.exists(args.input)
        vid = cv2.VideoCapture(int(args.input) if using_webcam else args.input)
        assert vid.isOpened()
        if using_webcam:
            print(f'Webcam #{int(args.input)} opened.')
        else:
            print(f'Input video "{args.input}" opened.')

        # Process the frames
        frame_number = 0
        window_title = os.path.splitext(os.path.basename(__file__))[0]
        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                   (0, 128, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255), (0, 255, 128), (255, 128, 0)]
        unidentified_face_colours = [(128, 128, 128), (192, 192, 192)]
        print('Processing started, press \'Q\' to quit.')
        while True:
            # Get a new frame
            _, frame = vid.read()
            if frame is None:
                break
            else:
                # Face and key points detection (should be replaced by RetinaFace)
                # Note: For re-id to work, a 'facial_landmarks' field should exist for each face. The landmarks will be
                #       used to calculate the face's bounding box so that the face patch could be cropped out and sent
                #       to the face recognition model to get the face descriptor. When using RetinaFace, we should use
                #       the 5 key points returned by the model. These landmarks don't need to be very accurate. But if
                #       RetinaFace's 5 key points are too far off, we can just put the 4 corners of the face bbox here.
                faces = detect_faces_and_key_points(face_detector, landmarker,
                                                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

                # Face tracking
                # Note: The tracker will give an 'id' (referred to as tracklet id) to each tracked face. This field is
                #       needed by the re-id algorithm. Note that the tracker may not award an 'id' to a face if it's
                #       too small, in which case the face should not be sent to the re-id algorithm
                faces = tracker.track(faces)
                tracked_faces = [face for face in faces if 'id' in face]

                # Head pose estimation
                # Note: Head pose estimation is need for two purposes: 1) To get the 'roll' (in-plane rotation) of the
                #       face so that it could be compensated for when cropping the face patch. Because face embeddings
                #       are usually not rotational-invariant, compensating for in-plane rotation could increase the
                #       model's accuracy. 2) To estimate 'pitch' and 'yaw' (out-of-plane rotations) of the faces so
                #       that those that are 'too non-frontal' will not be used by the face recognition model. This will
                #       decrease the number of false positives given by the re-id algorithm. Here we only perform roll
                #       estimation. Pitch and yaw estimation needs to be worked on later for the 5 landmarks given by
                #       RetinaFace.
                for face in tracked_faces:
                    pass

                # Face re-id
                # Note: The re-id algorithm will give a 'face_id' to each face. A value of 0 means the face has not
                #       been identified (yet).
                start_time = time.time()
                identities = reidentifier.reidentify_tracked_faces(frame, tracked_faces)
                elapsed_time = time.time() - start_time

                valid_face_ids = [identities[x]['face_id'] for x in identities.keys() if
                                  identities[x]['face_id'] > 0]
                if len(valid_face_ids) == 0:
                    print('Frame #%d processed in %.04f ms: no face is tracked.' %
                          (frame_number, elapsed_time * 1000.0))
                elif len(valid_face_ids) == 1:
                    print('Frame #%d processed in %.04f ms: face #%d is tracked.' %
                          (frame_number, elapsed_time * 1000.0, valid_face_ids[0]))
                else:
                    print('Frame #%d processed in %.04f ms: face #' % (frame_number, elapsed_time * 1000.0) +
                          ', #'.join([str(face_id) for face_id in valid_face_ids[0:-1]]) +
                          ' and #%d' % valid_face_ids[-1] + ' are tracked.')

                # Rendering
                for face in faces:
                    if 'id' in face:
                        face_id = identities[face['id']]['face_id']
                    else:
                        face_id = 0
                    if face_id > 0:
                        colour = colours[(face_id - 1) % len(colours)]
                        cv2.rectangle(frame, (int(face['bbox']['x1']), int(face['bbox']['y1'])),
                                      (int(face['bbox']['x2']), int(face['bbox']['y2'])), color=colour, thickness=2)
                        next_colour = colours[face_id % len(colours)]
                        plot_landmarks(frame, face['facial_landmarks'], next_colour, colour)
                        cv2.putText(frame, 'Face #%d' % face_id, (int(face['bbox']['x1']),
                                                                  int(face['bbox']['y1']) - 12),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6, colour, lineType=cv2.LINE_AA)
                    else:
                        cv2.rectangle(frame, (int(face['bbox']['x1']), int(face['bbox']['y1'])),
                                      (int(face['bbox']['x2']), int(face['bbox']['y2'])),
                                      color=unidentified_face_colours[0], thickness=2)
                        plot_landmarks(frame, face['facial_landmarks'], unidentified_face_colours[1],
                                       unidentified_face_colours[0])

                # Display the frame
                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) % 2 ** 16
                if key == ord('q') or key == ord('Q'):
                    print('\'Q\' pressed, we are done here.')
                    break
                else:
                    frame_number += 1
    finally:
        if vid is not None:
            vid.release()
        print('All done.')


if __name__ == '__main__':
    main()
