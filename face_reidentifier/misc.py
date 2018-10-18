import cv2
import numpy as np
from keras import backend as K
from keras.utils import layer_utils
from keras.layers import Flatten, Dense, Input, Activation, Conv2D, MaxPooling2D
from keras.models import Model


def load_vgg_face_16_model(weights_path, classes=2622):

    img_input = Input(shape=(224,224,3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(
        x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(
        x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(
        x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, name='fc6')(x)
    x = Activation('relu', name='fc6/relu')(x)
    x = Dense(4096, name='fc7')(x)
    x = Activation('relu', name='fc7/relu')(x)
    x = Dense(classes, name='fc8')(x)
    x = Activation('softmax', name='fc8/softmax')(x)

    inputs = img_input

    # Create model
    model = Model(inputs, x, name='vggface_vgg16')

    # if weights_path is not None:
    model.load_weights(weights_path, by_name=True) # load weights

    if K.backend() == 'theano':
        layer_utils.convert_all_kernels_in_model(model)

    return model


def equalise_histogram(image, use_bgr_colour_model=True):
    if image.ndim == 3 and image.shape[2] == 3:
        if use_bgr_colour_model:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
            hsv_image[:, :, 0] = cv2.equalizeHist(hsv_image[:, :, 0])
            return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR_FULL)
        else:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
            hsv_image[:, :, 0] = cv2.equalizeHist(hsv_image[:, :, 0])
            return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB_FULL)
    else:
        return cv2.equalizeHist(image)


def get_face_box(landmarks, margin=(0.0, 0.0, 0.0, 0.0), exclude_chin_points=False):
    # Get bounding box
    if exclude_chin_points and landmarks.shape[0] == 68:
        top_left = np.min(landmarks[17:], axis=0)
        bottom_right = np.max(landmarks[17:], axis=0)
    else:
        top_left = np.min(landmarks, axis=0)
        bottom_right = np.max(landmarks, axis=0)
    face_size = bottom_right - top_left
    top_left[0] = np.floor(top_left[0] - face_size[0] * margin[0])
    top_left[1] = np.floor(top_left[1] - face_size[1] * margin[1])
    bottom_right[0] = np.ceil(bottom_right[0] + face_size[0] * margin[2])
    bottom_right[1] = np.ceil(bottom_right[1] + face_size[1] * margin[3])

    # Make face box square
    difference = (bottom_right[1] - top_left[1] + 1) - (bottom_right[0] - top_left[0] + 1)
    if difference > 0:
        top_left[0] -= difference // 2
        bottom_right[0] += difference - difference // 2
    elif difference < 0:
        difference = -difference
        top_left[1] -= difference // 2
        bottom_right[1] += difference - difference // 2

    return top_left, bottom_right


def extract_face_image(image, landmarks, target_size, margin, head_pose=None,
                       eye_points=None, exclude_chin_points=False,
                       interpolation=cv2.INTER_CUBIC):
    # First, see whether we have head pose
    if head_pose is not None and len(head_pose) >= 3:
        roll = head_pose[2]
    else:
        roll = None

    # Get a proper bounding box for the face
    if margin is None:
        margin = (0.0, 0.0, 0.0, 0.0)
    if roll is None:
        top_left, bottom_right = get_face_box(landmarks, margin, exclude_chin_points)
    else:
        rotated_landmarks = cv2.getRotationMatrix2D((0, 0), roll, 1.0).dot(
            np.vstack((landmarks.T, np.ones(landmarks.shape[0])))).T
        top_left, bottom_right = get_face_box(rotated_landmarks, margin, exclude_chin_points)
        corners = np.array([[top_left[0], top_left[1], 1],
                            [bottom_right[0], top_left[1], 1],
                            [bottom_right[0], bottom_right[1], 1],
                            [top_left[0], bottom_right[1], 1]])
        top_left, bottom_right = get_face_box(cv2.getRotationMatrix2D(
            (0, 0), -roll, 1.0).dot(corners.T).T, (0, 0, 0, 0), exclude_chin_points)
        top_left -= 5
        bottom_right += 5

    # Enlarge the image if necessary
    padding = np.zeros((image.ndim, 2), dtype=int)
    if top_left[0] < 0:
        padding[1][0] = -int(top_left[0])
    if top_left[1] < 0:
        padding[0][0] = -int(top_left[1])
    if bottom_right[0] >= image.shape[1]:
        padding[1][1] = int(bottom_right[0] - image.shape[1] + 1)
    if bottom_right[1] >= image.shape[0]:
        padding[0][1] = int(bottom_right[1] - image.shape[0] + 1)
    image = np.pad(image, padding, 'symmetric')

    # Revise bounding box and landmarks accordingly
    landmarks = landmarks.copy()
    if eye_points is not None:
        eye_points = eye_points.copy()
    if top_left[0] < 0:
        bottom_right[0] -= top_left[0]
        landmarks[:, 0] -= top_left[0]
        if eye_points is not None:
            eye_points[:, 0] -= top_left[0]
        top_left[0] = 0
    if top_left[1] < 0:
        bottom_right[1] -= top_left[1]
        landmarks[:, 1] -= top_left[1]
        if eye_points is not None:
            eye_points[:, 1] -= top_left[1]
        top_left[1] = 0

    # Extract the face image
    face_image = image[int(top_left[1]): int(bottom_right[1] + 1),
                       int(top_left[0]): int(bottom_right[0] + 1)]
    landmarks[:, 0] -= top_left[0]
    landmarks[:, 1] -= top_left[1]
    if eye_points is not None:
        eye_points[:, 0] -= top_left[0]
        eye_points[:, 1] -= top_left[1]

    # If head pose is given, rotate everything
    if roll is not None:
        rotation_matrix = cv2.getRotationMatrix2D(((face_image.shape[1] - 1) / 2.0,
                                                   (face_image.shape[0] - 1) / 2.0), roll, 1.0)
        face_image = cv2.warpAffine(face_image, rotation_matrix,
                                    dsize=(face_image.shape[1], face_image.shape[0]),
                                    flags=(interpolation + cv2.WARP_FILL_OUTLIERS))
        landmarks = rotation_matrix.dot(np.vstack((landmarks.T, np.ones(landmarks.shape[0])))).T
        if eye_points is not None:
            eye_points = rotation_matrix.dot(np.vstack((eye_points.T, np.ones(eye_points.shape[0])))).T
        top_left, bottom_right = get_face_box(landmarks, margin, exclude_chin_points)
        if top_left[0] > 0:
            landmarks[:, 0] -= top_left[0]
            if eye_points is not None:
                eye_points[:, 0] -= top_left[0]
        if top_left[1] > 0:
            landmarks[:, 1] -= top_left[1]
            if eye_points is not None:
                eye_points[:, 1] -= top_left[1]
        face_image = face_image[max(int(top_left[1]), 0): min(int(bottom_right[1] + 1), face_image.shape[0]),
                                max(int(top_left[0]), 0): min(int(bottom_right[0] + 1), face_image.shape[1])]

    # Rescale landmarks and face image
    if target_size is not None and target_size[0] > 0 and target_size[1] > 0:
        landmarks[:, 0] *= float(target_size[0]) / max(face_image.shape[1], 1)
        landmarks[:, 1] *= float(target_size[1]) / max(face_image.shape[0], 1)
        if eye_points is not None:
            eye_points[:, 0] *= float(target_size[0]) / max(face_image.shape[1], 1)
            eye_points[:, 1] *= float(target_size[1]) / max(face_image.shape[0], 1)
        face_image = cv2.resize(face_image, target_size, interpolation=interpolation)
    else:
        face_image = face_image.copy()

    return face_image, landmarks, eye_points
