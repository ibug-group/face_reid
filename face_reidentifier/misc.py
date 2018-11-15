import cv2
import torch
import torch.nn as nn
import numpy as np


class VGG(nn.Module):
    def __init__(self, convolutional_layers_config, num_classes, batch_norm=False, init_weights=False):
        super(VGG, self).__init__()
        self.features = VGG.make_convolutional_layers(convolutional_layers_config, batch_norm)
        feature_channels = 0
        max_pools = 0
        for v in reversed(convolutional_layers_config):
            if v == 'M':
                max_pools += 1
            elif feature_channels <= 0:
                feature_channels = v
        self.classifier = nn.Sequential(
            nn.Linear(feature_channels * (224 / 2 ** max_pools) ** 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    @staticmethod
    def make_convolutional_layers(config, batch_norm=False):
        layers = []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def load_vgg_face_16_feature_extractor(weights_path):
    """
    Load from VGG 16-layer model (configuration "D")
    """

    model = VGG(num_classes=8, convolutional_layers_config=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                                                            512, 512, 512, 'M', 512, 512, 512, 'M'],
                batch_norm=False, init_weights=False)

    pretrained_dict = torch.load(weights_path)
    model.load_state_dict(pretrained_dict)

    classifier = list(model.classifier.children())
    del classifier[-2:]     # Drop the last 2 layers (fc8 and dropout)
    model.classifier = torch.nn.Sequential(*classifier)

    return model


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


def equalise_histogram(image, use_bgr_colour_model=True):
    if image.ndim == 3 and image.shape[2] == 3:
        if use_bgr_colour_model:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
            hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
            return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR_FULL)
        else:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
            hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
            return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB_FULL)
    else:
        return cv2.equalizeHist(image)
