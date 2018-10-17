import cv2
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
