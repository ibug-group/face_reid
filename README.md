# Face Re-ID package
Re-identify multiple faces tracked by ibug_face_tracker

## Dependencies
All dependencies of repository 'dlib_and_chehra_stuff' (https://github.com/IntelligentBehaviourUnderstandingGroup/dlib_and_chehra_stuff) plus Tensorflow, CUDA and Keras.

## Models
To run the script, you need to all models of repository 'dlib_and_chehra_stuff' (https://github.com/IntelligentBehaviourUnderstandingGroup/dlib_and_chehra_stuff) plus VggFace model in Keras (https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5).
Models should be placed under the './models' folder.

## Demo
Assume you are using default webcam (0), run:
python ibug_multi_face_reid_webcam_test.py 0 ibug_multi_face_reid_webcam_test.ini
