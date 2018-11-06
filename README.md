# The Face Re-ID Package and Demonstrator
A multi-face tracker that assigns each person an unique ID that is consistent through time so that we can keep track of the person's identity when she / he is moving in and out of the camera's field of view. 

[![Face Re-ID Demo](https://img.youtube.com/vi/DZ4XFO-56ww/0.jpg)](https://www.youtube.com/watch?v=DZ4XFO-56ww "Face Re-ID Demo")

## Dependencies
* [CUDA](https://developer.nvidia.com/cuda-90-download-archive), [cuDNN](https://developer.nvidia.com/cudnn), [TersorFlow](https://www.tensorflow.org/install/pip), and [Keras](https://keras.io/#installation). The code has been developed for ***Python 2.7*** on a ***Ubuntu*** machine with ***CUDA 9.0***, ***cuDNN 7.0***, ***TensorFlow 1.9***, and ***Keras 2.2***. However, it will also works in Python 3.x, on a Windows (7/8/10) machine, and / or with other (newer) versions of the aforementioned libraries as long as they are compatible with each other.
* OpenCV (`pip install opencv-python`), numpy (`pip install numpy`), and scipy (`pip install scipy`).
* [A multi-face landmark tracker](https://github.com/IntelligentBehaviourUnderstandingGroup/dlib_and_chehra_stuff). I used this [dlib](http://dlib.net/)-and-[chehra](https://ibug.doc.ic.ac.uk/resources/chehra-tracker-cvpr-2014/)-based tracker because it is very light weight (comparing to those based on deep-methods) while is still accurate enough for our application scenario, since our method does not require the coordinates of individual landmarks, only the faces' bounding box. N/onetheless, feel free to use a better tracker, such as [FAN](https://github.com/1adrianb/2D-and-3D-face-alignment) if you also need the landmarks for other purposes.
* [The VGGFace model for Keras](https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5). Please make sure to save it in the [./models](./models) folder.

## How to run the demo
Run `python ibug_multi_face_reid_webcam_test.py 0` in your terminal, in which 0 denotes the webcam you want to use.

## How to use the module

N In that case, you may also need to write some code to produce tracklet information, head pose and fitting score. For tracklet information, a simple association technique based on intersection-over-union (IoU) score [\[2\]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Jin_End-To-End_Face_Detection_ICCV_2017_paper.pdf) works quite well in practice. The latter two (head pose and fitting score) are not absolutely necessary, but will help nonetheless to reduce false positives when they are made available to the reidentification module. For head pose, a simple rigid registration through similarity transform would be sufficient. For fitting score, any confidence measure, heuristics, or a combination of both, can be used.

## References
[1] Wang Y, Shen J, Petridis S, Pantic M. [A real-time and unsupervised face Re-Identification system for Human-Robot Interaction](https://ibug.doc.ic.ac.uk/media/uploads/documents/a_real-time_and_unsupervised_face_re-identification_system_for_human-robot_interaction.pdf). Pattern Recognition Letters. 2018 Apr 9.

[2] Jin SY, Su H, Stauffer C, Learned-Miller EG. [End-to-End Face Detection and Cast Grouping in Movies Using Erdös-Rényi Clustering](http://openaccess.thecvf.com/content_ICCV_2017/papers/Jin_End-To-End_Face_Detection_ICCV_2017_paper.pdf). In ICCV 2017 Oct 1 (pp. 5286-5295).


All dependencies of repository 'dlib_and_chehra_stuff' (https://github.com/IntelligentBehaviourUnderstandingGroup/dlib_and_chehra_stuff) plus Tensorflow, CUDA and Keras.

## Models
To run the script, you need to all models of repository 'dlib_and_chehra_stuff' (https://github.com/IntelligentBehaviourUnderstandingGroup/dlib_and_chehra_stuff) plus VggFace model in Keras (https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5).

Models should be placed under './models' folder.

## Demo
Assume you are using default webcam (0), run:

`python ibug_multi_face_reid_webcam_test.py 0 ibug_multi_face_reid_webcam_test.ini`
