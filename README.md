# The Face Re-ID Package and Demonstrator
A multi-face tracker that assigns each person an unique ID that is consistent through time so that we can keep track of the person's identity when she / he is moving in and out of the camera's field of view. The algorithm is loosely based on our previous work [\[1\]](https://ibug.doc.ic.ac.uk/media/uploads/documents/a_real-time_and_unsupervised_face_re-identification_system_for_human-robot_interaction.pdf) but with a different, more pragmatic clustering approach that takes tracklet information, head pose, and fitting score into consideration so as to improve reidentification accuracy on live videos, especially when multiple people are in the scene.

[![Face Re-ID Demo](https://img.youtube.com/vi/DZ4XFO-56ww/0.jpg)](https://www.youtube.com/watch?v=DZ4XFO-56ww "Face Re-ID Demo")

## Dependencies
* [PyTorch](https://pytorch.org/) (`conda install pytorch torchvision -c pytorch`) with [CUDA](https://developer.nvidia.com/cuda-90-download-archive) and [cuDNN](https://developer.nvidia.com/cudnn) (though ***you don't need to manually install the latter two***). The code has been developed for ***Python 2.7*** on a ***Ubuntu*** machine with ***PyTorch 0.4.1*** and ***CUDA 9.0***. However, it will also works in Python 3.x, on a Windows (7/8/10) machine, and / or with other (newer) versions of the aforementioned libraries as long as they are compatible with each other.
* [OpenCV](https://opencv.org/) (`pip install opencv-python`), numpy (`pip install numpy`), and scipy (`pip install scipy`).
* [A multi-face landmark tracker](https://github.com/IntelligentBehaviourUnderstandingGroup/face_tracking). I used this [dlib](http://dlib.net/)-and-[chehra](https://ibug.doc.ic.ac.uk/resources/chehra-tracker-cvpr-2014/)-based tracker because it is very light weight (comparing to those based on deep-methods) while is still accurate enough for our application scenario, since our method does not require the coordinates of individual landmarks, only the faces' bounding box. Nonetheless, feel free to use a better tracker, such as [FAN](https://github.com/1adrianb/2D-and-3D-face-alignment) if you also need the landmarks for other purposes.

## How to run the demo
Just run `python ibug_multi_face_reid_webcam_test.py 0` from your terminal, in which 0 means you are to use the first (#0) webcam connected to your machine. Other parameters are configured by [face_reidentifier_test.ini](./face_reidentifier_test.ini). Although this file contains many parameters, most should be kept to their default value. Nonetheless, the following entries can / should be tuned to suit your need:
* `cv2.VideoCapture/*`: Dimension of the images captured from the webcam.
* `ibug_face_tracker.MultiFaceTracker/*`: Parameters controlling the [multi-face landmark tracker](https://github.com/IntelligentBehaviourUnderstandingGroup/dlib_and_chehra_stuff). Specifically, the following may need to be changed:
    * `ibug_face_tracker.MultiFaceTracker/repository_path`: Path of the tracker's repository. It could be left empty if the package is already made reachable to your Python interpreter.
    * `ibug_face_tracker.MultiFaceTracker/ert_model_path` and `ibug_face_tracker.MultiFaceTracker/auxiliary_model_path`: Path of the model files.
    * `ibug_face_tracker.MultiFaceTracker/faces_to_track`: The maximum number of faces to track at any given time.
    * `ibug_face_tracker.MultiFaceTracker/minimum_face_size`: To avoid false positives, faces smaller than this will be ignored.
* `face_reidentifier.FaceReidentifier/*`: Parameters controlling the face re-ID module, in which only the following should be tuned:
    * `face_reidentifier.FaceReidentifier/database_capacity`: The maximum number of identities to be remembered by the re-ID module. If new identity appears after the database is full, the module will forget the identity which has not appear for the longest of time (not necessarily the oldest one) to make space for the new identity.
    * `face_reidentifier.FaceReidentifier/gpu`: Index (>= 0) of the graphics card to use. If this parameter is left empty of if the designated card is unavailable, the code will fall back to run on CPU.
* `tracking_context/*`: Parameters concerning tracklet management:
    * `tracking_context/face_reidentification_interval`: Face reidentification will be performed only once per this number of frames. This is not only to reduce computational workload but also to avoid storing descriptors that are almost identical to each other. For the frames on which face reidentification is not performed, the face ID is propagated through tracklet continuity.
    * `tracking_context/minimum_tracking_length`: Face reidentification will only be performed on faces that have been continuously tracked for at least this number of frames. This is inline with the ghost elimination approach described in [\[1\]](https://ibug.doc.ic.ac.uk/media/uploads/documents/a_real-time_and_unsupervised_face_re-identification_system_for_human-robot_interaction.pdf).

## How to use the module
Please refer to the code in [face_reidentifier_test.py](./face_reidentifier_test.py) about how to use the face reidentification module in your own project. Please take note on how tracking context is managed as it influences the face reidentification result but is yet to be integrated into the module. This coupling issue will be addressed in a future update.

If you choose to use a different facial landmark tracker, you may also need to write some code to produce tracklet information, head pose and fitting score. For tracklet information, a simple association technique based on intersection-over-union (IoU) score [\[2\]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Jin_End-To-End_Face_Detection_ICCV_2017_paper.pdf) works quite well in practice. The latter two (head pose and fitting score) are not absolutely necessary, but will help nonetheless to reduce false positives when they are made available to the reidentification module. For head pose, a rigid registration through similarity transform would be sufficient. For fitting score, any confidence measure, heuristics, or a combination of both, can be used.

## References
[1] Wang Y, Shen J, Petridis S, Pantic M. [A real-time and unsupervised face Re-Identification system for Human-Robot Interaction](https://ibug.doc.ic.ac.uk/media/uploads/documents/a_real-time_and_unsupervised_face_re-identification_system_for_human-robot_interaction.pdf). Pattern Recognition Letters. 2018 Apr 9.

[2] Jin SY, Su H, Stauffer C, Learned-Miller EG. [End-to-End Face Detection and Cast Grouping in Movies Using Erdös-Rényi Clustering](http://openaccess.thecvf.com/content_ICCV_2017/papers/Jin_End-To-End_Face_Detection_ICCV_2017_paper.pdf). In ICCV 2017 Oct 1 (pp. 5286-5295).
