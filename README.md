# The Face Re-ID Package and Demonstrator
A multi-face tracker that assigns each person an unique ID that is consistent through time so that we can keep track of the person's identity when she / he is moving in and out of the camera's field of view. The algorithm is loosely based on our previous work [\[1\]](https://ibug.doc.ic.ac.uk/media/uploads/documents/a_real-time_and_unsupervised_face_re-identification_system_for_human-robot_interaction.pdf) but with a different, more pragmatic clustering approach that takes tracklet information, head pose, and fitting score into consideration so as to improve reidentification accuracy on live videos, especially when multiple people are in the scene. The following are included in this repository:
1. [ibug.face_reid](./ibug/face_reid): The Python package for face reidentification.
2. [face_reidentifier_test.py](./face_reidentifier_test.py): A demonstration of the face reidentification package on live video captured by webcam.

## Dependencies
* [PyTorch](https://pytorch.org/) (`conda install pytorch torchvision -c pytorch`) with [CUDA](https://developer.nvidia.com/cuda-90-download-archive) and [cuDNN](https://developer.nvidia.com/cudnn) (though ***you don't need to manually install the latter two***). The code has been developed for ***Python 3.5*** on a ***Ubuntu*** machine with ***PyTorch 0.4.1*** and ***CUDA 9.0***. However, it will also works in Python 2.7, on a Windows (7/8/10) machine, and / or with other (newer) versions of the aforementioned libraries as long as they are compatible with each other.
* [OpenCV](https://opencv.org/) (`pip install opencv-python`), [numpy](http://www.numpy.org/) (`pip install numpy`), and [scipy](https://www.scipy.org/) (`pip install scipy`).
* [A multi-face landmark tracker](https://github.com/IntelligentBehaviourUnderstandingGroup/face_tracking). I used this [dlib](http://dlib.net/)-and-[chehra](https://ibug.doc.ic.ac.uk/resources/chehra-tracker-cvpr-2014/)-based tracker because it is very light weight (comparing to those based on deep-methods) while is still accurate enough for our application scenario, since our method does not require the coordinates of individual landmarks, only the faces' bounding box. Nonetheless, feel free to use a better tracker, such as [FAN](https://github.com/1adrianb/2D-and-3D-face-alignment) if you also need the landmarks for other purposes.
* Last but not least, don't forget to ***[download the model file](https://drive.google.com/open?id=1sLtsfu_Ry_l_3iN6goRtI3Jd_E-bhoVB)*** and save it in the [models folder](./models).

## How to Run the Demo
Just run `python face_reidentifier_test.py 0` from your terminal, in which 0 means you are to use the first (#0) webcam connected to your machine. Other parameters are configured by [face_reidentifier_test.ini](./face_reidentifier_test.ini). Although this file contains many parameters, most should be kept to their default value. Nonetheless, the following entries can / should be tuned to suit your need:
* `cv2.VideoCapture/*`: Dimension of the images captured from the webcam.
* `ibug.face_tracking.MultiFaceTracker/*`: Parameters controlling the [multi-face landmark tracker](https://github.com/IntelligentBehaviourUnderstandingGroup/face_tracking). Specifically, the following may need to be changed:
    - `ibug.face_tracking.MultiFaceTracker/ert_model_path` and `ibug.face_tracking.MultiFaceTracker/auxiliary_model_path`: Path of the model files.
    - `ibug.face_tracking.MultiFaceTracker/faces_to_track`: The maximum number of faces to track at any given time.
    - `ibug.face_tracking.MultiFaceTracker/minimum_face_size`: To avoid false positives, faces smaller than this will be ignored.
* `ibug.face_reid.FaceReidentifierEx/*`: Parameters controlling the face re-ID module, in which the following may be tuned:
    - `ibug.face_reid.FaceReidentifierEx/database_capacity`: The maximum number of identities to be remembered by the re-ID module. If new identity appears after the database is full, the module will forget the identity which has not appear for the longest of time (not necessarily the oldest one) to make space for the new identity.
    - `ibug.face_reid.FaceReidentifierEx/gpu`: Index (>= 0) of the graphics card to use. If this parameter is left empty of if the designated card is unavailable, the code will fall back to run on CPU.
    - `ibug.face_reid.FaceReidentifierEx/reidentification_interval`: Face reidentification will be performed only once per this number of frames. This is not only to reduce computational workload but also to avoid storing descriptors that are almost identical to each other. For the frames on which face reidentification is not performed, the face ID is propagated through tracklet continuity.
    - `ibug.face_reid.FaceReidentifierEx/minimum_tracklet_length`: Face reidentification will only be performed on faces that have been continuously tracked for at least this number of frames. This is inline with the ghost elimination approach described in [\[1\]](https://ibug.doc.ic.ac.uk/media/uploads/documents/a_real-time_and_unsupervised_face_re-identification_system_for_human-robot_interaction.pdf).

## How to Install the Python Package
* To install: `python setup.py install`
* To uninstall: `pip uninstall ibug_face_reid`

## How to Use the Python Package
You can either use `ibug.face_reid.FaceReidentifier` or `ibug.face_reid.FaceReidentifierEx` for face reidentification. To use the former, you will need to perform tracklet management and face image extraction by yourself, while the latter works seamlessly with the face tracking result produced by `ibug.face_tracking.MultiFaceTracker` (see [ibug.face_tracking](https://github.com/IntelligentBehaviourUnderstandingGroup/face_tracking)). An example is shown below:

```python
import cv2
import numpy as np
from ibug.face_tracking import *
from ibug.face_reid import FaceReidentifierEx

# Create the facial landmark tracker (to track at most 6 faces)
ert_model_path = '../face_tracking/models/new3_49_pts_UAD_1_tr_6_cas_15.dat'
aux_model_path = '../face_tracking/models/additional_svrs.model'
tracker = MultiFaceTracker(ert_model_path, aux_model_path, faces_to_track=6)
# You may wish to tune the tracker's parameters here.

# Create the face reidentifier, using GPU #0
reidentifier = FaceReidentifierEx(model_path='./models/vggface16_pytorch_weights.pt',
                                  gpu=0)
# You may wish to tune the reidentifier's parameters here.

# Open the webcam
webcam = cv2.VideoCapture(0)

# Process incoming frames
while True:
    ret, frame = webcam.read()
    if ret:
        # Track faces in the frame
        tracker.track(frame)    # If your frame is not coming from OpenCV, set use_bgr_colour_model=False
        
        # Get face tracking result
        tracked_faces = tracker.current_result
        
        # Perform face reidentification
        identities = reidentifier.reidentify_tracked_faces(frame, tracked_faces)
        '''
        ibug.face_reid.FaceReidentifierEx.reidentify_tracked_faces
        
        Input:
        * frame: could be None if you choose to provide the extracted face images.
        * tracked_faces: a list of dictionaries, each containing the following:
            * id: tracklet ID of the face, must be unique, preferably non-negative.
            * facial_landmarks: facial landmarks for face image extraction.
            * [optional] roll: face roll (in degrees) to help face image extraction.
            * [optional] face_image: extracted face image. All other fields would be
              ignored if the face image is provided.
        * use_bgr_colour_model: Set this to True (the default value) if the frame is
          coming from OpenCV or False if otherwise.

        Output:
        A dictionary with tracklet IDs as keys. Each value consists of a dictionary  
        containing the following fields:
        * face_id: Identity of the face, starting from 1. A number of 0 means the 
          face is yet to be identified.
        * face_image: Extracted face image or None.
        '''
        
        # Plot result
        for face in tracked_faces:
            face_id = identities[face['id']]['face_id']
            if face_id > 0:
                FaceTracker.plot_landmark_connections(frame, 
                                                      face['facial_landmarks'])
                FaceTracker.plot_facial_landmarks(frame, 
                                                  face['facial_landmarks'])
                if 'eye_landmarks' in face:
                    FaceTracker.plot_eye_landmarks(frame, face['eye_landmarks'])
                text_origin = tuple(np.floor(np.min(face['facial_landmarks'], 
                                                    axis=0) + [2, -12]).astype(int))
                cv2.putText(frame, "Face #%d" % face_id, text_origin,
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0),
                            lineType=cv2.LINE_AA)
            else:
                FaceTracker.plot_landmark_connections(frame, 
                                                      face['facial_landmarks'],
                                                      colour=(192, 192, 192))
        
        # Show the image
        cv2.imshow("ibug.face_reid.FaceReidentifierEx", frame)
        key = cv2.waitKey(1) % 2 ** 16
        if key == ord('q') or key == ord('Q'):
            break
    else:
        break
cv2.destroyAllWindows()

# Close the webcam
webcam.release()
```

A more detailed example can be found in [face_reidentifier_test.py](./face_reidentifier_test.py).

If you choose to use a different facial landmark tracker, you may also need to write some code to produce tracklet information and head roll. For tracklet information, a simple association technique based on intersection-over-union (IoU) score [\[2\]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Jin_End-To-End_Face_Detection_ICCV_2017_paper.pdf) works quite well in practice. Head roll is not absolutely necessary, but will help nonetheless to reduce false positives when they are made available to the reidentification module. For this purpose, a rigid registration through similarity transform would be sufficient.

## References
[1] Wang Y, Shen J, Petridis S, Pantic M. [A real-time and unsupervised face Re-Identification system for Human-Robot Interaction](https://ibug.doc.ic.ac.uk/media/uploads/documents/a_real-time_and_unsupervised_face_re-identification_system_for_human-robot_interaction.pdf). Pattern Recognition Letters. 2018 Apr 9.

[2] Jin SY, Su H, Stauffer C, Learned-Miller EG. [End-to-End Face Detection and Cast Grouping in Movies Using Erdös-Rényi Clustering](http://openaccess.thecvf.com/content_ICCV_2017/papers/Jin_End-To-End_Face_Detection_ICCV_2017_paper.pdf). In ICCV 2017 Oct 1 (pp. 5286-5295).
