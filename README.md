# dlib_and_chehra_stuff
A repository of code that performs facial landmark tracking, eye point localisation, and head pose estimation. Internally it uses dlib (so, the ERT method) for landmark localisation. However, it performs tracking instead of per-frame face-detection + localisation thus it should run much faster than the vanilla dlib implementation. Moreover, we trained some additional 49 and 68-landmark models on a larger dataset. Specifically, the following tools are provided:
1. [iBUGFaceTracker](./Compiled/iBUGFaceTracker): A tool that tracks facial landmark in a list of video files.
2. [HeadPoseTest](./Compiled/HeadPoseTest): A tool performing head pose estimation in a list of folders with landmarks already provided.
3. [ibug_face_tracker](./ibug_face_tracker): The python module for facial landmark tracking.

## Dependencies
All executables can be used out-of-the-box. For the Python module, we include pre-built library files for both Ubuntu (built for Python 2.7) and Windows (built for Python 3.5). The executable files are self contained, thus require no additional installation steps. Nonetheless, if you want to use the Python module, you will need to install OpenCV (`pip install opencv-python`) and Dlib (`pip install dlib` or `conda install -c menpo dlib`) in your Python environment.

If you choose to build from source, you will need [opencv-2.4.13.4](https://github.com/opencv/opencv/archive/2.4.13.4.zip), [dlib-19.6](http://dlib.net/files/dlib-19.6.zip), and [Pybind11](https://github.com/pybind/pybind11). Additionally, if you're building on Windows, [boost 1.67](https://www.boost.org/users/history/version_1_67_0.html) is also needed. You should build OpenCV and Dlib from source using [CMake](https://cmake.org/) + gcc / Visual Studio. For OpenCV, you should build with `-DBUILD_SHARED_LIBS=ON -DBUILD_WITH_STATIC_CRT=OFF -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_OPENEXR=ON -DBUILD_PNG=ON`. For Dlib, you should build with `-DDLIB_USE_MKL_FFT=OFF`. When building Dlib on Windows, you may need to manually edit the Visual Studio project file to make sure you are using `/MD` and `/MDd` instead of `/MT` and `/MTd`. Dlib will build both statically-linked (\*.a or \*.lib) and dynamically-linked (\*.so or \*.dll) libraries, make sure you use the statically-linked one. Pybind11 is a header-only library, so you can use it as is. When building boost, please use `--toolset` option to select the correct MSVC version, use `address-model` to select the target platform (32 or 64 bit), and use `--stagedir` to configure the output directory.

## How to Build
- As a reminder, you don't need to build if:
    * You just want to use the executables;
    * You want to use the Python module in Python 2.7 on Ubuntu;
    * Or you want to use the Python module in Python 3.5 on Windows.
- Build on Ubuntu: Navigate to the repository, edit [makefile](./makefile) to put the correct lib and include paths, and then `make all`. 
- Build on Windows: Why would you want to do that? Joke as side, get yourself a copy of Visual Studio 2015, and open the [solution file](./Windows/Build/Build_All.sln). Then, please put the correct lib and include paths into the project settings, and build everything.

## How to Use the Executables
The following executables are provided:
1. [DlibTrackerTest](./Compiled/DlibTrackerTest) It uses your webcam to capture live video and plot tracking result on the screen.
2. [iBUGFaceTracker](./Compiled/iBUGFaceTracker) It tracks facial landmarks in a list of video files. Example: `./Compiled/iBUGFaceTracker ./Test_Data/Jobs.txt 4 1 40 2.0 2`
3. [HeadPoseTest](./Compiled/HeadPoseTest) It performs head pose estimation on a list of folders with landmarks already provided. Example: `./Compiled/HeadPoseTest ./Test_Data/Jobs2.txt 2`

## How to Use the Python Module
The Python module is fully contained in the [ibug_face_tracker](./ibug_face_tracker) folder, so please make sure your Python interpreter can see it. The module contains the following 4 classes:
1. `ibug_face_tracker.FaceTracker`: A tracker to track a single face in the scene. If multiple faces are presented, it will track the largest one until it is lost. Internally, it uses `dlib.frontal_face_detector` for face detection and `ibug_face_tracker.Auxiliary_Utility` for head pose estimation, eye point tracking, and fitting score calculation. This class also exposes some handy static methods for plotting the tracking result.
2. `ibug_face_tracker.Auxiliary_Utility`: A bag of auxiliary tools that performs head pose estimation, eye point tracking, and fitting score calculation.
3. `ibug_face_tracker.FacialLandmarkLocaliser`: The only purpose of this class is to enable model sharing between multiple instances of `ibug_face_tracker.FaceTracker` or `ibug_face_tracker.MultiFaceTracker`. As a result, this class does not expose any method at all.
4. `ibug_face_tracker.MultiFaceTracker`: Similar to `ibug_face_tracker.FaceTracker` but can track multiple faces simultaneously. You can specify how many faces you want to track during instantiation. This is the class you should normally use even when you just want to track a single face. Details about how to use this class is illustrated in the following example:
```
import cv2
import ibug_face_tracker

# Create the tracker
tracker = ibug_face_tracker.MultiFaceTracker("./models/new3_68_pts_UAD_1_tr_6_cas_15.dat",  # The ERT model
                                             "./models/additional_svrs.model",              # The auxiliary model
                                             faces_to_track=3)      # Track no more than 3 faces at a time

# Configure parameters (usually you shouldn't touch the parameters that are not listed here)
tracker.face_detection_interval = 8     # When less than 3 faces are tracked, detect face once in every 8 frames
tracker.face_detection_scale = 0.5      # Downsize the image to 0.5x during face detection (so it runs faster)
tracker.minimum_face_size = 128         # Ignore faces that are smaller than 128x128 pixels
tracker.failure_detection_interval = 3  # Perform failure detection once in every 3 frames
'''
tracker.estimate_head_pose = False      # When you don't want head pose
tracker.eye_iterations = 0              # When you don't want eye points
'''

# Open the webcam
webcam = cv2.VideoCapture(0)

# Process incoming frames
while True:
    ret, frame = webcam.read()
    if ret:
        # Track faces in the frame
        tracker.track(frame)    # If your frame is not coming from OpenCV, set use_bgr_colour_model=False
        
        # Get tracking result
        tracked_faces = tracker.current_result
        '''
        tracked_faces is a list of dictionaries. Each element may contain the following keys:
        id: ID of the face, which will only stay the same when the face is continuously tracked..
        facial_landmarks: Facial landmark coordinates stored in a Nx2 numpy array (N = 49 or 68)
        eye_landmarks: Eye point coordinates stored in a 14x2 numpy array
        pitch, yaw, and roll: Head pose
        '''
        
        # Plot tracking result
        for face in tracked_faces:
            ibug_face_tracker.FaceTracker.plot_landmark_connections(frame, face['facial_landmarks'])
            ibug_face_tracker.FaceTracker.plot_facial_landmarks(frame, face['facial_landmarks'])
            if 'eye_landmarks' in face:
                ibug_face_tracker.FaceTracker.plot_eye_landmarks(frame, face['eye_landmarks'])
        
        # Show the image
        cv2.imshow("ibug_face_tracker.MultiFaceTracker", frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break
    else:
        break

# Close the webcam
webcam.release()
```
For more examples, please refer to the following scripts:
1. [ibug_face_tracker_test.py](./ibug_face_tracker_test.py): Functionally equivalent to [iBUGFaceTracker](./Compiled/iBUGFaceTracker).
2. [ibug_face_tracker_webcam_test.py](./ibug_face_tracker_webcam_test.py): A test script for `ibug_face_tracker.FaceTracker` and `ibug_face_tracker.Auxiliary_Utility`.
3. [ibug_multi_face_tracker_webcam_test.py](./ibug_multi_face_tracker_webcam_test.py): A more elaborate demonstration of the multi-face tracker.
