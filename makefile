INC_DIR=-I ~/libs/opencv-2.4.13.4-static/include -I ~/libs/opencv-2.4.13.4-static/include/opencv \
	-I ~/libs/dlib-19.6-static/include -I ~/libs/dlib-19.6-static/include/dlib/image_processing \
	-I ~/libs/pybind11/include -I ~/miniconda2/pkgs/python-2.7.14-h1571d57_31/include/python2.7
LIB_DIR=-L ~/libs/opencv-2.4.13.4-static/lib -L ~/libs/dlib-19.6-static/lib
CC=g++

all: ./Compiled/iBUGFaceTracker ./Compiled/DlibTrackerTest ./Compiled/HeadPoseTest \
     ./Compiled/ibug_face_tracker.so

./Compiled/ibug_face_tracker.so: ./Compiled/ibug_face_tracker.o \
				 ./Compiled/dlib_facial_landmark_tracker.o \
				 ./Compiled/ChehraUtilities.o
	$(CC) $(LIB_DIR) -std=c++11 -shared -o ./Compiled/ibug_face_tracker.so \
	./Compiled/ibug_face_tracker.o ./Compiled/dlib_facial_landmark_tracker.o \
	./Compiled/ChehraUtilities.o -lopencv_highgui -lopencv_imgproc -lopencv_core \
	-lzlib -ldlib -lrt -lpthread -lm -ldl -lnsl -lblas -llapack
	cp ./Compiled/ibug_face_tracker.so ./ibug_face_tracker/

./Compiled/ibug_face_tracker.o: ./Source/ibug_face_tracker.cpp \
				./Source/dlib_facial_landmark_tracker.h \
				./Source/dlib_shape_predictor.h \
				./Source/ChehraUtilities.h
	$(CC) $(INC_DIR) -std=c++11 -fPIC -o ./Compiled/ibug_face_tracker.o \
	-c ./Source/ibug_face_tracker.cpp

./Compiled/HeadPoseTest: ./Compiled/HeadPoseTest.o \
			 ./Compiled/ChehraUtilities.o
	$(CC) $(LIB_DIR) -std=c++11 -o ./Compiled/HeadPoseTest ./Compiled/HeadPoseTest.o \
	./Compiled/ChehraUtilities.o -lopencv_imgproc -lopencv_core -lzlib -lpthread \
	-lboost_system -lboost_filesystem

./Compiled/HeadPoseTest.o: ./Source/HeadPoseTest.cpp \
			   ./Source/ChehraUtilities.h
	$(CC) $(INC_DIR) -std=c++11 -o ./Compiled/HeadPoseTest.o -c ./Source/HeadPoseTest.cpp

./Compiled/iBUGFaceTracker: ./Compiled/iBUGFaceTracker.o \
			    ./Compiled/dlib_facial_landmark_tracker.o \
			    ./Compiled/ChehraUtilities.o \
			    ./Compiled/videoreader.o
	$(CC) $(LIB_DIR) -std=c++11 -o ./Compiled/iBUGFaceTracker ./Compiled/iBUGFaceTracker.o \
	./Compiled/dlib_facial_landmark_tracker.o ./Compiled/ChehraUtilities.o \
	./Compiled/videoreader.o -lopencv_highgui -lopencv_imgproc -lopencv_core -llibpng -ldlib \
	-llibjpeg -lIlmImf -llibjasper -llibtiff -lzlib  -lrt -lpthread -lm -ldl -lnsl -lblas \
	-llapack -lavformat-ffmpeg -lavcodec-ffmpeg -lavutil-ffmpeg -lswscale-ffmpeg \
	-lboost_system -lboost_filesystem

./Compiled/DlibTrackerTest: ./Compiled/DlibTrackerTest.o \
			    ./Compiled/dlib_facial_landmark_tracker.o \
			    ./Compiled/ChehraUtilities.o
	$(CC) $(LIB_DIR) -std=c++11 -o ./Compiled/DlibTrackerTest ./Compiled/DlibTrackerTest.o \
	./Compiled/dlib_facial_landmark_tracker.o ./Compiled/ChehraUtilities.o  \
	-lopencv_highgui -lopencv_imgproc -lopencv_core -llibpng -ldlib -llibjpeg -lIlmImf \
	-llibjasper -llibtiff -lzlib  -lrt -lpthread -lm -ldl -lnsl -lblas -llapack \
	-lavformat-ffmpeg -lavcodec-ffmpeg -lavutil-ffmpeg -lswscale-ffmpeg -lboost_system \
	-lboost_filesystem

./Compiled/iBUGFaceTracker.o: ./Source/iBUGFaceTracker.cpp \
			      ./Source/dlib_facial_landmark_tracker.h \
			      ./Source/dlib_shape_predictor.h \
			      ./Source/ChehraUtilities.h \
			      ./Source/videoreader/scopeguard.h \
			      ./Source/videoreader/videoreader.h
	$(CC) $(INC_DIR) -std=c++11 -fPIC -o ./Compiled/iBUGFaceTracker.o -c ./Source/iBUGFaceTracker.cpp

./Compiled/DlibTrackerTest.o: ./Source/DlibTrackerTest.cpp \
			      ./Source/dlib_facial_landmark_tracker.h \
			      ./Source/dlib_shape_predictor.h \
			      ./Source/ChehraUtilities.h
	$(CC) $(INC_DIR) -std=c++11 -o ./Compiled/DlibTrackerTest.o -c ./Source/DlibTrackerTest.cpp

./Compiled/dlib_facial_landmark_tracker.o: ./Source/dlib_facial_landmark_tracker.cpp \
					   ./Source/dlib_facial_landmark_tracker.h \
					   ./Source/dlib_shape_predictor.h \
					   ./Source/ChehraUtilities.h
	$(CC) $(INC_DIR) -std=c++11 -fPIC -o ./Compiled/dlib_facial_landmark_tracker.o -c \
	./Source/dlib_facial_landmark_tracker.cpp

./Compiled/ChehraUtilities.o: ./Source/ChehraUtilities.cpp \
			      ./Source/ChehraUtilities.h
	$(CC) $(INC_DIR) -std=c++11 -fPIC -o ./Compiled/ChehraUtilities.o -c ./Source/ChehraUtilities.cpp

./Compiled/videoreader.o: ./Source/videoreader/scopeguard.h \
			  ./Source/videoreader/videoreader.h \
			  ./Source/videoreader/videoreader.cpp
	$(CC) $(INC_DIR) -std=c++11 -o ./Compiled/videoreader.o -c ./Source/videoreader/videoreader.cpp

clean:
	rm -f ./Compiled/*.o

