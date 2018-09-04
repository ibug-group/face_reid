import numpy as np
import cv2
from keras.layers import Flatten, Dense, Input, Activation, Conv2D, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras.utils import layer_utils
import scipy.spatial.distance as sd


def VggFace_VGG16(weights_path,classes=2622):

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


# histogram equalization for a color image
def hist_eq(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_hist = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_hist


# crop face with landmarks
def crop_face_img(frame,lm):

    # [minX,minY,maxX,maxY]
    det = [np.min(lm[:, 0]), np.min(lm[:, 1]), np.max(lm[:, 0]), np.max(lm[:, 1])]

    extend = 0.45
    tar_dim = 224  # target size for deep face net

    cropWidth = det[2] - det[0]
    cropHeight = det[3] - det[1]
    cropLength = int((cropWidth + cropHeight) / 2.0)

    cenX = int(det[0] + cropWidth / 2)
    cenY = int(det[1] + cropHeight / 2)
    halfLen = int((1 + extend) * cropLength / 2)

    # the boundary points
    x1 = cenX - halfLen
    y1 = cenY - halfLen
    x2 = cenX + halfLen
    y2 = cenY + halfLen

    # prevent out of frame
    x1 = max(1, x1)
    y1 = max(1, y1)
    x2 = min(x2, frame.shape[1])
    y2 = min(y2, frame.shape[0])

    rect=[x1,y1,x2,y2]

    ori_face = frame[y1:y2, x1:x2, :]
    # ori_face = ori_face[...,::-1]
    crop_face = cv2.resize(ori_face, (tar_dim, tar_dim))

    return crop_face,rect


# img in bgr format
def preprocess_image(img, meanRGB, use_histeq=True, convert_to_RGB=True):
    r, g, b = meanRGB

    if use_histeq:
        img=hist_eq(img)

    img = img.astype(np.float32)

    # subtract mean RGB
    img[..., 0] -= b
    img[..., 1] -= g
    img[..., 2] -= r

    if convert_to_RGB: # convert BGR to RGB format
        img= img[..., ::-1]

    img = np.expand_dims(img,0)

    return img



class FaceReID:

    def __init__(self, face_model_path, dist_thred=1.218, use_histeq=True, min_nn=4,
                 db_size1=30, db_size2=20, size_check_time=200, dist_method='euclidean'):

        self.db=[]  # descriptor database
        self.label=[] # labels for descriptor database
        self.face_IDs=[] # face ID predictions
        self.reg_ctr = 0  # counter for size regulation
        self.bbox=[]  # bounding box for plots

        self.meanRGB=[129.1863,104.7624,93.5940] # mean RGB values of the network
        self.face_model_path = face_model_path
        self.use_histeq=use_histeq
        self.dist_thred=dist_thred  # distance threshold
        self.min_nn=min_nn  # minimum number of neighbours
        self.dist_method=dist_method # which distance to use
        self.db_size1=db_size1  # maximum number of descriptors to keep for each ID
        self.db_size2=db_size2  # maximum number of IDs in the database
        self.size_check_time=size_check_time # frequency to check database size
        self.cols=[(192,192,192),(255,0,0),(0,255,0),(0,0,255),(255,255,0),
                   (0,255,255),(255,0,255),(128,0,0),(128,128,0)]
        # self.re_id_gap=re_id_gap  # the time gap to apply re-identifier

        model = VggFace_VGG16(self.face_model_path, classes=2622)
        model_out = model.get_layer('fc7/relu').output
        self.model = Model(model.input, model_out)

    # predict IDs using online DBScan
    def predict_IDs(self,fea_list):

        if len(self.db)==0:  # if database is empty, add feature to database and set labels to 0
            face_num=len(fea_list)

            self.db=np.asarray(fea_list,dtype=np.float32)
            self.label=np.asarray([0]*face_num,dtype=int)
            self.face_IDs=np.asarray([0]*face_num,dtype=int)

        else:
            db = np.copy(self.db)
            label2 = np.copy(self.label)
            label_idx=np.arange(self.label.size,dtype=int)
            this_IDs=[]

            for idx,fea in enumerate(fea_list):

                # calculate pairwise distance between this feature and the database
                pair_dist=sd.cdist(np.expand_dims(fea,0), db, self.dist_method)
                neig_idx=np.squeeze(pair_dist<=self.dist_thred)
                neig_num=np.sum(neig_idx)  # number of neighbours

                if neig_num < self.min_nn:  # not enough neigbhors, set as unpredicted
                    this_IDs.append(0)

                else:
                    # check if the majority ID of neighbours is 0; if 0, update these IDs
                    if np.bincount(label2[neig_idx]).argmax() == 0:
                        label2[neig_idx]=np.max(self.label)+1

                        ori_label_idx=label_idx[neig_idx]  # original index for these neighbours
                        self.label[ori_label_idx]=np.max(self.label)+1

                    this_IDs.append(np.bincount(label2[neig_idx]).argmax())

                # ignore neighbours of this ID
                db=db[~neig_idx,:]
                label2=label2[~neig_idx]
                label_idx=label_idx[~neig_idx]

            this_IDs=np.asarray(this_IDs)
            if np.asarray(fea_list).shape[-1]==self.db.shape[-1]:  # safety check
                self.db=np.append(self.db, np.asarray(fea_list), axis=0)
                self.label=np.append(self.label,this_IDs)
            self.face_IDs=this_IDs


    def limit_db_size(self):
        unique_label=np.unique(self.label)

        for lb in unique_label:
            idx=np.where(self.label==lb)[0]
            if idx.size>self.db_size1:
                # print('Limiting descriptors of one ID ...')
                num_to_reduce=idx.size-self.db_size1
                np.delete(self.db, idx[0:num_to_reduce], axis=0)
                np.delete(self.label,idx[0:num_to_reduce])

        if unique_label.size > self.db_size2:
            num_to_reduce=unique_label.size-self.db_size2
            for t in range(num_to_reduce):
                # print('Deleting all descriptors of certain IDs ...')
                lb=unique_label[t]
                idx = np.squeeze(self.label == lb)
                self.db=self.db[~idx,:]
                self.label=self.label[~idx]


    def plot_reid(self,frame):
        for idx,rect in enumerate(self.bbox):
            ID=self.face_IDs[idx]
            col = self.cols[np.mod(ID, len(self.cols))]
            cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),col,2)
            cv2.putText(frame,str(ID),(int(0.5*(rect[0]+rect[2])-14),int(rect[1]-10)),1,4,col,2)
        return frame


    # re-identify multiple faces
    def reidentify(self, frame, lm_list):

        fea_list=[]
        self.bbox=[]

        for lm in lm_list:
            crop_face,rect=crop_face_img(frame,lm)  # crop face out of frame
            self.bbox.append(rect)
            x = preprocess_image(crop_face,self.meanRGB,
                               use_histeq=self.use_histeq,convert_to_RGB=True)

            this_fea = self.model.predict(x)[0]  # get descriptor
            this_fea = this_fea/np.linalg.norm(this_fea)  # L2 nomarlize
            fea_list.append(this_fea)

        self.predict_IDs(fea_list)
        self.reg_ctr+=1

        if np.mod(self.reg_ctr,self.size_check_time)==0:
            self.limit_db_size()

    # clean database
    def reset_db(self):
        self.db=[]  # descriptor database
        self.label=[] # labels for descriptor database



