from .misc import *
from keras.models import Model
import scipy.spatial.distance as sd


class FaceReidentifier(object):
    def __init__(self, model_path='', distance_threshold=1.218, neighbour_count_threshold=4, database_capacity=30,
                 descriptor_list_capacity=20, descriptor_update_rate=0.05, unidentified_descriptor_list_capacity=30,
                 mean_rgb=(129.1863, 104.7624, 93.5940), distance_metric='euclidean', model=None):
        if model is None:
            model = load_vgg_face_16_model(model_path)
            model = Model(model.input, model.get_layer('fc7/relu').output)
        self._model = model
        self._distance_threshold = max(0.0, distance_threshold)
        self._neighbour_count_threshold = max(1, int(neighbour_count_threshold))
        self._database_capacity = max(1, int(database_capacity))
        self._descriptor_list_capacity = max(1, int(descriptor_list_capacity))
        self._descriptor_update_rate = max(0.0, min(descriptor_update_rate, 1.0))
        self._unidentified_descriptor_list_capacity = max(1, int(unidentified_descriptor_list_capacity))
        assert len(mean_rgb) == 3
        self._mean_rgb = mean_rgb
        self._distance_metric = distance_metric
        self._database = []
        self._unidentified_descriptors = []
        self._face_id_counter = 0

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def distance_threshold(self):
        return self._distance_threshold

    @distance_threshold.setter
    def distance_threshold(self, value):
        self._distance_threshold = max(0.0, value)

    @property
    def neighbour_count_threshold(self):
        return self._neighbour_count_threshold

    @neighbour_count_threshold.setter
    def neighbour_count_threshold(self, value):
        self._neighbour_count_threshold = max(1, int(value))

    @property
    def database_capacity(self):
        return self._database_capacity

    @database_capacity.setter
    def database_capacity(self, value):
        self._database_capacity = max(1, int(value))
        self._limit_database_size()

    @property
    def descriptor_list_capacity(self):
        return self._descriptor_list_capacity

    @descriptor_list_capacity.setter
    def descriptor_list_capacity(self, value):
        self._descriptor_list_capacity = max(1, int(value))
        self._limit_database_size()

    @property
    def descriptor_update_rate(self):
        return self._descriptor_update_rate

    @descriptor_update_rate.setter
    def descriptor_update_rate(self, value):
        self._descriptor_update_rate = max(0.0, min(value, 1.0))

    @property
    def unidentified_descriptor_list_capacity(self):
        return self._unidentified_descriptor_list_capacity

    @unidentified_descriptor_list_capacity.setter
    def unidentified_descriptor_list_capacity(self, value):
        self._unidentified_descriptor_list_capacity = max(1, int(value))
        self._limit_database_size()

    @property
    def mean_rgb(self):
        return self._mean_rgb

    @mean_rgb.setter
    def mean_rgb(self, value):
        assert len(value) == 3
        self._mean_rgb = value

    @property
    def distance_metric(self):
        return self._distance_metric

    @distance_metric.setter
    def distance_metric(self, value):
        self._distance_metric = value

    def _limit_database_size(self):
        if len(self._unidentified_descriptors) > self._unidentified_descriptor_list_capacity:
            self._unidentified_descriptors = \
                self._unidentified_descriptors[len(self._unidentified_descriptors) -
                                               self._unidentified_descriptor_list_capacity:]
        if len(self._database) > self._database_capacity:
            self._database = self._database[len(self._database) - self._database_capacity:]
        for face in self._database:
            if len(face['descriptors']) > self._descriptor_list_capacity:
                face['descriptors'] = face['descriptors'][len(face['descriptors']) -
                                                          self._descriptor_list_capacity:]

    def _compute_face_descriptors(self, face_images, use_bgr_colour_model=True):
        face_images = np.array(face_images).astype(np.float32)
        for face_image in face_images:
            if use_bgr_colour_model:
                face_image[..., 2] -= self._mean_rgb[0]
                face_image[..., 1] -= self._mean_rgb[1]
                face_image[..., 0] -= self._mean_rgb[2]
                face_image[...] = face_image[..., ::-1]
            else:
                face_image[..., 0] -= self._mean_rgb[0]
                face_image[..., 1] -= self._mean_rgb[1]
                face_image[..., 2] -= self._mean_rgb[2]
        face_descriptors = self._model.predict(face_images)
        for face_descriptor in face_descriptors:
            face_descriptor /= max(np.finfo(np.float32).eps, np.linalg.norm(face_descriptor))
        return face_descriptors

    def reidentify_faces(self, face_images, use_bgr_colour_model=True):
        if len(face_images) > 0:
            face_ids = [0] * len(face_images)

            # Calculate face
            face_descriptors = self._compute_face_descriptors(face_images, use_bgr_colour_model)

            # Prepare the data structure for distance calculation
            archived_face_descriptors = self._unidentified_descriptors
            archived_face_ids = [0] * len(self._unidentified_descriptors)
            for face in self._database:
                archived_face_descriptors += face['descriptors']
                archived_face_ids += [face['id']] * len(face['descriptors'])

            # Only continue if the archive is not empty
            if len(archived_face_descriptors) > 0:
                distances = sd.cdist(face_descriptors, archived_face_descriptors, self._distance_metric)
                for idx, descriptor in enumerate(face_descriptors):
                    neighbours = np.where(distances[idx, :] <= self._distance_threshold)
                    if len(neighbours) < self._neighbour_count_threshold:
                        # Not enough neighbours
                        pass
                    else:
                        pass
            else:
                self._unidentified_descriptors = face_descriptors

            self._limit_database_size()
            return face_ids
        else:
            return []

    def delete_unidentified_descriptors(self):
        self._unidentified_descriptors.clear()

    def reset(self, delete_unidentified_descriptors=True, reset_face_id_counter=True):
        self._database.clear()
        if delete_unidentified_descriptors:
            self.delete_unidentified_descriptors()
        if reset_face_id_counter:
            self._face_id_counter = 0



# class FaceReidentifier(object):
#
#     def __init__(self, face_model_path, dist_thred=1.218,  min_nn=4,
#                  db_size1=30, db_size2=20, dist_method='euclidean'):
#
#         self._db = []                                    # descriptor database
#         self._labels = []                                 # labels for descriptor database
#
#         self.meanRGB = [129.1863, 104.7624, 93.5940]    # mean RGB values of the network
#         self.face_model_path = face_model_path
#         self.use_histeq = use_histeq
#         self.dist_thred = dist_thred                    # distance threshold
#         self.min_nn = min_nn                            # minimum number of neighbours
#         self.dist_method = dist_method                  # which distance to use
#         self.db_size1 = db_size1                        # maximum number of descriptors to keep for each ID
#         self.db_size2 = db_size2                        # maximum number of IDs in the database
#
#         model = load_vgg_face_16_model(self.face_model_path, classes=2622)
#         model_out = model.get_layer('fc7/relu').output
#         self.model = Model(model.input, model_out)
#
#     # predict IDs using online DBScan
#     def predict_IDs(self,fea_list):
#
#         if len(self.db)==0:  # if database is empty, add feature to database and set labels to 0
#             face_num=len(fea_list)
#
#             self.db=np.asarray(fea_list,dtype=np.float32)
#             self.label=np.asarray([0]*face_num,dtype=int)
#             self.face_IDs=np.asarray([0]*face_num,dtype=int)
#
#         else:
#             db = np.copy(self.db)
#             label2 = np.copy(self.label)
#             label_idx=np.arange(self.label.size,dtype=int)
#             this_IDs=[]
#
#             for idx,fea in enumerate(fea_list):
#
#                 # calculate pairwise distance between this feature and the database
#                 pair_dist=sd.cdist(np.expand_dims(fea,0), db, self.dist_method)
#                 neig_idx=np.squeeze(pair_dist<=self.dist_thred)
#                 neig_num=np.sum(neig_idx)  # number of neighbours
#
#                 if neig_num < self.min_nn:  # not enough neigbhors, set as unpredicted
#                     this_IDs.append(0)
#
#                 else:
#                     # check if the majority ID of neighbours is 0; if 0, update these IDs
#                     if np.bincount(label2[neig_idx]).argmax() == 0:
#                         label2[neig_idx]=np.max(self.label)+1
#
#                         ori_label_idx=label_idx[neig_idx]  # original index for these neighbours
#                         self.label[ori_label_idx]=np.max(self.label)+1
#
#                     this_IDs.append(np.bincount(label2[neig_idx]).argmax())
#
#                 # ignore neighbours of this ID
#                 db=db[~neig_idx,:]
#                 label2=label2[~neig_idx]
#                 label_idx=label_idx[~neig_idx]
#
#             this_IDs=np.asarray(this_IDs)
#             if np.asarray(fea_list).shape[-1]==self.db.shape[-1]:  # safety check
#                 self.db=np.append(self.db, np.asarray(fea_list), axis=0)
#                 self.label=np.append(self.label,this_IDs)
#             self.face_IDs=this_IDs
#
#
#     def limit_db_size(self):
#         unique_label=np.unique(self.label)
#
#         for lb in unique_label:
#             idx=np.where(self.label==lb)[0]
#             if idx.size>self.db_size1:
#                 # print('Limiting descriptors of one ID ...')
#                 num_to_reduce=idx.size-self.db_size1
#                 np.delete(self.db, idx[0:num_to_reduce], axis=0)
#                 np.delete(self.label,idx[0:num_to_reduce])
#
#         if unique_label.size > self.db_size2:
#             num_to_reduce=unique_label.size-self.db_size2
#             for t in range(num_to_reduce):
#                 # print('Deleting all descriptors of certain IDs ...')
#                 lb=unique_label[t]
#                 idx = np.squeeze(self.label == lb)
#                 self.db=self.db[~idx,:]
#                 self.label=self.label[~idx]
#
#
#     # re-identify multiple faces
#     def reidentify(self, frame, lm_list):
#
#         fea_list=[]
#         self.bbox=[]
#
#         for lm in lm_list:
#             crop_face,rect=crop_face_img(frame,lm)  # crop face out of frame
#             self.bbox.append(rect)
#             x = preprocess_image(crop_face,self.meanRGB,
#                                use_histeq=self.use_histeq,convert_to_RGB=True)
#
#             this_fea = self.model.predict(x)[0]  # get descriptor
#             this_fea = this_fea/np.linalg.norm(this_fea)  # L2 nomarlize
#             fea_list.append(this_fea)
#
#         self.predict_IDs(fea_list)
#         self.reg_ctr+=1
#
#         if np.mod(self.reg_ctr,self.size_check_time)==0:
#             self.limit_db_size()
#
#     # clean database
#     def reset_db(self):
#         self.db=[]  # descriptor database
#         self.label=[] # labels for descriptor database

#
#
# # crop face with landmarks
# def crop_face_img(frame,lm):
#
#     # [minX,minY,maxX,maxY]
#     det = [np.min(lm[:, 0]), np.min(lm[:, 1]), np.max(lm[:, 0]), np.max(lm[:, 1])]
#
#     extend = 0.45
#     tar_dim = 224  # target size for deep face net
#
#     cropWidth = det[2] - det[0]
#     cropHeight = det[3] - det[1]
#     cropLength = int((cropWidth + cropHeight) / 2.0)
#
#     cenX = int(det[0] + cropWidth / 2)
#     cenY = int(det[1] + cropHeight / 2)
#     halfLen = int((1 + extend) * cropLength / 2)
#
#     # the boundary points
#     x1 = cenX - halfLen
#     y1 = cenY - halfLen
#     x2 = cenX + halfLen
#     y2 = cenY + halfLen
#
#     # prevent out of frame
#     x1 = max(1, x1)
#     y1 = max(1, y1)
#     x2 = min(x2, frame.shape[1])
#     y2 = min(y2, frame.shape[0])
#
#     rect=[x1,y1,x2,y2]
#
#     ori_face = frame[y1:y2, x1:x2, :]
#     # ori_face = ori_face[...,::-1]
#     crop_face = cv2.resize(ori_face, (tar_dim, tar_dim))
#
#     return crop_face,rect
#
#
#
#
#
#
#
# # img in bgr format
# def preprocess_image(img, meanRGB, use_histeq=True, convert_to_RGB=True):
#     r, g, b = meanRGB
#
#     if use_histeq:
#         img=hist_eq(img)
#
#     img = img.astype(np.float32)
#
#     # subtract mean RGB
#     img[..., 0] -= b
#     img[..., 1] -= g
#     img[..., 2] -= r
#
#     if convert_to_RGB: # convert BGR to RGB format
#         img= img[..., ::-1]
#
#     img = np.expand_dims(img,0)
#
#     return img
#
#
#
#
#
#
#     def plot_reid(self,frame):
#         for idx,rect in enumerate(self.bbox):
#             ID=self.face_IDs[idx]
#             col = self.cols[np.mod(ID, len(self.cols))]
#             cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),col,2)
#             cv2.putText(frame,str(ID),(int(0.5*(rect[0]+rect[2])-14),int(rect[1]-10)),1,4,col,2)
#         return frame
