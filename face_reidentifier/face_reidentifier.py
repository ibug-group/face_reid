from .misc import *
from keras.models import Model
import scipy.spatial.distance as sd
from collections import OrderedDict
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment


class FaceReidentifier(object):
    def __init__(self, model_path='', distance_threshold=1.0, neighbour_count_threshold=4, quality_threshold=1.0,
                 database_capacity=16, descriptor_list_capacity=16, descriptor_update_rate=0.1,
                 mean_rgb=(129.1863, 104.7624, 93.5940), distance_metric='euclidean', model=None):
        if model is None:
            model = load_vgg_face_16_model(model_path)
            model = Model(model.input, model.get_layer('fc7/relu').output)
        self._model = model
        self._distance_threshold = max(0.0, distance_threshold)
        self._neighbour_count_threshold = max(1, int(neighbour_count_threshold))
        self._quality_threshold = quality_threshold
        self._database_capacity = max(1, int(database_capacity))
        self._descriptor_list_capacity = max(1, int(descriptor_list_capacity))
        self._descriptor_update_rate = max(0.0, min(descriptor_update_rate, 1.0))
        assert len(mean_rgb) == 3
        self._mean_rgb = mean_rgb
        self._distance_metric = distance_metric
        self._database = OrderedDict()
        self._unidentified_tracklets = OrderedDict()
        self._exisiting_tracklet_ids = set()
        self._conflicts = []
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
    def quality_threshold(self):
        return self._quality_threshold

    @quality_threshold.setter
    def quality_threshold(self, value):
        self._quality_threshold = value

    @property
    def database_capacity(self):
        return self._database_capacity

    @database_capacity.setter
    def database_capacity(self, value):
        self._database_capacity = max(1, int(value))
        self._clean_database()

    @property
    def descriptor_list_capacity(self):
        return self._descriptor_list_capacity

    @descriptor_list_capacity.setter
    def descriptor_list_capacity(self, value):
        self._descriptor_list_capacity = max(1, int(value))
        self._clean_database()

    @property
    def descriptor_update_rate(self):
        return self._descriptor_update_rate

    @descriptor_update_rate.setter
    def descriptor_update_rate(self, value):
        self._descriptor_update_rate = max(0.0, min(value, 1.0))

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

    def reset(self, reset_face_id_counter=True):
        self._database.clear()
        self._unidentified_tracklets.clear()
        self._exisiting_tracklet_ids.clear()
        self._conflicts.clear()
        if reset_face_id_counter:
            self._face_id_counter = 0

    def _clean_database(self):
        # First, remove the unidentified tracklets that no longer exist
        for tracklet_id in list(self._unidentified_tracklets.keys()):
            if tracklet_id not in self._exisiting_tracklet_ids:
                del self._unidentified_tracklets[tracklet_id]

        # Next, remove the conflicts that are no longer relevant
        for idx in reversed(range(len(self._conflicts))):
            if (self._conflicts[idx][0] not in self._exisiting_tracklet_ids and
                    self._conflicts[idx][1] not in self._exisiting_tracklet_ids):
                del self._conflicts[idx]

        # Then, trim the database
        if len(self._database) > self._database_capacity:
            for face_id in list(self._database.keys())[0: len(self._database) - self._database_capacity]:
                del self._database[face_id]

        # Finally, remove the irrelevant ids from saved identities
        relevant_tracklet_ids = self._exisiting_tracklet_ids.union(np.array(self._conflicts).flatten())
        for face_id in self._database.keys():
            self._database[face_id]['tracklet_ids'].intersection_update(relevant_tracklet_ids)

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
            face_descriptor /= max(np.finfo(np.float).eps, np.linalg.norm(face_descriptor))
        return list(face_descriptors)

    def _update_identity(self, identity, face_descriptors, tracklet_id):
        for face_descriptor in face_descriptors:
            if len(identity['descriptors']) < self._descriptor_list_capacity:
                identity['descriptors'].append(face_descriptor)
            else:
                distances = sd.cdist([face_descriptor], identity['descriptors'],
                                     metric=self._distance_metric)[0]
                closest = np.argmin(distances)
                if distances[closest] <= self._distance_threshold:
                    updated_descriptor = (face_descriptor * self._descriptor_update_rate +
                                          identity['descriptors'][closest] * (1.0 - self._descriptor_update_rate))
                    del identity['descriptors'][closest]
                    identity['descriptors'].append(updated_descriptor)
                else:
                    del identity['descriptors'][0]
                    identity['descriptors'].append(face_descriptor)
        identity['tracklet_ids'].add(tracklet_id)

    def reidentify_faces(self, face_images, tracklet_ids, qualities=None, use_bgr_colour_model=True):
        assert len(face_images) == len(tracklet_ids)
        if len(face_images) > 0:
            face_ids = [0] * len(face_images)
            if qualities is None or len(qualities) != len(face_images):
                qualities = [self._quality_threshold] * len(face_images)

            # These are what we see now
            self._exisiting_tracklet_ids = set(tracklet_ids)

            # Calculate face descriptors
            face_descriptors = self._compute_face_descriptors(face_images, use_bgr_colour_model)

            # Update conflict list
            sorted_unique_tracklet_ids = sorted(list(self._exisiting_tracklet_ids))
            for idx, tracklet_id1 in enumerate(sorted_unique_tracklet_ids):
                for tracklet_id2 in enumerate(sorted_unique_tracklet_ids[idx + 1:]):
                    conflict = [tracklet_id1, tracklet_id2]
                    if conflict not in self._conflicts:
                        self._conflicts.append(conflict)

            # See if some of the faces are already tracked
            for idx in range(len(tracklet_ids)):
                saved_face_ids = list(self._database.keys())
                for saved_face_id in saved_face_ids:
                    if tracklet_ids[idx] in self._database[saved_face_id]['tracklet_ids']:
                        face_ids[idx] = saved_face_id
                        updated_identity = self._database[saved_face_id]
                        if qualities[idx] >= self._quality_threshold:
                            self._update_identity(updated_identity, [face_descriptors[idx]], face_ids[idx])
                        del self._database[saved_face_id]
                        self._database[saved_face_id] = updated_identity
                        break

            # For the remaining faces, add them as unidentified tracklets
            remaining_face_indices = [x for x in range(len(tracklet_ids)) if face_ids[x] == 0]
            for idx in remaining_face_indices:
                tracklet_id = tracklet_ids[idx]
                if qualities[idx] >= self._quality_threshold:
                    if tracklet_id in self._unidentified_tracklets:
                        self._unidentified_tracklets[tracklet_id].append(face_descriptors[idx])
                    else:
                        self._unidentified_tracklets[tracklet_id] = [face_descriptors[idx]]
                elif tracklet_id not in self._unidentified_tracklets:
                    self._unidentified_tracklets[tracklet_id] = []

            # For the next part, we only consider the faces of good quality
            remaining_face_indices = [x for x in remaining_face_indices if qualities[x] >= self._quality_threshold]
            updated_unidentified_tracklet_ids = [tracklet_ids[x] for x in remaining_face_indices]

            # For the updated unidentified tracklets, try to match them to existing identities
            if len(updated_unidentified_tracklet_ids) > 0 and len(self._database) > 0:
                remaining_face_descriptors = [self._unidentified_tracklets[x][-1] for x in
                                              updated_unidentified_tracklet_ids]

                # Compute distances between new and saved descriptors
                saved_face_descriptors = []
                saved_face_ids = []
                for saved_face_id in self._database.keys():
                    saved_face_descriptors += self._database[saved_face_id]['descriptors']
                    saved_face_ids += [saved_face_id] * len(self._database[saved_face_id]['descriptors'])
                distances = sd.cdist(remaining_face_descriptors, saved_face_descriptors, metric=self.distance_metric)
                max_distance = np.amax(distances)

                # Now compute similarities between new descriptors and existing identities
                similiarities = np.ones((len(remaining_face_descriptors), len(self._database)), dtype=np.float)
                # for idx, descriptor in enumerate(face_descriptors):
                #     neighbours = np.where(distances[idx, :] <= self._distance_threshold)[0]
                #     if len(neighbours) >= self._neighbour_count_threshold:
                #         neighbour_face_indices, counts = np.unique(
                #             [archived_face_indices[x] for x in neighbours], return_counts=True)
                #         for idx2 in range(len(neighbour_face_indices)):
                #             similarities[idx, neighbour_face_indices[idx2]] = \
                #                 (counts[idx2] * 2 + 1) * max_distance - np.min(
                #                     [distances[idx, x2] for x2 in [x for x in neighbours if
                #                                                    archived_face_indices[x] ==
                #                                                    neighbour_face_indices[idx2]]])

            # Finally, see if new identities have emerged
            remaining_face_indices = [x for x in range(len(tracklet_ids)) if face_ids[x] == 0]
            if len(remaining_face_indices) > 0:
                for idx in remaining_face_indices:
                    tracklet_id = tracklet_ids[idx]
                    if len(self._unidentified_tracklets[tracklet_id]) >= self._neighbour_count_threshold:
                        self._face_id_counter += 1
                        face_ids[idx] = self._face_id_counter
                        self._database[face_ids[idx]] = {'descriptors': self._unidentified_tracklets[tracklet_id],
                                                         'tracklet_ids': set([tracklet_id])}
                        del self._unidentified_tracklets[tracklet_id]

            self._clean_database()
            return face_ids

            # # Calculate face descriptors
            # face_descriptors = self._compute_face_descriptors(face_images, use_bgr_colour_model)
            #
            # # Only continue if the archive is not empty
            # if len(self._database) > 0 or len(self._unidentified_descriptors) > 0:
            #     # First, try to associate current descriptors to existing identities
            #     if len(self._database) > 0:
            #         archived_face_descriptors = []
            #         archived_face_indices = []
            #         for idx, face in enumerate(self._database):
            #             archived_face_descriptors += face['descriptors']
            #             archived_face_indices += [idx] * len(face['descriptors'])
            #         distances = sd.cdist(face_descriptors, archived_face_descriptors, self._distance_metric)
            #         max_distance = np.amax(distances)
            #
            #         # Calculate similarities between descriptors and existing identities
            #         similarities = -np.ones((len(face_descriptors), len(self._database)), dtype=np.float)
            #         for idx, descriptor in enumerate(face_descriptors):
            #             neighbours = np.where(distances[idx, :] <= self._distance_threshold)[0]
            #             if len(neighbours) >= self._neighbour_count_threshold:
            #                 neighbour_face_indices, counts = np.unique(
            #                     [archived_face_indices[x] for x in neighbours], return_counts=True)
            #                 for idx2 in range(len(neighbour_face_indices)):
            #                     similarities[idx, neighbour_face_indices[idx2]] = \
            #                         (counts[idx2] * 2 + 1) * max_distance - np.min(
            #                             [distances[idx, x2] for x2 in [x for x in neighbours if
            #                                                            archived_face_indices[x] ==
            #                                                            neighbour_face_indices[idx2]]])
            #
            #         # Assign descriptors to existing identities
            #         similarities[similarities < 0.0] *= (len(face_descriptors) * len(self._database) *
            #                                              np.amax(similarities)) ** 2
            #         rows, cols = linear_sum_assignment(-similarities)
            #         for [idx1, idx2] in np.vstack((rows, cols)).T:
            #             if similarities[idx1, idx2] > 0.0:
            #                 if len(self._database[idx2]['descriptors']) < self._descriptor_list_capacity:
            #                     self._database[idx2]['descriptors'].append(face_descriptors[idx1])
            #                 else:
            #                     to_be_updated = np.argmin(sd.cdist([face_descriptors[idx1]],
            #                                                        self._database[idx2]['descriptors'])[0])
            #                     new_descriptor = (face_descriptors[idx1] * self._descriptor_update_rate +
            #                                       self._database[idx2]['descriptors'][to_be_updated] *
            #                                       (1.0 - self._descriptor_update_rate))
            #                     del self._database[idx2]['descriptors'][to_be_updated]
            #                     self._database[idx2]['descriptors'].append(new_descriptor)
            #                 face_ids[idx1] = self._database[idx2]['id']
            #
            #     # Then, try to find new identifies
            #     unassigned_descriptor_indices = [x for x in range(len(face_descriptors)) if face_ids[x] == 0]
            #     if len(unassigned_descriptor_indices) > 0 and len(self._unidentified_descriptors) > 0:
            #         for idx in unassigned_descriptor_indices:
            #             descriptors = self._unidentified_descriptors + [face_descriptors[idx]]
            #             labels = DBSCAN(eps=self._distance_threshold, min_samples=self._neighbour_count_threshold + 1,
            #                             metric=self._distance_metric).fit(descriptors).labels_
            #             if labels[-1] >= 0:
            #                 # New identify found!
            #                 self._face_id_counter += 1
            #                 new_identity = {'id': self._face_id_counter,
            #                                 'descriptors': [self._unidentified_descriptors[x] for x in
            #                                                 range(len(self._unidentified_descriptors)) if
            #                                                 labels[x] == labels[-1]]}
            #                 for idx2 in reversed(range(len(self._unidentified_descriptors))):
            #                     if labels[idx2] == labels[-1]:
            #                         del self._unidentified_descriptors[idx2]
            #                 self._database.append(new_identity)
            #                 face_ids[idx] = new_identity['id']
            #
            #     # Add the left-overs to the unidentified descriptor list
            #     self._unidentified_descriptors += [face_descriptors[x] for x in
            #                                        range(len(face_descriptors)) if face_ids[x] == 0]
            # else:
            #     self._unidentified_descriptors = face_descriptors
        else:
            return []
