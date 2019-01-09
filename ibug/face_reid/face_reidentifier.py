import cv2
import warnings
from .misc import *
import scipy.spatial.distance as sd
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from copy import deepcopy


class FaceReidentifier(object):
    def __init__(self, model_path="", distance_threshold=1.0, neighbour_count_threshold=4, quality_threshold=1.0,
                 database_capacity=16, descriptor_list_capacity=16, descriptor_update_rate=0.1,
                 mean_rgb=(129.1863, 104.7624, 93.5940), distance_metric='euclidean', model=None, gpu=None):
        if len(model_path) > 0:
            model = load_vgg_face_16_feature_extractor(model_path)
        self._model = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if gpu is not None and torch.cuda.is_available():
                    self._gpu = int(gpu)
                    capability = torch.cuda.get_device_capability(self._gpu)
                    if capability[0] >= 3 and capability != (3, 0):
                        self._device = torch.device('cuda:%d' % self._gpu)
                        self._model = model.to(self._device)
        except:
            pass
        finally:
            if self._model is None:
                self._gpu = None
                self._device = torch.device('cpu')
                self._model = model.to(self._device)
        self._model.eval()
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
        self._clean_database()

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

    @property
    def gpu(self):
        return self._gpu

    def reset(self, reset_face_id_counter=True):
        self._database.clear()
        self._unidentified_tracklets.clear()
        self._exisiting_tracklet_ids.clear()
        del self._conflicts[:]
        if reset_face_id_counter:
            self._face_id_counter = 0

    def _clean_database(self):
        # First, remove the unidentified tracklets that no longer exist
        for tracklet_id in list(self._unidentified_tracklets.keys()):
            if tracklet_id not in self._exisiting_tracklet_ids:
                del self._unidentified_tracklets[tracklet_id]

        # Then, for each unidentified tracklet, limit the number of saved descriptors
        for tracklet_id in self._unidentified_tracklets.keys():
            if len(self._unidentified_tracklets[tracklet_id]) > self._neighbour_count_threshold:
                del self._unidentified_tracklets[tracklet_id][
                    0: len(self._unidentified_tracklets[tracklet_id]) - self._neighbour_count_threshold]

        # Next, remove the conflicts that are no longer relevant
        for idx in reversed(range(len(self._conflicts))):
            if (self._conflicts[idx][0] not in self._exisiting_tracklet_ids and
                    self._conflicts[idx][1] not in self._exisiting_tracklet_ids):
                del self._conflicts[idx]

        # Then, trim the database
        if len(self._database) > self._database_capacity:
            for face_id in list(self._database.keys())[0: len(self._database) - self._database_capacity]:
                del self._database[face_id]

        # Finally, for each saved identity, remove the irrelevant tracklet ids and trim the descriptor list
        relevant_tracklet_ids = self._exisiting_tracklet_ids.union(np.array(self._conflicts).flatten())
        for face_id in self._database.keys():
            self._database[face_id]['tracklet_ids'].intersection_update(relevant_tracklet_ids)
            if len(self._database[face_id]['descriptors']) > self._descriptor_list_capacity:
                del self._database[face_id]['descriptors'][
                    0: len(self._database[face_id]['descriptors']) - self._descriptor_list_capacity]

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
        face_images = torch.from_numpy(face_images.transpose([0, 3, 1, 2]))
        if self._device.type == 'cpu':
            face_descriptors = self._model(face_images).detach().numpy()
        else:
            face_descriptors = self._model(face_images.to(self._device)).detach().cpu().numpy()
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
                if (distances[closest] <= self._distance_threshold and
                        (distances <= self._distance_threshold).sum() >= self._neighbour_count_threshold):
                    updated_descriptor = (face_descriptor * self._descriptor_update_rate +
                                          identity['descriptors'][closest] * (1.0 - self._descriptor_update_rate))
                    del identity['descriptors'][closest]
                    identity['descriptors'].append(updated_descriptor)
                else:
                    del identity['descriptors'][0]
                    identity['descriptors'].append(face_descriptor)
        identity['tracklet_ids'].add(tracklet_id)
        return identity

    def reidentify_faces(self, face_images, tracklet_ids, qualities=None, use_bgr_colour_model=True):
        assert len(face_images) == len(tracklet_ids)
        number_of_faces = len(face_images)
        if number_of_faces > 0:
            face_ids = [0] * len(face_images)
            if qualities is None:
                qualities = [self._quality_threshold] * len(face_images)
            if len(qualities) < number_of_faces:
                qualities += [self._quality_threshold] * (number_of_faces - len(qualities))
            elif len(qualities) > number_of_faces:
                del qualities[number_of_faces:]

            # These are what we see now
            self._exisiting_tracklet_ids = set(tracklet_ids)
            assert len(self._exisiting_tracklet_ids) == number_of_faces

            # Calculate face descriptors
            face_descriptors = self._compute_face_descriptors(face_images, use_bgr_colour_model)

            # Update conflict list
            sorted_unique_tracklet_ids = sorted(list(self._exisiting_tracklet_ids))
            for idx, tracklet_id1 in enumerate(sorted_unique_tracklet_ids):
                for tracklet_id2 in sorted_unique_tracklet_ids[idx + 1:]:
                    conflict = [tracklet_id1, tracklet_id2]
                    if conflict not in self._conflicts:
                        self._conflicts.append(conflict)

            # See if some of the faces are already tracked
            for idx in range(number_of_faces):
                saved_face_ids = list(self._database.keys())
                for saved_face_id in saved_face_ids:
                    if tracklet_ids[idx] in self._database[saved_face_id]['tracklet_ids']:
                        face_ids[idx] = saved_face_id
                        if qualities[idx] >= self._quality_threshold:
                            self._database[saved_face_id] = self._update_identity(
                                self._database.pop(saved_face_id), [face_descriptors[idx]], tracklet_ids[idx])
                        else:
                            self._database[saved_face_id] = self._database.pop(saved_face_id)
                        break

            # For the remaining faces, add them as unidentified tracklets for now
            remaining_face_indices = [x for x in range(number_of_faces) if face_ids[x] == 0]
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
                saved_face_ids = list(self._database.keys())
                saved_face_descriptors = []
                saved_face_id_list = []
                for saved_face_id in saved_face_ids:
                    saved_face_descriptors += self._database[saved_face_id]['descriptors']
                    saved_face_id_list += [saved_face_id] * len(self._database[saved_face_id]['descriptors'])
                distances = sd.cdist(remaining_face_descriptors, saved_face_descriptors, metric=self.distance_metric)
                max_distance = np.amax(distances)

                # Now compute similarities between new descriptors and existing identities
                similarities = -np.ones((len(remaining_face_descriptors), len(saved_face_ids)), dtype=np.float)
                for idx, descriptor in enumerate(remaining_face_descriptors):
                    neighbours = np.where(distances[idx, :] <= self._distance_threshold)[0]
                    if len(neighbours) >= self._neighbour_count_threshold:
                        neighbour_face_ids, counts = np.unique([saved_face_id_list[x] for x in neighbours],
                                                               return_counts=True)
                        for idx2 in range(len(neighbour_face_ids)):
                            saved_face_id = neighbour_face_ids[idx2]
                            saved_face_idx = saved_face_ids.index(saved_face_id)
                            similarities[idx, saved_face_idx] = \
                                (counts[idx2] * 2 + 1) * max_distance - np.min(
                                    [distances[idx, x2] for x2 in [x for x in neighbours if
                                                                   saved_face_id_list[x] == saved_face_id]])

                # Enforce the constraints
                updated_unidentified_tracklets = {}
                for idx in range(len(updated_unidentified_tracklet_ids)):
                    updated_unidentified_tracklets[updated_unidentified_tracklet_ids[idx]] = idx
                saved_tracklets = {}
                for idx, face_id in enumerate(saved_face_ids):
                    for tracklet_id in self._database[face_id]['tracklet_ids']:
                        saved_tracklets[tracklet_id] = idx
                for conflict in self._conflicts:
                    if conflict[0] in updated_unidentified_tracklets and conflict[1] in saved_tracklets:
                        similarities[updated_unidentified_tracklets[conflict[0]],
                                     saved_tracklets[conflict[1]]] = -1.0
                    if conflict[1] in updated_unidentified_tracklets and conflict[0] in saved_tracklets:
                        similarities[updated_unidentified_tracklets[conflict[1]],
                                     saved_tracklets[conflict[0]]] = -1.0

                # Assign descriptors to existing identities
                similarities[similarities < 0.0] *= (similarities.shape[0] * similarities.shape[1] *
                                                     np.amax(similarities)) ** 2
                rows, cols = linear_sum_assignment(-similarities)
                associations = []
                for [idx1, idx2] in np.vstack((rows, cols)).T:
                    if similarities[idx1, idx2] > 0.0:
                        tracklet_id = updated_unidentified_tracklet_ids[idx1]
                        face_id = saved_face_ids[idx2]
                        associations.append((tracklet_id, face_id))
                        face_ids[remaining_face_indices[idx1]] = face_id
                for association in associations:
                    tracklet_id = association[0]
                    face_id = association[1]
                    self._database[face_id] = self._update_identity(self._database.pop(face_id),
                                                                    self._unidentified_tracklets[tracklet_id],
                                                                    tracklet_id)
                    del self._unidentified_tracklets[tracklet_id]

            # Finally, see if new identities have emerged
            remaining_face_indices = [x for x in range(number_of_faces) if face_ids[x] == 0]
            for idx in remaining_face_indices:
                tracklet_id = tracklet_ids[idx]
                if len(self._unidentified_tracklets[tracklet_id]) >= self._neighbour_count_threshold:
                    self._face_id_counter += 1
                    face_ids[idx] = self._face_id_counter
                    self._database[face_ids[idx]] = self._update_identity({'descriptors': [], 'tracklet_ids': set()},
                                                                          self._unidentified_tracklets[tracklet_id],
                                                                          tracklet_id)
                    del self._unidentified_tracklets[tracklet_id]

            self._clean_database()
            return face_ids
        else:
            return []


class FaceReidentifierEx(FaceReidentifier):
    def __init__(self, reidentification_interval=8, minimum_tracklet_length=6,
                 minimum_face_size=0.0, face_margin=(0.225, 0.225, 0.225, 0.225),
                 exclude_chin_points=True, equalise_histogram=True,
                 normalised_face_size=224, *args, **kwargs):
        super(FaceReidentifierEx, self).__init__(*args, **kwargs)
        self._reidentification_interval = max(1, int(reidentification_interval))
        self._minimum_tracklet_length = max(1, int(minimum_tracklet_length))
        self._minimum_face_size = max(0.0, float(minimum_face_size))
        assert len(face_margin) == 4
        self._face_margin = face_margin
        self._exclude_chin_points = bool(exclude_chin_points)
        self._equalise_histogram = bool(equalise_histogram)
        self._normalised_face_size = max(1, int(normalised_face_size))
        self._tracking_context = {}
        self._reidentification_countdown = self._reidentification_interval

    @property
    def reidentification_interval(self):
        return self._reidentification_interval

    @reidentification_interval.setter
    def reidentification_interval(self, value):
        self._reidentification_interval = max(1, int(value))

    @property
    def minimum_tracklet_length(self):
        return self._minimum_tracklet_length

    @minimum_tracklet_length.setter
    def minimum_tracklet_length(self, value):
        self._minimum_tracklet_length = max(1, int(value))

    @property
    def minimum_face_size(self):
        return self._minimum_face_size

    @minimum_face_size.setter
    def minimum_face_size(self, value):
        self._minimum_face_size = max(0.0, float(value))

    @property
    def face_margin(self):
        return self._face_margin

    @face_margin.setter
    def face_margin(self, value):
        assert len(value) == 4
        self._face_margin = value

    @property
    def exclude_chin_points(self):
        return self._exclude_chin_points

    @exclude_chin_points.setter
    def exclude_chin_points(self, value):
        self._exclude_chin_points = bool(value)

    @property
    def equalise_histogram(self):
        return self._equalise_histogram

    @equalise_histogram.setter
    def equalise_histogram(self, value):
        self._equalise_histogram = bool(value)

    @property
    def normalised_face_size(self):
        return self._normalised_face_size

    @normalised_face_size.setter
    def normalised_face_size(self, value):
        self._normalised_face_size = max(1, int(value))

    def reidentify_tracked_faces(self, frame, tracked_faces, force_reidentification=False,
                                 ignore_minimum_tracklet_length=False, ignore_quality=False,
                                 use_bgr_colour_model=True):
        # Update tracking context
        for tracklet_id in self._tracking_context.keys():
            self._tracking_context[tracklet_id]['tracked'] = False
        for face in tracked_faces:
            tracklet_id = face['id']

            # Update tracklet length
            if tracklet_id in self._tracking_context:
                self._tracking_context[tracklet_id]['tracklet_length'] += 1
            else:
                self._tracking_context[tracklet_id] = {'tracklet_length': 1, 'face_id': 0}
            self._tracking_context[tracklet_id]['tracked'] = True

            # Extract face information
            if 'face_image' in face:
                self._tracking_context[tracklet_id]['face_image'] = face['face_image']
            else:
                self._tracking_context[tracklet_id]['face_image'] = None
            if 'facial_landmarks' in face:
                self._tracking_context[tracklet_id]['facial_landmarks'] = face['facial_landmarks']
            else:
                self._tracking_context[tracklet_id] = None
            if 'roll' in face:
                self._tracking_context[tracklet_id]['head_pose'] = (0.0, 0.0, face['roll'])
            else:
                self._tracking_context[tracklet_id]['head_pose'] = None
            if 'quality' in face:
                self._tracking_context[tracklet_id]['quality'] = face['quality']
            else:
                if ('facial_landmarks' in face and (
                        face['facial_landmarks'][:, 0].min() <= 0.0 or
                        face['facial_landmarks'][:, 1].min() <= 0.0 or
                        frame is not None and face['facial_landmarks'][:, 0].max() >= frame.shape[1] or
                        frame is not None and face['facial_landmarks'][:, 1].max() >= frame.shape[0] or
                        max(face['facial_landmarks'][:, 0].max() - face['facial_landmarks'][:, 0].min(),
                            face['facial_landmarks'][:, 1].max() - face['facial_landmarks'][:, 1].min()) <
                        self._minimum_face_size)):
                    self._tracking_context[tracklet_id]['quality'] = self.quality_threshold - 1.0
                elif 'most_recent_fitting_scores' in face:
                    self._tracking_context[tracklet_id]['quality'] = np.max(face['most_recent_fitting_scores'])
                else:
                    self._tracking_context[tracklet_id]['quality'] = self.quality_threshold
        for tracklet_id in list(self._tracking_context.keys()):
            if not self._tracking_context[tracklet_id]['tracked']:
                del self._tracking_context[tracklet_id]

        # Manage reidentification countdown
        if self._reidentification_countdown > 0:
            self._reidentification_countdown -= 1

        # Reidentify the faces
        if force_reidentification or self._reidentification_countdown <= 0:
            self._reidentification_countdown = self._reidentification_interval
            tracklets_to_be_identified = []
            for tracklet_id in self._tracking_context.keys():
                if (ignore_minimum_tracklet_length or
                        self._tracking_context[tracklet_id]['tracklet_length'] >=
                        self._minimum_tracklet_length):
                    face_image = self._tracking_context[tracklet_id]['face_image']
                    if face_image is not None:
                        if (face_image.shape[0] != self._normalised_face_size or
                                face_image.shape[1] != self._normalised_face_size):
                            face_image = cv2.resize(face_image, (self._normalised_face_size,
                                                                 self._normalised_face_size))
                    elif frame is not None and self._tracking_context[tracklet_id]['facial_landmarks'] is not None:
                        face_image = extract_face_image(frame,
                                                        self._tracking_context[tracklet_id]['facial_landmarks'],
                                                        (self._normalised_face_size, self._normalised_face_size),
                                                        self._face_margin,
                                                        self._tracking_context[tracklet_id]['head_pose'],
                                                        exclude_chin_points=self._exclude_chin_points)[0]
                    if face_image is not None:
                        if self._equalise_histogram:
                            self._tracking_context[tracklet_id]['face_image'] = equalise_histogram(
                                face_image, use_bgr_colour_model=use_bgr_colour_model)
                        else:
                            self._tracking_context[tracklet_id]['face_image'] = face_image
                        tracklets_to_be_identified.append(tracklet_id)
            face_images = [self._tracking_context[x]['face_image'] for x in tracklets_to_be_identified]
            if ignore_quality:
                qualities = None
            else:
                qualities = [self._tracking_context[x]['quality'] for x in tracklets_to_be_identified]
            face_ids = self.reidentify_faces(face_images, tracklets_to_be_identified, qualities,
                                             use_bgr_colour_model=use_bgr_colour_model)
            for idx, tracklet_id in enumerate(tracklets_to_be_identified):
                self._tracking_context[tracklet_id]['face_id'] = face_ids[idx]

        # We are done
        result = {}
        for tracklet_id in self._tracking_context.keys():
            result[tracklet_id] = {'face_id': self._tracking_context[tracklet_id]['face_id'],
                                   'face_image': self._tracking_context[tracklet_id]['face_image']}
        return result

    def reset(self, reset_reidentification_countdown=True, reset_face_id_counter=True):
        super(FaceReidentifierEx, self).reset(reset_face_id_counter)
        self._tracking_context.clear()
        if reset_reidentification_countdown:
            self._reidentification_countdown = self._reidentification_interval
