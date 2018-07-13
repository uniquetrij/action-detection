import inspect
import os
import pathlib
import pickle
import random
import cv2
from fnmatch import fnmatch
from operator import itemgetter
import numpy as np

class LabeledPathListingGenerator:

    def __init__(self, base_path, ext_list):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)
        self.__load_dataset()

    def __load_dataset(self):

        pattern = '*['
        for ext in self.ext_list:
            pattern += ext + '|'
        pattern = pattern[:-1] + ']'
        paths = []
        for path, subdirs, files in os.walk(self.base_path):
            for name in files:
                if fnmatch(name, pattern):
                    paths.append(str(pathlib.PurePath(path, name)))
        return paths

    def gnerate_labels(self, pickle_video_label=None, pickle_label_encoding=None):
        paths = self.__load_dataset()

        dict_label_encoding = {}
        list_video_label = {}

        for path in paths:
            label = self.__get_label_from_path(path)
            if not label in dict_label_encoding:
                dict_label_encoding[label] = len(dict_label_encoding)
            list_video_label[path] = label

        for key in dict_label_encoding:
            value = dict_label_encoding[key]
            encoded = np.zeros(len(dict_label_encoding))
            encoded[value] = 1
            dict_label_encoding[key] = encoded

        list_video_label = [[k, v] for k, v in list_video_label.items()]
        random.shuffle(list_video_label)

        if pickle_video_label is not None:
            self.__save_pickle(list_video_label, pickle_video_label)
        if pickle_label_encoding is not None:
            self.__save_pickle(dict_label_encoding, pickle_label_encoding)

        self.dict_label_encoding = dict_label_encoding
        self.list_video_label = list_video_label

        return DataLoader(pickle_video_label, pickle_label_encoding)

    def __get_label_from_path(self, path):
        return path.split('/')[-2]

    def __save_pickle(self, obj, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


class DataLoader:

    def __init__(self, video_label_pickle, label_encoding_pickle):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)

        self.dict_label_encoding = self.__load_pickle(label_encoding_pickle)
        self.list_video_label = self.__load_pickle(video_label_pickle)
        self.size = len(self.list_video_label)
        self.__generate_video_label_encoded()

    def __generate_video_label_encoded(self):
        self.list_video_label_encoded = []
        for data in self.list_video_label:
            self.list_video_label_encoded.append([data[0], self.get_label_encoding(data[1])])

    def __load_pickle(self, filename):
        with open(filename, 'rb') as handle:
             return pickle.load(handle)

    def get_video_label(self, index = None):
        if index is not None:
            return self.list_video_label[index]
        return self.list_video_label

    def get_label_encoding(self, label = None):
        if label is not None:
            return self.dict_label_encoding[label]
        return self.dict_label_encoding

    def get_video_label_encoded(self, index=None):
        if index is not None:
            return self.list_video_label_encoded[index]
        return self.list_video_label_encoded


    def generate_split(self, split_fractions):
        start_index = 0
        splits = []

        for i in range(len(split_fractions) - 1):
            end_index = int(split_fractions[i] * len(self.list_video_label_encoded))
            splits.append(DataSet(self.list_video_label_encoded[start_index:start_index + end_index]))
            start_index += end_index
        splits.append(DataSet(self.list_video_label_encoded[start_index:]))
        return splits


class DataSet:
    def __init__(self, video_label_encoded):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)

        capture = cv2.VideoCapture(video_label_encoded[0][0])
        ret, frame = capture.read()
        self.size = len(self.video_label_encoded)
        self.height = frame.shape[0]
        self.width = frame.shape[1]
        self.channels = frame.shape[2]

    def set_attributes(self, frame_count=10, resize_width=None, resize_height=None, to_gray=False):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)

        if resize_width is None:
            self.resize_width = self.width
        if resize_height == None:
            self.resize_height = self.height
        if to_gray:
            self.resize_channels = 1
        else:
            self.resize_channels = self.channels

    def get_paths(self):
        return list(map(itemgetter(0), self.video_label_encoded))

    def get_labels(self):
        return list(map(itemgetter(1), self.video_label_encoded))

    def get_X_y(self):
        random.shuffle(self.video_label_encoded)
        X, y = [], []
        for example in self.video_label_encoded:
            video, label = self.__prepare_example(example)
            X.append(video)
            y.append(label)
        return np.array(X), np.array(y)

    def __prepare_example(self, example):
        frames = []
        capture = cv2.VideoCapture(example[0])
        for k in range(self.frame_count):
            ret, frame = capture.read()

            frame = cv2.resize(frame, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
            if self.to_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.array(frame)[:, :, np.newaxis]
            frames.append(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
        video = np.array(frames)
        label = example[1]
        return video, label


    def get_batch_generator(self, batch_size):
        while(True):
            random.shuffle(self.video_label_encoded)
            for batch_i in range(0, len(self.video_label_encoded), batch_size):
                X, y = [], []
                for example in self.video_label_encoded[batch_i:batch_i + batch_size]:
                    video, label = self.__prepare_example(example)
                    X.append(video)
                    y.append(label)

                yield np.array(X), np.array(y)

    def get_shape(self):
        return (self.frame_count, self.resize_width, self.resize_height, self.resize_channels)