import inspect
import os
import pathlib
import pickle
import random
from _operator import itemgetter
from fnmatch import fnmatch

import cv2
import numpy as np


class Util:
    @staticmethod
    def args_to_attributes(self, frame):
        args, _, _, values = inspect.getargvalues(frame)
        for key, value in [(i, values[i]) for i in args]:
            setattr(self, key, value)


class ExamplesListing:
    def __init__(self, dict_video_label, dict_label_encoding):
        Util.args_to_attributes(self, inspect.currentframe())

    @classmethod
    def __load_datapaths(cls, base_path, file_ext_list):
        pattern = '*['
        for ext in file_ext_list:
            pattern += ext + '|'
        pattern = pattern[:-1] + ']'
        paths = []
        for path, subdirs, files in os.walk(base_path):
            for name in files:
                if fnmatch(name, pattern):
                    paths.append(str(pathlib.PurePath(path, name)))
        return paths

    @classmethod
    def __get_label_from_path(cls, path):
        return path.split('/')[-2]

    @classmethod
    def __gnerate_labels(cls,base_path, file_ext_list):
        paths = ExamplesListing.__load_datapaths(base_path, file_ext_list)

        dict_label_encoding = {}
        dict_video_label = {}

        for path in paths:
            label = ExamplesListing.__get_label_from_path(path)
            if not label in dict_label_encoding:
                dict_label_encoding[label] = len(dict_label_encoding)
            dict_video_label[path] = label

        for key in dict_label_encoding:
            value = dict_label_encoding[key]
            encoded = np.zeros(len(dict_label_encoding))
            encoded[value] = 1
            dict_label_encoding[key] = tuple(encoded)

        return dict_video_label, dict_label_encoding

    @classmethod
    def from_path(cls, base_path, file_ext_list, pickle_path=None):
        dict_video_label, dict_label_encoding = ExamplesListing.__gnerate_labels(base_path, file_ext_list)
        if pickle_path is not None:
            params = {"dict_video_label": dict_video_label, "dict_label_encoding": dict_label_encoding}
            pickle.dump(params, open(pickle_path, "wb"))
        return ExamplesListing(dict_video_label, dict_label_encoding)

    @classmethod
    def from_pickle(cls, pickle_file):
        with open(pickle_file, mode='rb') as f:
            params = pickle.load(f)
        return ExamplesListing(params["dict_video_label"], params["dict_label_encoding"])

    def to_examples_set(self):
        return ExamplesSet.from_examples_listing(self)

class ExamplesSet:
    def __init__(self, examples_list, class_labels_dict):
        Util.args_to_attributes(self, inspect.currentframe())

    @classmethod
    def from_examples_listing(cls, examples_listing, pickle_path=None):

        class_labels_dict =  [{v: k} for k, v in examples_listing.dict_label_encoding.items()]
        examples_list = []
        list_video_label = [[k, v] for k, v in examples_listing.dict_video_label.items()]
        for data in list_video_label:
            examples_list.append([data[0], examples_listing.dict_label_encoding[data[1]]])
        random.shuffle(examples_list)
        if pickle_path is not None:
            params = {"examples_list": examples_list, "class_labels_dict":class_labels_dict}
            pickle.dump(params, open(pickle_path, "wb"))
        return ExamplesSet(examples_list, class_labels_dict)

    @classmethod
    def from_pickle(cls, pickle_file):
        with open(pickle_file, mode='rb') as f:
            params = pickle.load(f)
        return  ExamplesSet(params["examples_list"], params["class_labels_dict"])

    def split(self, split_fractions):
        start_index = 0
        splits = []

        for i in range(len(split_fractions) - 1):
            end_index = int(split_fractions[i] * len(self.examples_list))
            splits.append(ExamplesSet(self.examples_list[start_index:start_index + end_index], self.class_labels_dict))
            start_index += end_index
        splits.append(ExamplesSet(self.examples_list[start_index:], self.class_labels_dict))
        return splits

    def to_dataset(self, frame_count=10, resize_width=None, resize_height=None, to_gray=False):
        return DataSet(self, frame_count, resize_width, resize_height, to_gray)

class DataSet:
    def __init__(self, examples_set, frame_count=10, resize_width=None, resize_height=None, to_gray=False):
        Util.args_to_attributes(self, inspect.currentframe())

        capture = cv2.VideoCapture(examples_set.examples_list[0][0])
        ret, frame = capture.read()
        self.size = len(examples_set.examples_list)
        self.height = frame.shape[0]
        self.width = frame.shape[1]
        self.channels = frame.shape[2]

        if resize_width is None:
            self.resize_width = self.width
        if resize_height == None:
            self.resize_height = self.height
        if to_gray:
            self.resize_channels = 1
        else:
            self.resize_channels = self.channels

    def get_paths(self):
        return list(map(itemgetter(0), self.examples_set.examples_list))

    def get_labels(self):
        return list(map(itemgetter(1), self.examples_set.examples_list))

    def get_X_y(self):
        random.shuffle(self.examples_set.examples_list)
        X, y = [], []
        for example in self.examples_set.examples_list:
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
            random.shuffle(self.examples_set.examples_list)
            for batch_i in range(0, len(self.examples_set.examples_list), batch_size):
                X, y = [], []
                for example in self.examples_set.examples_list[batch_i:batch_i + batch_size]:
                    video, label = self.__prepare_example(example)
                    X.append(video)
                    y.append(label)

                yield np.array(X), np.array(y)

    def get_shape(self):
        return (self.frame_count, self.resize_width, self.resize_height, self.resize_channels)

    def get_class_count(self):
        return len(self.examples_set.class_labels_dict)

if __name__ == '__main__':
    ExamplesListing.from_path("dataset", [".avi"], pickle_path="PathLabelListing.pkl")
    examples_listing = ExamplesListing.from_pickle("PathLabelListing.pkl")
    ExamplesSet.from_examples_listing(examples_listing, "ExamplesSet.pkl")
    examples_set = ExamplesSet.from_pickle("ExamplesSet.pkl")

    print(examples_set.to_dataset().get_X_y())



