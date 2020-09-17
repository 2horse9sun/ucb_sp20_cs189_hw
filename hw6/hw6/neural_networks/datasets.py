"""
Author: Sophia Sanborn
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas
"""

import numpy as np
import math
from neural_networks.utils.data_processing import normalize, standardize
from neural_networks.utils.data_structures import integers_to_one_hot


def initialize_dataset(
    name, batch_size=50,
):

    if name == "higgs":
        training_set = np.load("datasets/higgs/higgs_train_data.npy")
        training_labels = np.load("datasets/higgs/higgs_train_labels.npy")

        validation_set = np.load("datasets/higgs/higgs_val_data.npy")
        validation_labels = np.load("datasets/higgs/higgs_val_labels.npy")

        test_set = np.load("datasets/higgs/higgs_test_data.npy")
        test_labels = np.zeros((test_set.shape[0], 2))

        dataset = Dataset(
            training_set=training_set,
            training_labels=training_labels,
            validation_set=validation_set,
            validation_labels=validation_labels,
            test_set=test_set,
            test_labels=test_labels,
            batch_size=batch_size,
        )
        return dataset

    elif name == "sinewave":
        training_set = np.load("datasets/sinewave/sinewave_train_data.npy")
        training_labels = np.load("datasets/sinewave/sinewave_train_labels.npy")

        validation_set = np.load("datasets/sinewave/sinewave_val_data.npy")
        validation_labels = np.load("datasets/sinewave/sinewave_val_labels.npy")

        test_set = np.load("datasets/sinewave/sinewave_test_data.npy")
        test_labels = np.load("datasets/sinewave/sinewave_test_labels.npy")

        dataset = Dataset(
            training_set=training_set,
            training_labels=training_labels,
            validation_set=validation_set,
            validation_labels=validation_labels,
            test_set=test_set,
            test_labels=test_labels,
            batch_size=batch_size,
        )
        return dataset

    elif name == "iris":
        training_set = np.load("datasets/iris/iris_train_data.npy")
        training_labels = np.load("datasets/iris/iris_train_labels.npy")

        validation_set = np.load("datasets/iris/iris_val_data.npy")
        validation_labels = np.load("datasets/iris/iris_val_labels.npy")

        test_set = np.load("datasets/iris/iris_test_data.npy")
        test_labels = np.load("datasets/iris/iris_test_labels.npy")

        dataset = Dataset(
            training_set=training_set,
            training_labels=training_labels,
            validation_set=validation_set,
            validation_labels=validation_labels,
            test_set=test_set,
            test_labels=test_labels,
            batch_size=batch_size,
        )
        return dataset

    else:
        raise NotImplementedError


class Data:
    def __init__(
        self, data, batch_size=50, labels=None, out_dim=None,
    ):

        self.data_ = data
        self.labels = labels
        self.out_dim = out_dim
        self.iteration = 0
        self.batch_size = batch_size
        self.n_samples = data.shape[0]
        self.samples_per_epoch = math.ceil(self.n_samples / batch_size)

    def shuffle(self):
        idxs = np.arange(self.n_samples)
        np.random.shuffle(idxs)

        self.data_ = self.data_[idxs]
        if self.labels is not None:
            self.labels = self.labels[idxs]

    def sample(self, shuffle=True):
        if self.iteration == 0 and shuffle:
            self.shuffle()

        low = self.iteration * self.batch_size
        high = self.iteration * self.batch_size + self.batch_size

        self.iteration += 1
        self.iteration = self.iteration % self.samples_per_epoch

        if self.labels is not None:
            return self.data_[low:high], self.labels[low:high]
        else:
            return self.data_[low:high]

    def reset(self):
        self.iteration = 0


class Dataset:
    def __init__(
        self,
        training_set,
        training_labels,
        batch_size,
        validation_set=None,
        validation_labels=None,
        test_set=None,
        test_labels=None,
    ):

        self.batch_size = batch_size
        self.n_training = training_set.shape[0]
        self.n_validation = validation_set.shape[0]
        self.out_dim = training_labels.shape[1]

        self.train = Data(
            data=training_set,
            batch_size=batch_size,
            labels=training_labels,
            out_dim=self.out_dim,
        )

        if validation_set is not None:
            self.validate = Data(
                data=validation_set,
                batch_size=batch_size,
                labels=validation_labels,
                out_dim=self.out_dim,
            )

        if test_set is not None:
            self.test = Data(
                data=test_set,
                batch_size=batch_size,
                labels=test_labels,
                out_dim=self.out_dim,
            )
