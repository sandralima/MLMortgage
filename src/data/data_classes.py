"""Module for loading the real datasets."""

import numpy as np


class DataBatch(object):
    """ABC."""

    def __init__(self, in_tuple):
        if in_tuple[0].shape[0] != in_tuple[1].shape[0]:
            raise ValueError('Sizes should match!')
        self.orig = Data(in_tuple)
        self.features, self.labels = in_tuple
        self._num_examples, self._num_classes = self.labels.shape        
        self.weights = self.orig.labels @ get_weights(self.orig.labels) # 30/01/2018: this weights are not used in the tensor model!
        # print("DataBatch weights Matrix Multiplication shape?: ", self.weights.shape)
        # print("DataBatch weights Matrix Multiplication?: ", self.weights)
        self._global_index = 0
    # this method batches the training set in lots of size batch_size, if it reaches the end, concatenates the tail with the front and continues until the num_epoch.
    def next_batch(self, batch_size):
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('batch_size has to be of int type.')
        # if self._global_index == 0:
        #     self.sample()
        if self._global_index + batch_size <= self._num_examples:
            temp_features = self.features[self._global_index:
                                          self._global_index + batch_size, :]
            temp_labels = self.labels[self._global_index:
                                      self._global_index + batch_size]
            self._global_index += batch_size
            # if _global_index has become _num_examples, we need to reset it to
            # zero. Otherwise, we don't change it. The following line does this.
            self._global_index = self._global_index % self._num_examples
        else:
            temp_end = self._global_index + batch_size - self._num_examples
            temp_features = np.concatenate(
                (self.features[self._global_index:, :],
                 self.features[:temp_end, :]),
                axis=0)
            temp_labels = np.concatenate(
                (self.labels[self._global_index:], self.labels[:temp_end]),
                axis=0)
            self._global_index = temp_end
            # self.shuffle()
            self._global_index = 0
        return temp_features, temp_labels, np.array(
            [1.0], dtype=np.dtype('float32'))  # temp_weights

    def shuffle(self):
        """Reshuffle the dataset and its corresponding labels."""
        permutation = np.random.permutation(self._num_examples)
        self.features = self.features[permutation, :]
        self.labels = self.labels[permutation]
        return

    def sample(self):
        """Sample with replacement."""
        probs = self.weights / self.weights.sum()
        gamma = 0  # .8
        probs = gamma * probs + (1 - gamma) / self._num_examples
        indices = np.random.choice(
            self._num_examples, size=self._num_examples, replace=True, p=probs)
        self.features = self.orig.features[indices, :]
        self.labels = self.orig.labels[indices]
        # self.weights = self.weights_orig[indices]

    @property
    def num_examples(self):
        """Get the number of examples in the dataset."""
        return self._num_examples

    @property
    def num_classes(self):
        """Get the number of examples in the dataset."""
        return self._num_classes


class Data(object):
    """ABC."""

    def __init__(self, in_tuple):
        if in_tuple[0].shape[0] != in_tuple[1].shape[0]:
            raise ValueError('Sizes should match!')
        self.features, self.labels = in_tuple
        self._num_examples, self._num_classes = self.labels.shape
        # print("Data Features shape: ", self.features.shape)
        ## print("Data Features: ", self.features)
        ### print("Data column labels: ", self.features.axes[0].tolist()) #numpy.ndarray doesn´t have an attribute named axes
        # print("Data - Data Labels shape: ", self.labels.shape)
        # print("Data Labels: ", self.labels)

    @property
    def num_examples(self):
        """Get the number of examples in the dataset."""
        return self._num_examples

    @property
    def num_classes(self):
        """Get the number of examples in the dataset."""
        return self._num_classes


class Dataset(object):
    """A new class to represent learning datasets."""

    def __init__(self, train_tuple, valid_tuple, test_tuple, feature_columns):
        self.train = DataBatch(train_tuple)
        self.validation = Data(valid_tuple)
        self.test = Data(test_tuple)
        self.feature_columns = feature_columns
        # print('train set:', self.train) #returns an object of type: train set: <data_classes.DataBatch object at 0x0000029C911B04A8>
        # print('validation set:', self.validation) #validation set: <data_classes.Data object at 0x0000029C91164B70>
        # print('test set:', self.test) #test set: <data_classes.Data object at 0x0000029C911B5550>


def get_weights(labels):
    """Get the weights per class."""
    # weights = np.ones_like(self.labels[1, :])
    weights = labels.shape[0] / (1e-8 + labels.sum(axis=0))
    # print(weights)
    # weights = np.array(
    #     [
    #         5.561735, 2.349348, 6.397953, 2.575793, 0.056791, 2.591479,
    #         94.966762
    #     ],
    #     dtype=self.labels.dtype)
    return weights
