"""Module for loading the real datasets."""

import numpy as np
import pandas as pd

class DataBatch(object):
    """ABC."""

    def __init__(self, in_tuple=None, h5_dataset=None, dtype='train'):
        if (in_tuple!=None):
            self.orig = Data(in_tuple)
            self.features, self.labels = in_tuple
            self._num_examples, self._num_classes = self.labels.shape        
            self.weights = self.orig.labels @ get_weights(self.orig.labels) # 30/01/2018: this weights are not used in the tensor model!
            self._global_index = 0            
        elif (h5_dataset!=None):
            if h5_dataset.get_storer(dtype + '/features').nrows != h5_dataset.get_storer(dtype + '/labels').nrows:
                raise ValueError('DataBatch: Sizes should match!')  
            self.features, self.labels = None, None
            self.h5_path = h5_dataset._path
            self.dtype = dtype
            self._num_examples, self._num_classes = h5_dataset.get_storer(dtype + '/labels').nrows, h5_dataset.get_storer(dtype + '/labels').ncols            
            self._num_columns = h5_dataset.get_storer('train/features').attrs.num_columns
            self._global_index = 0            

        
    # this method batches the training set in lots of size batch_size, if it reaches the end, concatenates the tail with the front and continues until the num_epoch.
    def next_batch(self, batch_size):
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('batch_size has to be of int type.')
        # if self._global_index == 0:
        #     self.sample()
        # print('self._global_index: ', self._global_index)
        # print('self._global_index end: ', self._global_index + batch_size)
                
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


    def next_ooc_batch(self, batch_size):
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('DataBatch: batch_size has to be of int type.')
        if (self.h5_path==None):
            raise ValueError('DataBatch: The file_dataset was not loaded!')                      
                
        if self._global_index + batch_size <= self._num_examples:            
#            temp_features = self.features.iloc[self._global_index:self._global_index + batch_size, :]            
#            temp_labels = self.labels.iloc[self._global_index:self._global_index + batch_size]
            temp_features = pd.read_hdf(self.h5_path, self.dtype + '/features', start=self._global_index, stop=self._global_index + batch_size)
            temp_labels = pd.read_hdf(self.h5_path, self.dtype + '/labels', start=self._global_index, stop=self._global_index + batch_size)            
            self._global_index += batch_size            
        else:            
#            temp_features = self.features.iloc[self._global_index:, :]
#            temp_labels = self.labels.iloc[self._global_index:, :]                        
            temp_features = pd.read_hdf(self.h5_path, self.dtype + '/features', start=self._global_index)            
            temp_labels = pd.read_hdf(self.h5_path, self.dtype + '/labels', start=self._global_index)            
            # hdf = pd.read_hdf('storage.h5', 'd1', where=['A>.5'], columns=['A','B'])
            self._global_index = 0

        # print('self._global_index: ', self._global_index)            
        temp_features = temp_features.reindex(np.random.permutation(temp_features.index))
        temp_labels = temp_labels.reindex(np.random.permutation(temp_labels.index))            
        return temp_features, temp_labels, np.array([1.0], dtype=np.dtype('float32'))  # temp_weights

    def shuffle(self):
        """Reshuffle the dataset and its corresponding labels."""
        permutation = np.random.permutation(self._num_examples)
        self.features = self.features[permutation, :]
        self.labels = self.labels[permutation]
        return
    
    def shuffle(self, data, labels):
        """Reshuffle the dataset data and its corresponding labels."""
        rows = np.shape(data)[0]
        permutation = np.random.permutation(rows)
        data = data[permutation, :]
        labels = labels[permutation]
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

    @property
    def num_columns(self):
        """Get the number of examples in the dataset."""
        return self._num_columns

class Data(object):
    """ABC."""

    def __init__(self, in_tuple=None):
        if in_tuple !=None:
            if in_tuple[0].shape[0] != in_tuple[1].shape[0]:
                raise ValueError('Sizes should match!')
            self.features, self.labels = in_tuple
            self._num_examples, self._num_classes = self.labels.shape

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

    def __init__(self, train_tuple=None, valid_tuple=None, test_tuple=None, feature_columns=None, h5_dataset=None):
        if (train_tuple!=None and valid_tuple!=None and test_tuple!=None):
            self.train = DataBatch(train_tuple)
            self.validation = Data(valid_tuple)
            self.test = Data(test_tuple)
            self.feature_columns = feature_columns
        elif (h5_dataset!=None):            
            self.train = DataBatch(h5_dataset=h5_dataset, dtype='train')            
            self.validation = DataBatch(h5_dataset=h5_dataset, dtype='valid') # Data((h5_dataset.get('valid/features'), h5_dataset.get('valid/labels')))
            self.test = DataBatch(h5_dataset=h5_dataset, dtype='test') # Data((h5_dataset.get('test/features'), h5_dataset.get('test/labels'))) #if it gives some trouble, it will be loaded at the end.


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
