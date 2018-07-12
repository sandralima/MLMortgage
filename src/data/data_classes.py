"""Module for loading the real datasets."""

import numpy as np
import pandas as pd
import os
import math
import glob
import random_permutation as rp
from datetime import datetime

RANDOM_SEED = 123 # eliminate for taking from the clock!


class DataBatch(object):
    """ABC."""

    def __init__(self, path, period_array, train_dict=None, in_tuple=None, dtype='train'):
        if (in_tuple!=None):
            self.orig = Data(in_tuple)
            self.features, self.labels = in_tuple
            self._num_examples, self._num_classes = self.labels.shape        
            self.weights = self.orig.labels @ get_weights(self.orig.labels) # 30/01/2018: this weights are not used in the tensor model!
            self._global_index = 0            
        elif (path!=None):            
            self.features, self.labels = None, None
            self.h5_path = path
            self.dtype = dtype
            self.dataset_index = 0   
            self.all_files = glob.glob(os.path.join(self.h5_path, "*.h5"))
            self.dataset = pd.HDFStore(glob.glob(os.path.join(self.h5_path, "*.h5"))[self.dataset_index]) # the first file of the path
            self._num_examples = self.dataset.get_storer('features').nrows
            # self._num_columns = self.dataset.get_storer('features').attrs.num_columns
            self._num_columns = len(self.dataset.get_storer('features').attrs.non_index_axes[0][1]) - len(self.dataset.get_storer('features').attrs.data_columns)
            self._num_classes = self.dataset.get_storer('labels').ncols - len(self.dataset.get_storer('labels').attrs.data_columns)                       
            self._global_index = 0    
            # this is for testing and training sets:
            self.period_range =  period_array #set(range(period_array[0], period_array[1]+1))
            #self.period_features = set(list(self.dataset['features'].index.get_level_values(2)))
            #self.period_inter = self.period_features.intersection(self.period_range)            
            self.train_dict = train_dict            
            if train_dict:
                self._loan_random = rp.CustomRandom(self.train_dict[0]) # np.random.RandomState(RANDOM_SEED)
            if (dtype == 'train' and train_dict == None):
                raise ValueError('DataBatch: The training dictionary was not loaded!')

        
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


    def next_sequential_batch_period(self, batch_size):
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('DataBatch: batch_size has to be of int type.')
        if (self.dataset==None):
            raise ValueError('DataBatch: The file_dataset was not loaded!')                      
                
        if self._global_index + batch_size <= self._num_examples: 
            temp_features = self.dataset.select('features', "PERIOD>=" + str(self.period_range[0]) + ' & PERIOD<=' + str(self.period_range[1]), start=self._global_index, stop=self._global_index + batch_size)
            temp_labels = self.dataset.select('labels', "PERIOD>=" + str(self.period_range[0]) + ' & PERIOD<=' + str(self.period_range[1]), start=self._global_index, stop=self._global_index + batch_size)
            self._global_index += batch_size            
        else:  
            temp_features = self.dataset.select('features', "PERIOD>=" + str(self.period_range[0]) + ' & PERIOD<=' + str(self.period_range[1]), start=self._global_index)
            temp_labels = self.dataset.select('labels', "PERIOD>=" + str(self.period_range[0]) + ' & PERIOD<=' + str(self.period_range[1]), start=self._global_index)            
            self._global_index = 0
            self.dataset_index += 1
            self.dataset.close()
            self.dataset = pd.HDFStore(self.all_files[self.dataset_index]) # the next file of the path
            self._num_examples = self.dataset.get_storer('features').nrows

        return temp_features, temp_labels, np.array([1.0], dtype=np.dtype('float32'))  # temp_weights

        
    def next_sequential_batch(self, batch_size):
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('DataBatch: batch_size has to be of int type.')
        if (self.dataset==None):
            raise ValueError('DataBatch: The file_dataset was not loaded!')                      
                
        if self._global_index + batch_size <= self._num_examples:            
            temp_features = pd.read_hdf(self.dataset, 'features', start=self._global_index, stop=self._global_index + batch_size)
            temp_labels = pd.read_hdf(self.dataset, 'labels', start=self._global_index, stop=self._global_index + batch_size)            
            self._global_index += batch_size            
        else:            
            temp_features = pd.read_hdf(self.dataset, 'features', start=self._global_index)            
            temp_labels = pd.read_hdf(self.dataset, 'labels', start=self._global_index)            
            # hdf = pd.read_hdf('storage.h5', 'd1', where=['A>.5'], columns=['A','B'])
            self._global_index = 0
            self.dataset_index += 1
            
            if (self.dataset_index <len(self.all_files)):
                self.dataset.close()
                self.dataset = pd.HDFStore(self.all_files[self.dataset_index]) # the next file of the path
                self._num_examples = self.dataset.get_storer('features').nrows

        return temp_features, temp_labels, np.array([1.0], dtype=np.dtype('float32'))  # temp_weights


    def next_random_batch_perfiles(self, batch_size): # pending!!
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('DataBatch: batch_size has to be of int type.')
        if (self.h5_path==None):
            raise ValueError('DataBatch: The file_dataset was not loaded!')              
        # all_files = glob.glob(os.path.join(self.h5_path, "*.h5"))
        records_per_file = math.ceil(np.float32(batch_size / len(self.all_files)))     
        
        #period_range =  set(range(self.period_range[0], self.period_range[1]+1))        
        features_list = self.dataset.get_storer('features').attrs.non_index_axes[0][1][3:]
        temp_features = pd.DataFrame(None,columns=features_list)
        labels_list = self.dataset.get_storer('labels').attrs.non_index_axes[0][1][3:]
        temp_labels = pd.DataFrame(None,columns=labels_list)        
        for file_path in self.all_files:
            # if self.dataset.is_open: self.dataset.close()
            self.dataset = pd.HDFStore(file_path) # the first file of the path
            self._num_examples = self.dataset.get_storer('features').nrows
            self._num_columns = self.dataset.get_storer('features').ncols - len(self.dataset.get_storer('features').attrs.data_columns)
            self._num_classes = self.dataset.get_storer('labels').ncols - len(self.dataset.get_storer('labels').attrs.data_columns)
            # random_loan= np.random.sample(range(self._num_examples), k=records_per_file) # if one is after the training dates?
            period_random = np.random.RandomState()
            for i in range(records_per_file):                
                while True:
                    try:
                        random_loan = self._loan_random.randint(self._num_examples)
                        loan_id = self.dataset.select('features', "PERIOD>=" + str(self.period_range[0]) + ' & PERIOD<=' + str(self.period_range[1]),start=random_loan, stop=random_loan+1).index.get_level_values(0)[0]
                        if str(loan_id):                    
                            df_features = self.dataset.select('features', "PERIOD>=" + str(self.period_range[0]) + ' & PERIOD<=' + str(self.period_range[1]) + ' & LOAN_ID=' + str(loan_id))
                            df_labels = self.dataset.select('labels', "PERIOD>=" + str(self.period_range[0]) + ' & PERIOD<=' + str(self.period_range[1]) + ' & LOAN_ID=' + str(loan_id))
                            # df_features = self.dataset['features'].loc[(loan_id, slice(None), slice(None)), :]                
                            # df_labels = self.dataset['labels'].loc[(loan_id, slice(None), slice(None)), :]
                            if (df_features.shape[0] > 0):
                                r_period = period_random.randint(df_features.shape[0])
                                temp_features = pd.concat([temp_features, df_features.iloc[r_period, :].to_frame().T], ignore_index=True, copy=False)
                                temp_labels = pd.concat([temp_labels, df_labels.iloc[r_period, :].to_frame().T], ignore_index=True, copy=False)                        
                                break
                    except Exception as e:
                        print('Invalid Loan: ' + str(e))
                        
            print('temp_features')
            self.dataset.close()                            
        return temp_features, temp_labels, np.array([1.0], dtype=np.dtype('float32'))  # temp_weights
    
    def next_random_batch(self, batch_size): # pending!! --_perfiles
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('DataBatch: batch_size has to be of int type.')
        if (self.h5_path==None):
            raise ValueError('DataBatch: The file_dataset was not loaded!')              

        features_list = self.dataset.get_storer('features').attrs.non_index_axes[0][1][3:]
        temp_features = pd.DataFrame(None,columns=features_list)
        labels_list = self.dataset.get_storer('labels').attrs.non_index_axes[0][1][3:]
        temp_labels = pd.DataFrame(None,columns=labels_list) 
        random_batch = self._loan_random.get_batch(batch_size)
        startTime = datetime.now()                        
        for i in random_batch:                                
            try:                
                partial_number = 0
                values_list = list(self.train_dict.values())
                for e in values_list[1:]:
                    partial_number += e['nrows']
                    if partial_number >= i:
                        break
                if self.dataset.is_open: self.dataset.close()
                self.dataset = pd.HDFStore(e['path']) # the first file of the path
                self._num_examples = self.dataset.get_storer('features').nrows
                self._num_columns = self.dataset.get_storer('features').ncols - len(self.dataset.get_storer('features').attrs.data_columns)
                self._num_classes = self.dataset.get_storer('labels').ncols - len(self.dataset.get_storer('labels').attrs.data_columns)
                true_loan = self._num_examples - (partial_number - i)
                df_features = self.dataset.select('features', start=true_loan, stop=true_loan+1)
                df_labels = self.dataset.select('labels', start=true_loan, stop=true_loan+1)
                temp_features = pd.concat([temp_features, df_features], ignore_index=True, copy=False)
                temp_labels = pd.concat([temp_labels, df_labels], ignore_index=True, copy=False)                        
                # self.dataset.close()                            
            except Exception as e:
                print('Invalid Loan: ' + str(e))
        
        print('Time for Getting' + str(batch_size) +' random elements: ', datetime.now() - startTime)            
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
    
    def __exit__(self, *args):
        if self.dataset.is_open: self.dataset.close()

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

    def __init__(self, train_tuple=None, valid_tuple=None, test_tuple=None, feature_columns=None, train_path=None, valid_path=None, test_path=None, 
                 train_period=[121, 279], valid_period=[280,285], test_period=[286, 304], train_dict=None):
        if (train_tuple!=None and valid_tuple!=None and test_tuple!=None):
            self.train = DataBatch(train_tuple)
            self.validation = Data(valid_tuple)
            self.test = Data(test_tuple)
            self.feature_columns = feature_columns
        elif (train_path==None or valid_path==None or test_path==None):  
            raise ValueError('DataBatch: The path for at least one set was not loaded!')
        else:
            self.train = DataBatch(train_path, train_period, dtype='train', train_dict=train_dict) 
            self.validation = DataBatch(valid_path, valid_period, dtype='valid') # Data((h5_dataset.get('valid/features'), h5_dataset.get('valid/labels')))
            self.test = DataBatch(test_path, test_period, dtype='test') # Data((h5_dataset.get('test/features'), h5_dataset.get('test/labels'))) #if it gives some trouble, it will be loaded at the end.


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
