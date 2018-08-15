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

    def __init__(self, path, period_array, in_tuple=None, dtype='train'):
        if (in_tuple!=None):
            self.orig = Data(in_tuple)
            self.features, self.labels = in_tuple
            self._current_num_examples, self._num_classes = self.labels.shape        
            self.weights = self.orig.labels @ get_weights(self.orig.labels) # 30/01/2018: this weights are not used in the tensor model!
            self._file_index = 0            
        elif (path!=None):            
            self.features, self.labels = None, None
            self.h5_path = path
            self.dtype = dtype
            self._total_num_examples = 0
            self.all_files = glob.glob(os.path.join(self.h5_path, "*.h5"))                                    
            
            if dtype == 'train':                
                self._dict = self.get_metadata_dataset_repeats(1)
                self._loan_random = rp.CustomRandom(self._total_num_examples) # np.random.RandomState(RANDOM_SEED)                
            else:
                self._dict = self.get_metadata_dataset()
            if (self._dict == None):
                raise ValueError('DataBatch: The dictionary was not loaded!')
            
            self.dataset_index = 0 #to record the access to files
            self._file_index = 0 #to record the sequential order inside a file               
            # self.dataset = pd.HDFStore(self.all_files[self.dataset_index]) # the first file of the path
            # self._current_num_examples = self.dataset.get_storer(self.dtype+'/features').nrows
            # self._num_columns = self.dataset.get_storer('features').attrs.num_columns
            self.index_length = len(self._dict[self.dataset_index]['dataset'].get_storer(self.dtype+'/features').attrs.data_columns)            
            self._num_columns = self._dict[self.dataset_index]['dataset'].get_storer(self.dtype+ '/features').ncols - self.index_length
            self._num_classes = self._dict[self.dataset_index]['dataset'].get_storer(self.dtype+'/labels').ncols - self.index_length   
            self.features_list = self._dict[self.dataset_index]['dataset'].get_storer(self.dtype+'/features').attrs.non_index_axes[0][1][self.index_length:]
            self.labels_list = self._dict[self.dataset_index]['dataset'].get_storer(self.dtype+'/labels').attrs.non_index_axes[0][1][self.index_length:]                        
            self.period_range =  period_array #set(range(period_array[0], period_array[1]+1))
            #self.period_features = set(list(self.dataset['features'].index.get_level_values(2)))
            #self.period_inter = self.period_features.intersection(self.period_range)            


    def get_metadata_dataset(self):
        try:                          
            files_dict = {}  
            for i, file_path in zip(range(len(self.all_files)), self.all_files):    
                dataset = pd.HDFStore(file_path) # the first file of the path
                nrows = dataset.get_storer(self.dtype + '/features').nrows
                files_dict[i] = {'path': file_path, 'nrows': nrows, 
                                 'init_index': self._total_num_examples, 'end_index': self._total_num_examples + nrows,
                                  'dataset' : dataset}        
                self._total_num_examples += nrows
                print('dict: ', files_dict[i], ' accumulated rows: ', self._total_num_examples)
                # if dataset.is_open: dataset.close()
            return files_dict        
        except  Exception  as e:        
            raise ValueError('Error in retrieving the METADATA object: ' + str(e))    

    def get_metadata_dataset_repeats(self, repeats):
        try:                          
            files_dict = {}  
            index = 0
            for z in range(repeats):
                for file_path in self.all_files:
                    dataset = pd.HDFStore(file_path) # the first file of the path
                    nrows = dataset.get_storer(self.dtype + '/features').nrows
                    files_dict[index] = {'path': file_path, 'nrows': nrows, 
                              'init_index': self._total_num_examples, 'end_index': self._total_num_examples + nrows,
                              'dataset' : dataset}        
                    self._total_num_examples += nrows
                    print('dict: ', files_dict[index], ' total rows: ', self._total_num_examples)
                    index += 1
                    # if dataset.is_open: dataset.close()
            return files_dict        
        except  Exception  as e:        
            raise ValueError('Error in retrieving the METADATA object: ' + str(e))            

    # this method batches the training set in lots of size batch_size, if it reaches the end, concatenates the tail with the front and continues until the num_epoch.
    def next_batch(self, batch_size):
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('batch_size has to be of int type.')
        # if self._file_index == 0:
        #     self.sample()
        # print('self._file_index: ', self._file_index)
        # print('self._file_index end: ', self._file_index + batch_size)
                
        if self._file_index + batch_size <= self._current_num_examples:
            temp_features = self.features[self._file_index:
                                          self._file_index + batch_size, :]
            temp_labels = self.labels[self._file_index:
                                      self._file_index + batch_size]
            self._file_index += batch_size
            # if _global_index has become _num_examples, we need to reset it to
            # zero. Otherwise, we don't change it. The following line does this.
            self._file_index = self._file_index % self._current_num_examples
        else:
            temp_end = self._file_index + batch_size - self._current_num_examples
            temp_features = np.concatenate(
                (self.features[self._file_index:, :],
                 self.features[:temp_end, :]),
                axis=0)
            temp_labels = np.concatenate(
                (self.labels[self._file_index:], self.labels[:temp_end]),
                axis=0)
            self._file_index = temp_end
            # self.shuffle()
            self._file_index = 0
            
        return temp_features, temp_labels, np.array(
            [1.0], dtype=np.dtype('float32'))  # temp_weights


    def next_sequential_batch_period(self, batch_size):
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('DataBatch: batch_size has to be of int type.')
        if (self.dataset==None):
            raise ValueError('DataBatch: The file_dataset was not loaded!')                      
                
        if self._file_index + batch_size <= self._current_num_examples: 
            temp_features = self.dataset.select('features', "PERIOD>=" + str(self.period_range[0]) + ' & PERIOD<=' + str(self.period_range[1]), start=self._file_index, stop=self._file_index + batch_size)
            temp_labels = self.dataset.select('labels', "PERIOD>=" + str(self.period_range[0]) + ' & PERIOD<=' + str(self.period_range[1]), start=self._file_index, stop=self._file_index + batch_size)
            self._file_index += batch_size            
        else:  
            temp_features = self.dataset.select('features', "PERIOD>=" + str(self.period_range[0]) + ' & PERIOD<=' + str(self.period_range[1]), start=self._file_index)
            temp_labels = self.dataset.select('labels', "PERIOD>=" + str(self.period_range[0]) + ' & PERIOD<=' + str(self.period_range[1]), start=self._file_index)            
            self._file_index = 0
            self.dataset_index += 1
            self.dataset.close()
            self.dataset = pd.HDFStore(self.all_files[self.dataset_index]) # the next file of the path
            self._current_num_examples = self.dataset.get_storer('features').nrows

        return temp_features, temp_labels, np.array([1.0], dtype=np.dtype('float32'))  # temp_weights

        
    def next_sequential_batch(self, batch_size):
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('DataBatch: batch_size has to be of int type.')
        if (self._dict[self.dataset_index]['dataset']==None):
            raise ValueError('DataBatch: The file_dataset was not loaded!')                      
                
        if self._file_index + batch_size <= self._dict[self.dataset_index]['nrows']:            
            temp_features = pd.read_hdf(self._dict[self.dataset_index]['dataset'], self.dtype+'/features', start=self._file_index, stop=self._file_index + batch_size)
            temp_labels = pd.read_hdf(self._dict[self.dataset_index]['dataset'], self.dtype+'/labels', start=self._file_index, stop=self._file_index + batch_size)
            self._file_index += batch_size
        else:            
            temp_features = pd.read_hdf(self._dict[self.dataset_index]['dataset'], self.dtype+'/features', start=self._file_index)            
            temp_labels = pd.read_hdf(self._dict[self.dataset_index]['dataset'], self.dtype+'/labels', start=self._file_index)            
            # hdf = pd.read_hdf('storage.h5', 'd1', where=['A>.5'], columns=['A','B'])
            self._file_index = 0
            self.dataset_index += 1
            
            if (self.dataset_index >= len(self.all_files)):
                self.dataset_index = 0
            
            # self.dataset.close()
            # self.dataset = pd.HDFStore(self.all_files[self.dataset_index]) # the next file of the path
            # self._current_num_examples = self.dataset.get_storer(self.dtype+'/features').nrows
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
            self._current_num_examples = self.dataset.get_storer('features').nrows
            self._num_columns = self.dataset.get_storer('features').ncols - len(self.dataset.get_storer('features').attrs.data_columns)
            self._num_classes = self.dataset.get_storer('labels').ncols - len(self.dataset.get_storer('labels').attrs.data_columns)
            # random_loan= np.random.sample(range(self._num_examples), k=records_per_file) # if one is after the training dates?
            period_random = np.random.RandomState()
            for i in range(records_per_file):                
                while True:
                    try:
                        random_loan = self._loan_random.randint(self._current_num_examples)
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
    

    def next_random_batch(self, batch_size): # pending!! --_exp
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('DataBatch: batch_size has to be of int type.')
        if (self.h5_path==None):
            raise ValueError('DataBatch: The file_dataset was not loaded!')              
        # all_files = glob.glob(os.path.join(self.h5_path, "*.h5"))        
        
        #period_range =  set(range(self.period_range[0], self.period_range[1]+1))            
        
        #temp_features = pd.DataFrame(None,columns=self.features_list)        
        #temp_labels = pd.DataFrame(None,columns=self.labels_list)  
        temp_features = np.empty((0,len(self.features_list)))
        temp_labels = np.zeros((0,len(self.labels_list)))
        random_batch = np.array(list(self._loan_random.get_batch(batch_size)))
        #startTime = datetime.now()       
        #partial_number = 0         
        orb_size = 0
        for k, v in self._dict.items():
            try:                
                # startTime1 = datetime.now()                
                #if self.dataset.is_open: 
                #    self.dataset.close()
                    # gc.collect()
                records_per_file = np.logical_and(random_batch>=v['init_index'], random_batch<(v['end_index']))                        
                orb = np.sort(random_batch[records_per_file]) - v['init_index']      
                # print('File: ', k, 'Time for random selection: ', datetime.now() - startTime1, ' records: ',  len(orb))
                if (len(orb)>0):
                    #file_path = v['path']                    
                    #startTime2 = datetime.now()
                    #self.dataset = pd.HDFStore(file_path) # the first file of the path
                    #self._current_num_examples = self.dataset.get_storer(self.dtype + '/features').nrows
                    # String_batch = ', '.join(map(str, orb))
                    # df_features = self.dataset.select(self.dtype + '/features', "level_0 in [" + String_batch + "]")
                    # df_labels = self.dataset.select(self.dtype + '/labels', "level_0 in [" + String_batch + "]")                    
                    #temp_features = np.concatenate((temp_features, self.dataset.select(self.dtype+'/features', where=orb).values)) #this way is heavy
                    #temp_labels = np.concatenate((temp_labels, self.dataset.select(self.dtype+'/labels', where=orb).values))

                    df_features = v['dataset'].select(self.dtype+'/features', where=orb)
                    df_labels = v['dataset'].select(self.dtype+'/labels', where=orb)
                    
                    # df_features = pd.read_hdf(self.dataset, self.dtype+'/features', where=orb) # the same time as above
                    # df_labels = pd.read_hdf(self.dataset, self.dtype+'/labels', where=orb)            

                    #print('File: ', k, 'Time for one file lecture: ', datetime.now() - startTime2, ' records: ',  len(orb))
                    
#                    startTime3 = datetime.now()
                    temp_features = np.concatenate((temp_features, df_features.values))
                    temp_labels = np.concatenate((temp_labels, df_labels.values))
#                    print('File: ', k, 'Time for append: ', datetime.now() - startTime3, ' records: ',  len(orb))
                    
                    #print('File ', k, ': ',file_path, ' Time for one file lecture/append: ', datetime.now() - startTime1, ' records: ',  len(orb))            
                    orb_size += len(orb)
                #partial_number = partial_number + self._current_num_examples
            except Exception as e:
                print('Invalid Range: ' + str(e))                                    

        # the same permutation:       
        permutation = np.random.permutation(len(temp_features))
        temp_features = temp_features[permutation]
        temp_labels = temp_labels[permutation]        
        #np.random.shuffle(temp_features)
        #np.random.shuffle(temp_labels)
        # print('Time for Getting ', orb_size, ' random elements: ', datetime.now() - startTime)                    
        return temp_features, temp_labels, np.array([1.0], dtype=np.dtype('float32'))  # temp_weights    
    
    
    def next_random_batch_ind_access(self, batch_size): # pending!! --_ind_access
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('DataBatch: batch_size has to be of int type.')
        if (self.h5_path==None):
            raise ValueError('DataBatch: The file_dataset was not loaded!')              

        features_list = self.dataset.get_storer('features').attrs.non_index_axes[0][1][self.index_length:]
        temp_features = pd.DataFrame(None,columns=features_list)
        labels_list = self.dataset.get_storer('labels').attrs.non_index_axes[0][1][self.index_length:]
        temp_labels = pd.DataFrame(None,columns=labels_list) 
        random_batch = self._loan_random.get_batch(batch_size)
        startTime = datetime.now()                        
        for i in random_batch:                                
            try:                
                startTime1 = datetime.now()
                partial_number = 0
                values_list = list(self._dict.values())
                for e in values_list:
                    partial_number += e['nrows']
                    if partial_number >= i:
                        break                
                if self.dataset.is_open: self.dataset.close()
                self.dataset = pd.HDFStore(e['path']) # the first file of the path
                self._current_num_examples = self.dataset.get_storer('features').nrows
                self._num_columns = self.dataset.get_storer('features').ncols - len(self.dataset.get_storer('features').attrs.data_columns)
                self._num_classes = self.dataset.get_storer('labels').ncols - len(self.dataset.get_storer('labels').attrs.data_columns)
                true_loan = self._current_num_examples - (partial_number - i)
                df_features = self.dataset.select('features', start=true_loan, stop=true_loan+1)
                df_labels = self.dataset.select('labels', start=true_loan, stop=true_loan+1)
                temp_features = pd.concat([temp_features, df_features], ignore_index=True, copy=False)
                temp_labels = pd.concat([temp_labels, df_labels], ignore_index=True, copy=False)                        
                print('Time for Getting one element: ', datetime.now() - startTime1)            
                # self.dataset.close()                            
            except Exception as e:
                print('Invalid Loan: ' + str(e))
        
        print('Time for Getting' + str(batch_size) +' random elements: ', datetime.now() - startTime)            
        return temp_features, temp_labels, np.array([1.0], dtype=np.dtype('float32'))  # temp_weights    
    
    def shuffle(self):
        """Reshuffle the dataset and its corresponding labels."""
        permutation = np.random.permutation(self._current_num_examples)
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
        probs = gamma * probs + (1 - gamma) / self._current_num_examples
        indices = np.random.choice(
            self._current_num_examples, size=self._current_num_examples, replace=True, p=probs)
        self.features = self.orig.features[indices, :]
        self.labels = self.orig.labels[indices]
        # self.weights = self.weights_orig[indices]

    def total_num_batch(self, batch_size):
        
        total_batch = 0        
        values_list = list(self._dict.values())
        for e in values_list:            
            total_batch += math.ceil(np.float32( e['nrows'] / batch_size))
        return total_batch


    @property
    def total_num_examples(self):
        """Get the number of examples in the dataset."""
        return self._total_num_examples            

    @property
    def num_classes(self):
        """Get the number of examples in the dataset."""
        return self._num_classes

    @property
    def num_columns(self):
        """Get the number of examples in the dataset."""
        return self._num_columns
    
    def __del__(self, *args):
        for k, v in self._dict.items():
            try:
                if v['dataset'].is_open: 
                    v['dataset'].close()
                    print('__del__ ', v['path'], ': File Closed')
            except Exception as e:
                print('__del__ Error Closing Files: ' + str(e))                                    
                
    def __exit__(self, type, value, traceback): # __exit__(self, *args):
        for k, v in self._dict.items():
            try:
                if v['dataset'].is_open: 
                    v['dataset'].close()
                    print('__exit__ ', v['path'], ': File Closed')
            except Exception as e:
                print('__exit__ Error Closing Files: ' + str(e))                                    
        


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
                 train_period=[121, 279], valid_period=[280,285], test_period=[286, 304]):
        if (train_tuple!=None and valid_tuple!=None and test_tuple!=None):
            self.train = DataBatch(train_tuple, train_period)
            self.validation = Data(valid_tuple)
            self.test = Data(test_tuple)
            self.feature_columns = feature_columns
        elif (train_path==None or valid_path==None or test_path==None):  
            raise ValueError('DataBatch: The path for at least one set was not loaded!')
        else:
            self.train = DataBatch(train_path, train_period, dtype='train') 
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
