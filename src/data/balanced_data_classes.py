"""Module for loading the real datasets."""

import numpy as np
import pandas as pd
import os
import math
import glob
import random_permutation as rp
from datetime import datetime
import sys

RANDOM_SEED = 123 # eliminate for taking from the clock!


class DataBatch(object):
    """ABC."""

    def __init__(self, architecture, path, period_array, in_tuple=None, dtype='train'):
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
            self.all_files = glob.glob(os.path.join(self.h5_path, "*.h5"))    

            if (dtype == 'valid'):
                num_exam = architecture['total_num_examples'] # architecture['valid_num_examples']
                self._dict = self.get_metadata_dataset(num_exam)
            elif (dtype== 'test'):
                num_exam = architecture['total_num_examples']
                self._dict = self.get_metadata_dataset(num_exam)
            else:
                num_exam = architecture['total_num_examples']
                self._dict = self.get_balance_dataset(num_exam) #self.get_metadata_dataset(num_exam)
                                                
            if (self._dict == None):
                raise ValueError('DataBatch: The dictionary was not loaded!')
                        
            self.dataset_index = 0 #to record the access to files
            self._file_index = 0 #to record the sequential order inside a file               
            # self.dataset = pd.HDFStore(self.all_files[self.dataset_index]) # the first file of the path
            # self._current_num_examples = self.dataset.get_storer(self.dtype+'/features').nrows
            # self._num_columns = self.dataset.get_storer('features').attrs.num_columns
            self.period_range =  period_array #set(range(period_array[0], period_array[1]+1))
            self._num_columns = architecture['n_input'] # dataset_file.get_storer(self.dtype+ '/features').ncols - self.index_length
            self._num_classes = architecture['n_classes'] # dataset_file.get_storer(self.dtype+'/labels').ncols - self.index_length   

            #self.period_features = set(list(self.dataset['features'].index.get_level_values(2)))
            #self.period_inter = self.period_features.intersection(self.period_range)            
        else: #Dataset empty!
            self._dict = None

    def get_balance_dataset(self, max_rows):
        try:                          
            files_dict = {}          
            #h5_path = os.path.join(PRO_DIR, 'chuncks_random_c1mill')
            #all_files = glob.glob(os.path.join(h5_path, "*.h5"))
            total_files = len(self.all_files)            
            lab_classes = ['0', '3', '6', '9', 'C', 'F', 'R']
            
            for nclass in lab_classes:
                files_dict[nclass] = {'nrows': 0, 'init_index': 0, 'end_index': 0,
                              'dataset_features' : [], 'dataset_labels': []}                
                    
            self._total_num_examples = 0
            records_pfile = max_rows // total_files
            for i, file_path in zip(range(total_files), self.all_files):    
                with pd.HDFStore(file_path) as dataset_file:                                
                    print(file_path, '...to load')
                    #columns = dataset_file.get_storer(dtype+'/labels').attrs.non_index_axes[0][1]
                    total_records_file=0
                    for nclass in (set(lab_classes) - set('C')):
                        print('class No.: ', nclass)                       
                        class_features = dataset_file.select(self.dtype+'/features',  "index=='"+ str(nclass) + "'", start=0, stop=1500000)                    
                        class_labels = dataset_file.select(self.dtype+'/labels',  "index=='"+ str(nclass) + "'", start=0, stop=1500000)                    
                        #class_labels_idx = class_labels.index.tolist()
                        #dataset_file.select(dtype+'/features').iloc[class_labels_idx.get_loc()].values
                        #total_rows = len(class_labels_idx)
                        files_dict[nclass]['dataset_features'].extend(class_features.values) #, stop=500000 
                        files_dict[nclass]['dataset_labels'].extend(class_labels.values)
                        total_rows = len(class_labels.values)
                        files_dict[nclass]['nrows'] += total_rows                    
                        max_rows -= total_rows                                                                                                                            
                        self._total_num_examples += total_rows
                        total_records_file += total_rows                    
                        print(file_path, ' class ', nclass,': loaded in RAM, nrows: ', total_rows)                                        
    
                    if (max_rows <= 0):
                        break
                    
    
                    print('Majority class: C')                    
                    current_nrecords =   records_pfile -  total_records_file             
                    for class_features in dataset_file.select(self.dtype+'/features',  'index==C', chunksize= current_nrecords):
                        files_dict['C']['dataset_features'].extend(class_features.values) #, stop=500000     
                        break
                    
                    total_rows = 0
                    for class_labels in dataset_file.select(self.dtype+'/labels', 'index==C', chunksize=current_nrecords):
                        files_dict['C']['dataset_labels'].extend(class_labels.values)
                        total_rows = len(class_labels.values)
                        break
                                        
                    #class_labels_idx = class_labels.index.tolist()
                    #total_rows = len(class_labels_idx)
                    
                    
                    files_dict['C']['nrows'] += total_rows                    
                    max_rows -= total_rows                                                                                                                            
                    self._total_num_examples += total_rows                    
                    print(file_path, ' class C: loaded in RAM, nrows: ', total_rows)
    
                    if (max_rows <= 0):
                        break                

            for nclass in lab_classes:
                files_dict[nclass]['random_gen'] = rp.CustomRandom(files_dict[nclass]['nrows']-1) 
                
            return files_dict        
        except  Exception  as e:        
            raise ValueError('Error in retrieving the METADATA object: ' + str(e))            
    
            
    def get_metadata_dataset(self, max_rows):
        try:                          
            files_dict = {}  
            self._total_num_examples = 0
            ok_inputs = True
            files_dict[0] = {}
            files_dict[0]['dataset_features'] = [] # np.empty((max_rows, num_feat), dtype=np.float32)
            files_dict[0]['dataset_labels'] = [] # np.empty((max_rows,num_class), dtype=np.int8)            
            for i, file_path in zip(range(len(self.all_files)), self.all_files):    
                with pd.HDFStore(file_path) as dataset_file:                
                    print(file_path, '...to load')
                    total_rows = dataset_file.get_storer(self.dtype + '/features').nrows
                    if (total_rows <= max_rows):
                        max_rows -= total_rows
                        files_dict[0]['dataset_features'].extend(dataset_file.select(self.dtype+'/features', start=0).values) #, stop=500000                        
                        files_dict[0]['dataset_labels'].extend(dataset_file.select(self.dtype+'/labels', start=0, stop=total_rows).values)
                    else:
                        total_rows = max_rows
                        files_dict[0]['dataset_features'].extend(dataset_file.select(self.dtype+'/features', start=0, stop=total_rows).values) #, stop=500000
                        files_dict[0]['dataset_labels'].extend(dataset_file.select(self.dtype+'/labels', start=0, stop=total_rows).values)
                                                                                    
                    if (ok_inputs): 
                        self.index_length = len(dataset_file.get_storer(self.dtype+'/features').attrs.data_columns)            
                        self.features_list = dataset_file.get_storer(self.dtype+'/features').attrs.non_index_axes[0][1][self.index_length:]
                        self.labels_list = dataset_file.get_storer(self.dtype+'/labels').attrs.non_index_axes[0][1][self.index_length:]                        
                        ok_inputs = False                    
                        
                    self._total_num_examples += total_rows                    
                    print(file_path, ' loaded in RAM')
                    if (total_rows == max_rows):
                        break

            files_dict[0]['nrows'] = self._total_num_examples
            files_dict[0]['init_index'] = 0
            files_dict[0]['end_index'] = self._total_num_examples                         
            class_weights = np.sum(files_dict[0]['dataset_labels'], axis=0)
            print('class_weights', class_weights)
            class_weights = np.round(class_weights/np.float32(self._total_num_examples),decimals=3)
            # 1-weights approach:
            class_weights = np.subtract([1]*len(class_weights), class_weights)
            #normalizing 1-weights approach:
            #sumcw = np.sum(class_weights)
            #class_weights = np.round(class_weights/np.float32(sumcw),decimals=3)
            print('class_weights', class_weights)
            files_dict[0]['class_weights']  = class_weights

            return files_dict        
        except  Exception  as e:        
            raise ValueError('Error in retrieving the METADATA object: ' + str(e))    

            
#    def get_metadata_dataset_repeats(self, repeats):
#        try:                          
#            files_dict = {}  
#            index = 0
#            for z in range(repeats):
#                for file_path in self.all_files:
#                    dataset_file = pd.HDFStore(file_path) # the first file of the path                    
#                    dataset_features = dataset_file.select(self.dtype+'/features', start=0, stop=500000).values # , stop=5000000
#                    nrows = dataset_features.shape[0] # dataset_file.get_storer(self.dtype + '/features').nrows
#                    dataset_labels = dataset_file.select(self.dtype+'/labels', start=0, stop=nrows).values
#                    files_dict[index] = {'path': file_path, 'nrows': nrows, 
#                                 'init_index': self._total_num_examples, 'end_index': self._total_num_examples + nrows,
#                                  'dataset' : dataset_file, 'dataset_features' : dataset_features, 'dataset_labels': dataset_labels}                            
#                    self._total_num_examples += nrows
#                    print('dict: ', files_dict[index], ' total rows: ', self._total_num_examples)
#                    index += 1
#                    # if dataset.is_open: dataset.close()
#            return files_dict        
#        except  Exception  as e:        
#            raise ValueError('Error in retrieving the METADATA object: ' + str(e))            

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
        if (self._dict==None):
            raise ValueError('DataBatch: The dataset was not loaded!')                      
                
        if self._file_index + batch_size <= self._dict[self.dataset_index]['nrows']:            
            # temp_features = pd.read_hdf(self._dict[self.dataset_index]['dataset'], self.dtype+'/features', start=self._file_index, stop=self._file_index + batch_size)
            # temp_labels = pd.read_hdf(self._dict[self.dataset_index]['dataset'], self.dtype+'/labels', start=self._file_index, stop=self._file_index + batch_size)
            # temp_features = self._dict[self.dataset_index]['dataset'].select(self.dtype+'/features', start=self._file_index, stop=self._file_index + batch_size)
            temp_features = np.array(self._dict[self.dataset_index]['dataset_features'][self._file_index: self._file_index + batch_size])
            temp_labels = np.array(self._dict[self.dataset_index]['dataset_labels'][self._file_index: self._file_index + batch_size])
            self._file_index += batch_size
        else:            
            # temp_features = pd.read_hdf(self._dict[self.dataset_index]['dataset'], self.dtype+'/features', start=self._file_index)            
            # temp_labels = pd.read_hdf(self._dict[self.dataset_index]['dataset'], self.dtype+'/labels', start=self._file_index)            
            temp_features = np.array(self._dict[self.dataset_index]['dataset_features'][self._file_index :])
            temp_labels = np.array(self._dict[self.dataset_index]['dataset_labels'][self._file_index :])
            # hdf = pd.read_hdf('storage.h5', 'd1', where=['A>.5'], columns=['A','B'])
            self._file_index = 0
            #self.dataset_index += 1
            
            #if (self.dataset_index >= len(self.all_files)):
            #    self.dataset_index = 0
            
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
        temp_features = np.empty((batch_size,len(self.features_list)))
        temp_labels = np.zeros((batch_size,len(self.labels_list)))
        random_batch = np.array(list(self._loan_random.get_batch(batch_size)))
        #startTime = datetime.now()       
        #partial_number = 0         
        orb_size = 0        
        for k, v in self._dict.items():
            try:                
                #startTime1 = datetime.now()                
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

                    # df_features = v['dataset_features'][orb, :]
                    temp_features[orb_size : orb_size + len(orb), :] = np.array([v['dataset_features'][index] for index in orb]) # v['dataset_features'][orb, :]
                    # df_labels = v['dataset_labels'][orb, :]
                    temp_labels[orb_size : orb_size + len(orb), :] = np.array([v['dataset_labels'][index] for index in orb]) # v['dataset_labels'][orb, :]                    
                    # df_features = v['dataset'].select(self.dtype+'/features', where=orb)
                    #df_labels = v['dataset'].select(self.dtype+'/labels', where=orb)
                    
                    # df_features = pd.read_hdf(self.dataset, self.dtype+'/features', where=orb) # the same time as above
                    # df_labels = pd.read_hdf(self.dataset, self.dtype+'/labels', where=orb)            

                    #print('File: ', k, 'Time for one file lecture: ', datetime.now() - startTime2, ' records: ',  len(orb))
                    
#                    startTime3 = datetime.now()
                    # temp_features = np.concatenate((temp_features, df_features))
                    # temp_labels = np.concatenate((temp_labels, df_labels))
#                    print('File: ', k, 'Time for append: ', datetime.now() - startTime3, ' records: ',  len(orb))
                    
                    # print('File ', k, ': ',file_path, ' Time for one file lecture/append: ', datetime.now() - startTime1, ' records: ',  len(orb))            
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
        #print('Time for Getting ', orb_size, ' random elements: ', datetime.now() - startTime)                    
        return temp_features, temp_labels #, np.array([1.0], dtype=np.dtype('float32'))  # temp_weights    


    def next_balanced_batch(self, batch_size): # pending!! --_exp
        """Get the next batch of the data with the given batch size."""
        if not isinstance(batch_size, int):
            raise TypeError('DataBatch: batch_size has to be of int type.')
        if (self.h5_path==None):
            raise ValueError('DataBatch: The file_dataset was not loaded!')              
        temp_features = np.empty((batch_size, self._num_columns))
        temp_labels = np.zeros((batch_size, self._num_classes))     
        orb_size = 0        
        records_pclass = batch_size // self._num_classes
        for k, v in self._dict.items():
            try:     
                random_batch = np.array(list(v['random_gen'].get_batch(records_pclass)))
                #orb = np.random.randint(low=0, high=v['nrows'], size=records_pclass)
                orb=sorted(random_batch)
                if (len(orb)>0):
                    temp_features[orb_size : orb_size + len(orb), :] = np.array([v['dataset_features'][index] for index in orb]) # v['dataset_features'][orb, :]
                    temp_labels[orb_size : orb_size + len(orb), :] = np.array([v['dataset_labels'][index] for index in orb]) # v['dataset_labels'][orb, :]                    
                    orb_size += len(orb)
            except Exception as e:
                print('Invalid Range: ' + str(e))                                    
                
        if (records_pclass * self._num_classes < batch_size):
            #print('batch_size - orb_size: ', batch_size - orb_size)
            remainders = batch_size - orb_size
            random_batch = np.array(list(self._dict['C']['random_gen'].get_batch(remainders)))
            orb=sorted(random_batch)
            temp_features[orb_size : , :] = np.array([self._dict['C']['dataset_features'][index] for index in orb]) # 
            temp_labels[orb_size : , :] = np.array([self._dict['C']['dataset_labels'][index] for index in orb]) # 

        # the same permutation:       
        permutation = np.random.permutation(len(temp_features))
        temp_features = temp_features[permutation]
        temp_labels = temp_labels[permutation]        
        return temp_features, temp_labels #, np.array([1.0], dtype=np.dtype('float32'))  # temp_weights    
    
    
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
    
    @property
    def class_weights(self):
        return self._dict[0]['class_weights']


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

    def __init__(self, architecture, train_tuple=None, valid_tuple=None, test_tuple=None, feature_columns=None, train_path=None, valid_path=None, test_path=None, 
                 train_period=[121, 279], valid_period=[280,285], test_period=[286, 304]):
        if (train_tuple!=None and valid_tuple!=None and test_tuple!=None):
            self.train = DataBatch(train_tuple, train_period)
            self.validation = Data(valid_tuple)
            self.test = Data(test_tuple)
            self.feature_columns = feature_columns
        elif (train_path==None and valid_path==None and test_path==None):  
            raise ValueError('DataBatch: The path for at least one set was not loaded!')
        else:
            self.train = DataBatch(architecture, train_path, train_period, dtype='train') 
            self.validation = DataBatch(architecture, valid_path, valid_period, dtype='valid') # Data((h5_dataset.get('valid/features'), h5_dataset.get('valid/labels')))
            self.test = DataBatch(architecture, test_path, test_period, dtype='test') # Data((h5_dataset.get('test/features'), h5_dataset.get('test/labels'))) #if it gives some trouble, it will be loaded at the end.


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
