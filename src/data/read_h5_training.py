# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:18:09 2018

@author: sandr
"""

import numpy as np
import pandas as pd
import os
from os.path import abspath
from pathlib import Path
from inspect import getsourcefile
import glob


PRO_DIR = os.path.join(Path(abspath(getsourcefile(lambda:0))).parents[2], 'data', 'processed') 



#to see the content of c1mill files:
def get_metadata_dataset(dtype):
    try:                          
        files_dict = {}  
        total_num_examples = 0
        h5_path = os.path.join(PRO_DIR, 'chuncks_random_c1mill/cmill_AWS')
        all_files = glob.glob(os.path.join(h5_path, "*.h5"))
        for i, file_path in zip(range(len(all_files)), all_files):    
            with pd.HDFStore(file_path) as dataset_file:                
                dataset_features = dataset_file.select(dtype+'/features', start=0, stop=100000).values #, stop=500000
                nrows = dataset_features.shape[0] # dataset_file.get_storer(self.dtype + '/features').nrows
                dataset_labels = dataset_file.select(dtype+'/labels', start=0, stop=nrows).values
                index_length = len(dataset_file.get_storer(dtype+'/features').attrs.data_columns)
                num_columns = dataset_file.get_storer(dtype+ '/features').ncols - index_length
                num_classes = dataset_file.get_storer(dtype+'/labels').ncols - index_length
                files_dict[i] = {'path': file_path, 'nrows': nrows, 
                             'init_index': total_num_examples, 'end_index': total_num_examples + nrows,
                              'dataset_features' : dataset_features, 'dataset_labels': dataset_labels, 'index_length':index_length,
                              'num_columns': num_columns, 'num_classes': num_classes}        
                total_num_examples += nrows                    
        return files_dict        
    except  Exception  as e:        
        raise ValueError('Error in retrieving the METADATA object: ' + str(e))    
        
        
def get_metadata_dataset1(dtype, max_rows, num_feat, num_class):
    try:                          
        files_dict = {}  
        _total_num_examples = 0        
        dataset_features = [] # np.empty((max_rows, num_feat), dtype=np.float32)
        dataset_labels = [] # np.empty((max_rows,num_class), dtype=np.int8)            
        h5_path = os.path.join(PRO_DIR, 'chuncks_random_c1mill/cmill_AWS')
        all_files = glob.glob(os.path.join(h5_path, "*.h5"))
        for i, file_path in zip(range(len(all_files)), all_files):    
            with pd.HDFStore(file_path) as dataset_file:                
                print(file_path, '...to load')
                total_rows = dataset_file.get_storer(dtype + '/features').nrows
                if (total_rows <= max_rows):
                    max_rows -= total_rows
                    dataset_features.extend(dataset_file.select(dtype+'/features', start=0).values) #, stop=500000                        
                    dataset_labels.extend(dataset_file.select(dtype+'/labels', start=0, stop=total_rows).values)
                else:
                    total_rows = max_rows
                    dataset_features.extend(dataset_file.select(dtype+'/features', start=0, stop=total_rows).values) #, stop=500000
                    dataset_labels.extend(dataset_file.select(dtype+'/labels', start=0, stop=total_rows).values)
                                                                                                    
                _total_num_examples += total_rows                    
                print(file_path, ' loaded in RAM')
                if (total_rows == max_rows):
                    break

        files_dict[0] = {'nrows': _total_num_examples, 'init_index': 0, 'end_index': _total_num_examples,
         'dataset_features' : dataset_features, 'dataset_labels': dataset_labels}        

        return files_dict        
    except  Exception  as e:        
        raise ValueError('Error in retrieving the METADATA object: ' + str(e))            


def get_balance_dataset(dtype, max_rows, num_feat, num_class):
    try:                          
        files_dict = {}          
        h5_path = os.path.join(PRO_DIR, 'chuncks_random_c1mill')
        all_files = glob.glob(os.path.join(h5_path, "*.h5"))
        total_files = len(all_files)            
        lab_classes = ['0', '3', '6', '9', 'C', 'F', 'R']
        
        for nclass in lab_classes:
            files_dict[nclass] = {'nrows': 0, 'init_index': 0, 'end_index': 0,
                          'dataset_features' : [], 'dataset_labels': []}                
                
        _total_num_examples = 0
        records_pfile = max_rows // total_files
        for i, file_path in zip(range(total_files), all_files):    
            with pd.HDFStore(file_path) as dataset_file:                                
                print(file_path, '...to load')
                #columns = dataset_file.get_storer(dtype+'/labels').attrs.non_index_axes[0][1]
                total_records_file=0
                for nclass in (set(lab_classes) - set('C')):
                    print('class No.: ', nclass)                       
                    class_features = dataset_file.select(dtype+'/features',  "index=='"+ str(nclass) + "'", start=0, stop=600000)                    
                    class_labels = dataset_file.select(dtype+'/labels',  "index=='"+ str(nclass) + "'", start=0, stop=600000)                    
                    #class_labels_idx = class_labels.index.tolist()
                    #dataset_file.select(dtype+'/features').iloc[class_labels_idx.get_loc()].values
                    #total_rows = len(class_labels_idx)
                    files_dict[nclass]['dataset_features'].extend(class_features.values) #, stop=500000 
                    files_dict[nclass]['dataset_labels'].extend(class_labels.values)
                    total_rows = len(class_labels.values)
                    files_dict[nclass]['nrows'] += total_rows                    
                    max_rows -= total_rows                                                                                                                            
                    _total_num_examples += total_rows
                    total_records_file += total_rows                    
                    print(file_path, ' class ', nclass,': loaded in RAM')                                        

                if (max_rows <= 0):
                    return files_dict                
                

                print('Majority class: C')                    
                current_nrecords =   records_pfile -  total_records_file             
                class_features = dataset_file.select(dtype+'/features',  'index==C', chunk_size=current_nrecords)                    
                class_labels = dataset_file.select(dtype+'/labels', 'index==C', chunk_size=current_nrecords)                
                total_rows = len(class_labels.values)
                #class_labels_idx = class_labels.index.tolist()
                #total_rows = len(class_labels_idx)
                files_dict['C']['dataset_features'].extend(class_features.values) #, stop=500000 
                files_dict['C']['dataset_labels'].extend(class_labels.values)
                files_dict['C']['nrows'] += total_rows                    
                max_rows -= total_rows                                                                                                                            
                _total_num_examples += total_rows                    
                print(file_path, ' class C: loaded in RAM')

                if (max_rows <= 0):
                    break                

        
        return files_dict        
    except  Exception  as e:        
        raise ValueError('Error in retrieving the METADATA object: ' + str(e))            

    
#train_dict = get_metadata_dataset('train')
#print(train_dict)
        
#train_dict = get_metadata_dataset1('train', 100000, 258, 7)
#print(train_dict)

train_dict = get_balance_dataset('train', 500000, 258, 7)
print(train_dict.keys())
