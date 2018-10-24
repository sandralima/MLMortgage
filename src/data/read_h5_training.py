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

train_dict = get_metadata_dataset('train')

print(train_dict)
