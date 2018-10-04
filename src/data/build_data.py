# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 19:00:55 2018

@author: sandr
"""
import numpy as np
import pandas as pd
import get_raw_data as grd
import logging
import os
from dotenv import find_dotenv, load_dotenv
import data_classes
import Normalizer
import datetime
import glob
from os.path import abspath
from pathlib import Path
from inspect import getsourcefile
from datetime import datetime
import math
import argparse
import sys
import tensorflow as tf


from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder

DT_FLOAT = np.float32 
DT_BOOL = np.uint8
RANDOM_SEED = 123
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.propagate = False # it will not log to console.

RAW_DIR = os.path.join(Path(abspath(getsourcefile(lambda:0))).parents[2], 'data', 'raw') 
PRO_DIR = os.path.join(Path(abspath(getsourcefile(lambda:0))).parents[2], 'data', 'processed') 


def drop_descrip_cols(data):
    '''Exclude from the dataset 'data' the following descriptive columns :
             to_exclude = [
            'LOAN_ID',
            'ASOFMONTH',
            'MONTH',
            'TIME',
            'CURRENT_INVESTOR_CODE_253',
            'LLMA2_VINTAGE_2005',
            'MBA_DELINQUENCY_STATUS',
            'STATE']   
        Args: 
            data (DataFrame): Input Dataset which is modified in place.
        Returns: 
            None
        Raises:        
    '''
    logger.name = 'drop_descrip_cols'
    to_exclude = [
        'LOAN_ID',
        'ASOFMONTH',
        'MONTH',
        'TIME',
        'CURRENT_INVESTOR_CODE_253',
        'LLMA2_VINTAGE_2005',
        'MBA_DELINQUENCY_STATUS',
        'STATE',]
    data.drop(to_exclude, axis=1, inplace=True)
    logger.info('...Descriptive columns Excluded from dataset...')
    return None


def allfeatures_drop_cols(data, columns):
    '''Exclude from the dataset 'data' the descriptive columns as parameters.
        Args: 
            data (DataFrame): Input Dataset which is modified in place.
        Returns: 
            None
        Raises:        
    '''
    logger.name = 'allfeatures_drop_cols'    
    data.drop(columns, axis=1, inplace=True)
    logger.info('...Columns Excluded from dataset...')
    return None

         
        
    
def drop_low_variation(data, low_freq_cols=None):    
    '''Exclude the columns not showing enough population variation. 
    
    Args: 
        data (DataFrame): Input Dataset which is modified in place.
        low_freq_cols (Array of column names): In case of 'None' drop the boolean columns  with a mean less than 0.05 or  abs(100 - mean) < thresh (almost 1).
        Otherwise, drops these columns from the Dataset.
    Returns: 
        Array of String. Array of dropped column names.
    Raises:        
    '''
    logger.name = 'drop_low_variation'
    if low_freq_cols is None:
        low_freq_cols = []
        thresh = 0.05  # this is 5 basis points.
        for name in data.columns.values:
            if data[name].dtype == DT_BOOL:
                temp = 100 * data[name].mean()
                # print(name, temp)
                if temp < thresh or abs(100 - temp) < thresh:
                    data.drop(name, axis=1, inplace=True)
                    # print('Dropped', name)
                    low_freq_cols.append(name)
    else:
        for name in low_freq_cols:
            data.drop(name, axis=1, inplace=True)
            logger.info('Dropped: ' + name)
    logger.info('...Columns Excluded from dataset...')
    logger.info('Size of the database after drop_low_variation:' + str(data.shape))
    return low_freq_cols


def drop_paidoff_reo(data):
    '''Exclude rows: 'MBA_DELINQUENCY_STATUS_0<1 & MBA_DELINQUENCY_STATUS_R<1'.
       Exclude columns: ['MBA_DELINQUENCY_STATUS_0', 'MBA_DELINQUENCY_STATUS_R'].
    
    Args: 
        data (DataFrame): Input Dataset which is modified in place.        
    Returns: 
        None.
    Raises:        
    '''
    logger.name = 'drop_paidoff_reo'    
    logger.info('Size Dataset: '+ str(data.shape))
    data.query(
        'MBA_DELINQUENCY_STATUS_0<1 & MBA_DELINQUENCY_STATUS_R<1',
        inplace=True)    
    data.drop(
        ['MBA_DELINQUENCY_STATUS_0', 'MBA_DELINQUENCY_STATUS_R'],
        axis=1,
        inplace=True)
    logger.info('New Size Dataset: '+ str(data.shape))
    return None


def extract_numeric_labels(data, label_column='MBA_DELINQUENCY_STATUS_next'):
    '''Extract the labels from Dataset, order-and-transform them into numeric labels.
    
    Args: 
        data (DataFrame): Input Dataset which is modified in place.        
        label_column (string): Default 'MBA_DELINQUENCY_STATUS_next'. 
    Returns: 
        DataSeries. Numeric label column.
    Raises:        
    '''    
    logger.name = 'extract_numeric_labels'
    labels = data[label_column]    
    all_labels = labels.value_counts().index.tolist()
    all_labels.sort()    
    dict_labels = dict(zip(all_labels, np.arange(len(all_labels))))
    labels = labels.map(dict_labels) 
    logger.info('mapped labels: ' + str(dict_labels))
    data.drop(label_column, axis=1, inplace=True)
    logger.info('...Labels extracted from Dataset...')
    logger.info('Size of the dataset after extract_labels:' + str(data.shape))
    return labels


def allfeatures_extract_labels(data, columns='MBA_DELINQUENCY_STATUS_next'):
     logger.name = 'allfeatures_extract_labels'
     if (type(columns)==str):
         indices = [i for i, elem in enumerate(data.columns) if columns in elem] # (alphabetically ordered)
     else:
        indices =  columns 
        
     if indices:
         labels = data[data.columns[indices]]
         data.drop(data.columns[indices], axis=1, inplace=True)    
         logger.info('...Labels extracted from Dataset...')
         return labels
     else: return None


def oneHotEncoder_np(column, typ=DT_FLOAT):
    ''' Encode categorical integer features using a one-hot aka one-of-K scheme from numpy library.
        
    Args: 
        column (Series): Input Categorical integer column.        
    Returns: 
        Numpy Array. Boolean sparse matrix of categorical features.        
    Raises:         
    '''
    logger.name = 'oneHotEncoder_np'
    label_num = len(column.value_counts())
    one_hot_labels = (
        np.arange(label_num) == column[:, None]).astype(DT_FLOAT)    
    logger.info('...labels changed to one-hot-encoding (numpy)....')
    return one_hot_labels


def oneHotEncoder_sklearn(column):
    ''' Encode categorical integer features using a one-hot aka one-of-K scheme from sklearn library.
        
    Args: 
        column (Series): Input Categorical integer column.        
    Returns: 
        Numpy Array. Float sparse matrix of categorical features.        
    Raises:         
    '''
    logger.name = 'oneHotEncoder_sklearn'
    enc = OneHotEncoder()
    arr = column.values
    arr = arr.reshape(-1,1)
    enc.fit(arr)
    arr = enc.transform(arr).toarray()
    logger.info('...labels changed to one-hot-encoding (sklearn)....')
    return arr


def reformat_data_labels(data, labels, typ=DT_FLOAT):
    '''Reformat the pd.DataFrames and pd.Series to np.arrays of a specific type. 
    
    Args: 
        data (DataFrame): Input Dataset. Before this, All categorical features must be changed to one-hot-encoding.
        labels (Series): Class Column of Numeric labels.
    Returns: 
        np.Array. Float Dataset of features.
        np.Array. Float Column of labels.
    Raises: 
        ValueError: 'data and labels have to be aligned!'
    '''    
    logger.name = 'reformat'
    if not (data.index == labels.index).all():
        raise ValueError('data and labels have to be aligned!')
    data_mat = data.values.astype(typ, casting='same_kind')
    labels_mat = labels.values.astype(typ, casting='same_kind')
    logger.info('...Reformatted Dataset...')
    return data_mat, labels_mat


def reformat(data, typ=DT_FLOAT):
    '''Reformat the pd.DataFrames and pd.Series to np.arrays of a specific type. 
    
    Args: 
        data (DataFrame or Series): Input Dataset. Before this, All categorical features must be changed to one-hot-encoding.        
    Returns: 
        np.Array. Float Dataset of features.
        np.Array. Float Column of labels.
    Raises:         
    '''    
    logger.name = 'reformat'    
    data_mat = data.values.astype(typ) # , casting='same_kind'
    logger.info('...Reformatted data...')
    return data_mat


def normalize(data):
    '''Transform features to follow a normal distribution using quantiles information. The transformation is applied on each feature independently.
       Note that this transform is non-linear. It may distort linear correlations between variables measured at the same scale but renders variables measured at different scales more directly comparable.
    
    Args: 
        data (DataFrame): Input Dataset. Before this, All features must be reformatted to numeric values.
    Returns: 
        DataFrame. Normalized Dataset.
    Raises:        
    '''    
    logger.name = 'normalize'
    normalizer = QuantileTransformer(output_distribution='normal')  # 'uniform'    
    logger.info('...Normalized Dataset...')
    return normalizer.fit_transform(data)


def oneHotDummies_column(column, categories):
    '''Convert categorical variable into dummy/indicator variables.
    
    Args: 
        column (Series): Input String Categorical Column.
    Returns: 
        DataFrame. Integer Sparse binary matrix of categorical features.
    Raises:        
    '''    
    logger.name = 'oneHotDummies_column: ' +  column.name
    cat_column = pd.Categorical(column.astype('str'), categories=categories)
    cat_column = pd.get_dummies(cat_column)   # in the same order as categories! (alphabetically ordered) 
    cat_column = cat_column.add_prefix(column.name + '_')
    if (cat_column.isnull().any().any()):
        null_cols = cat_column.columns[cat_column.isnull().any()]
        print(cat_column[null_cols].isnull().sum())
        print(cat_column[cat_column.isnull().any(axis=1)][null_cols].head(50))
    return cat_column
    

def encode_binary_to_labeled_column(sparse_data):
    '''convert a Dataframe of binary columns into a single series column labeled by the binary column names.
    if a record doesn't belong to any class, this method assign automatically this record to the first class.

    Args: 
        sparse_data (DataFrame): Sparse matrix of categorical columns.
    Returns: 
        Series. Single string column of categorical features.
    Raises:        
    '''
    return sparse_data.idxmax(axis=1)


def encode_sparsematrix(data, x):
    '''load and encode from binary sparse matrix which match with substring criteria in the column names to be transformed to a categorical column.

    Args: 
        sparse_data (DataFrame): Sparse matrix of categorical columns.
        x (String). substring criteria to filter the column names.
    Returns: 
        Series. Single string column of categorical features.
    Raises:        
    '''
    sparse_matrix = np.where(pd.DataFrame(data.columns.values)[0].str.contains(x))   
    subframe= data.iloc[:, sparse_matrix[0]]
    return encode_binary_to_labeled_column(subframe)


def get_datasets(data, train_num, valid_num, test_num, weight_flag=False, stratified_flag=False, refNorm=True):        
    '''Sample and transform the data and split it into training, test and validation sets. This function uses:
        -drop_paidoff_reo(...)
        -sample_data(...) from get_raw_data
        -drop_descrip_cols(...)
        -drop_low_variation(...)
        -extract_numeric_labels(...)
        -oneHotEncoder_np(...)
        -reformat(...)
        -normalize(...)        
    Args: 
        data (DataFrame): Input Dataset.
        train_num (Integer): Number of training samples.
        valid_num (Integer): Number of Validation samples.
        test_num (Integer): Number of Testing samples.
        weight_flag (boolean): Default False. True if it will execute a pondered sampling.
        refNorm (boolean). Default True. True if it will execute reformatting and normalization over the selected dataset.
    Returns: 
        Array of tuples (Numpy Array, Numpy Array). Each tuple for training (data, labels), validation (data, labels) and testing (data, labels).                 
        Feature columns (list). List of string names of the columns for training, validation and test sets. 
    Raises:
        ValueError: 'data and labels have to be aligned!'
    '''    
    # Dropping paid-off and REO loans.
    drop_paidoff_reo(data)    
    
    print('Sampling the dataset and shuffling the results....')
    np.random.seed(RANDOM_SEED)
    if (stratified_flag == False):
        data_df = grd.sample_data(data, train_num + valid_num + test_num, weight_flag)
    else:
        data_df = grd.stratified_sample_data(data, float(train_num + valid_num + test_num)/float(data.shape[0]))
        
    logger.info('Size of the database after sampling: ' + str(data_df.shape))
    
    drop_descrip_cols(data_df)
    print('Droppping low-variation variables.....')
    drop_low_variation(data_df, None)        
    print('Getting the numeric labels...')
    labels_df = extract_numeric_labels(data_df)    
    if not (data_df.index == labels_df.index).all():
        raise ValueError('data and labels have to be aligned!')    
    
    labels = oneHotEncoder_np(labels_df)    
    if (refNorm==True):
        print('Reformating and normalizing the data.....')
        data = reformat(data_df)
        data = normalize(data)
        logger.name = 'get_datasets'
        train = (data[:train_num, ], labels[:train_num, ])
        logger.info('Training labels shape: ' + str(train[1].shape))
        valid = (data[train_num:train_num + valid_num, ],
             labels[train_num:train_num + valid_num, ])
        logger.info('Validation labels shape: ' + str(valid[1].shape))
        test = (data[train_num + valid_num:, ], labels[train_num + valid_num:, ])
        logger.info('Test labels shape: ' + str(test[1].shape))
        return train, valid, test, data_df.columns.values.tolist()
    else:
        logger.name = 'get_datasets'
        train = (np.array(data_df.iloc[:train_num, ]), labels[:train_num, ])        
        logger.info('Training labels shape: ' + str(train[1].shape))
        valid = (np.array(data_df.iloc[train_num:train_num + valid_num, ]),
             labels[train_num:train_num + valid_num, ])
        logger.info('Validation labels shape: ' + str(valid[1].shape))
        test = (np.array(data_df.iloc[train_num + valid_num:, ]), labels[train_num + valid_num:, ])
        logger.info('Test labels shape: ' + str(test[1].shape))
        return train, valid, test, data_df.columns.values.tolist()


def drop_invalid_delinquency_status(data, gflag, log_file):   
    
    logger.name = 'drop_invalid_delinquency_status'
    delinq_ids =  data[data['MBA_DELINQUENCY_STATUS'].isin(['0', 'R', 'S', 'T', 'X', 'Z'])]['LOAN_ID']
    groups = data[data['LOAN_ID'].isin(delinq_ids)][['LOAN_ID', 'PERIOD', 'MBA_DELINQUENCY_STATUS', 'DELINQUENCY_STATUS_NEXT']].groupby('LOAN_ID') 
    groups_list = list(groups)
    
    iuw= pd.Index([])
    
    if gflag != '': 
        try:
            iuw= iuw.union(groups.get_group(gflag).index[0:])
        except  Exception  as e:
            print(str(e))
                
    if data.iloc[-1]['LOAN_ID'] in groups.groups.keys():
        gflag = data.iloc[-1]['LOAN_ID']
    else:
        gflag = ''
                
    for k, group in groups_list: 
        li= group.index[(group['MBA_DELINQUENCY_STATUS'] =='S') | (group['MBA_DELINQUENCY_STATUS'] =='T') 
                         | (group['MBA_DELINQUENCY_STATUS'] =='X') | (group['MBA_DELINQUENCY_STATUS'] =='Z')].tolist()
        if li: iuw= iuw.union(group.index[group.index.get_loc(li[0]):])
        # In case of REO or Paid-Off, we need to exclude since the next record:
        df_delinq_01 = group[(group['MBA_DELINQUENCY_STATUS'] =='0') | (group['MBA_DELINQUENCY_STATUS'] =='R')]
        if df_delinq_01.shape[0]>0: 
            track_i = df_delinq_01.index[0]
            iuw= iuw.union(group.index[group.index.get_loc(track_i)+1:])
        
    if iuw!=[]:
        #log_df = data.loc[iuw]
        log_file.write('drop_invalid_delinquency_status - Total rows: %d\r\n' % len(iuw)) # (log_df.shape[0])
        #log_file.write(data.iloc[iuw])
        # np.savetxt(log_file, log_df.values, header=str(log_df.columns.values), delimiter=',')            
        # log_df.to_csv(log_file, index=False, mode='a')
        data.drop(iuw, inplace=True) 
        logger.info('invalid_delinquency_status dropped')             
    
    return gflag


def custom_robust_normalizer(ncols, dist_file, normalizer_type='robust_scaler_sk', center_value='median'):            
    norm_cols = []
    scales = []
    centers = []
    for i, x in enumerate (ncols):                        
        x_frame = dist_file.iloc[:, np.where(pd.DataFrame(dist_file.columns.values)[0].str.contains(x+'_Q'))[0]]    
        if not x_frame.empty:       
            iqr = float(pd.to_numeric(x_frame[x+'_Q3'], errors='coerce').subtract(pd.to_numeric(x_frame[x+'_Q1'], errors='coerce')))
            if iqr!=0: 
                norm_cols.append(x)                
                scales.append(iqr)                    
                if center_value == 'median':
                    centers.append( float(x_frame[x+'_MEDIAN']) )   
                else:
                    centers.append( float(x_frame[x+'_Q1']) )                       
#        else:
#            scales.append(float(0.))
#            centers.append(float(0.))
                
    if (normalizer_type == 'robust_scaler_sk'):    
        normalizer = RobustScaler()
        normalizer.scale_ = scales
        normalizer.center_ = centers        
    elif (normalizer_type == 'percentile_scaler'):    
        normalizer = Normalizer.Normalizer(scales, centers)     
    else: normalizer=None                  
    
    return norm_cols, normalizer

def custom_minmax_normalizer(ncols, scales, dist_file):    
    norm_cols = []
    minmax_scales = []
    centers = []
    # to_delete =[]
    for i, x in enumerate (ncols):  
#        if scales[i] == 0:
        x_frame = dist_file.iloc[:, np.where(pd.DataFrame(dist_file.columns.values)[0].str.contains(x+'_M'))[0]]    
        if not(x_frame.empty) and (x_frame.shape[1]>1):            
            minmax_scales.append(float(x_frame[x+'_MAX'].subtract(x_frame[x+'_MIN'])))                            
            centers.append( float(x_frame[x+'_MIN']))
            norm_cols.append(x)
            # to_delete.append(i)
        
    normalizer = Normalizer.Normalizer(minmax_scales, centers)         
    
    return norm_cols, normalizer #, to_delete

def imputing_nan_values(nan_dict, distribution):        
    new_dict = {}
    for k,v in nan_dict.items():
        if v=='median':
            new_dict[k] = float(distribution[k+'_MEDIAN'])    
        else:
            new_dict[k] = v
            
    return new_dict

def splitDataFrameIntoSmaller(df, chunkSize = 1200): 
    listOfDf = list()     
    numberChunks = math.ceil(len(df) / chunkSize)        
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def tag_chunk(tag, label, chunk, chunk_periods, tag_period, log_file, with_index, tag_index, hdf=None, tfrec=None):
    inter_periods = list(chunk_periods.intersection(set(range(tag_period[0], tag_period[1]+1))))
    log_file.write('Periods corresponding to ' + tag +' period: %s\r\n' % str(inter_periods))
    p_chunk = chunk.loc[(slice(None), slice(None), slice(None), inter_periods), :]
    log_file.write('Records for ' + tag +  ' Set - Number of rows: %d\r\n' % (p_chunk.shape[0]))
    print('Records for ' + tag + ' Set - Number of rows:', p_chunk.shape[0])
    if (p_chunk.shape[0] > 0):
        if (with_index==True):
            p_chunk.index = pd.MultiIndex.from_tuples([(i, x[1], x[2],x[3]) for x,i in zip(p_chunk.index, range(tag_index, tag_index + p_chunk.shape[0]))])        
            labels = allfeatures_extract_labels(p_chunk, columns=label)
            print('chunk and labels shape: ', (p_chunk.shape[0] == labels.shape[0]))
            p_chunk = p_chunk.astype(DT_FLOAT)
            labels = labels.astype(np.int8)
            if (p_chunk.shape[0] != labels.shape[0]) : 
                print('Error in shapes:', p_chunk.shape, labels.shape)
            else :
                if (hdf!=None):
                    hdf.put(tag + '/features', p_chunk, append=True)
                    hdf.put(tag + '/labels', labels, append=True)                         
                elif (tfrec!=None):
                    for row, lab in zip(p_chunk.values, labels.values):
                        feature = {tag + '/labels': _int64_feature(lab),
                                   tag + '/features': _float_feature(row)}
                        # Create an example protocol buffer
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        tfrec.write(example.SerializeToString())                            
                    tfrec.flush()
                tag_index += p_chunk.shape[0]
        else:
            p_chunk.reset_index(drop=True, inplace=True)
            labels = allfeatures_extract_labels(p_chunk, columns=label)
            if (hdf!=None): #only for h5 files:
                pc_subframes = splitDataFrameIntoSmaller(p_chunk, chunkSize = 1000)
                for sf in pc_subframes:
                    hdf.put(tag + '/features', sf.astype(DT_FLOAT), append=True)
                lb_subframes = splitDataFrameIntoSmaller(labels, chunkSize = 1000)
                for sf in lb_subframes:
                    hdf.put(tag + '/labels', sf.astype('int8'), append=True)
                    
    return tag_index
        

def prepro_chunk(file_name, file_path, chunksize, label, log_file, nan_cols, categorical_cols, descriptive_cols, time_cols, robust_cols, minmax_cols,
                 robust_normalizer, minmax_normalizer, dist_file, with_index, refNorm, train_period, valid_period, test_period, hdf=None, tfrec=None):
    gflag = ''    
    i = 1                  
    train_index = 0
    valid_index = 0
    test_index = 0
    for chunk in pd.read_csv(file_path, chunksize = chunksize, sep=',', low_memory=False):    
        print('chunk: ', i, ' chunk size: ', chunk.shape[0])
        log_file.write('chunk: %d, chunk size: %d \n' % (i, chunk.shape[0]))
        chunk.columns = chunk.columns.str.upper()                            
        
        log_df = chunk[chunk[label].isnull()]
        log_file.write('Dropping Rows with Null Labels - Number of rows: %d\r\n' % (log_df.shape[0]))
        # log_df.to_csv(log_file, index=False, mode='a') #summary
        chunk.drop(chunk.index[chunk[label].isnull()], axis=0, inplace=True)
        
        log_df = chunk[chunk['INVALID_TRANSITIONS']==1]
        log_file.write('Dropping Rows with Invalid Transitions - Number of rows: %d\r\n' % (log_df.shape[0]))                        
        # log_df.to_csv(log_file, index=False, mode='a') #summary
        chunk.drop(chunk.index[chunk['INVALID_TRANSITIONS']==1], axis=0, inplace=True)        
        
        gflag = drop_invalid_delinquency_status(chunk, gflag, log_file)               
                    
        null_columns=chunk.columns[chunk.isnull().any()]
        log_df = chunk[chunk.isnull().any(axis=1)][null_columns]
        log_file.write('Filling NULL values - (rows, cols) : %d, %d\r\n' % (log_df.shape[0], log_df.shape[1]))            
        # log_df.to_csv(log_file, index=False, mode='a')  #summary          
        log_df = chunk[null_columns].isnull().sum().to_frame().reset_index()
        log_df.to_csv(log_file, index=False, mode='a')                                    
        nan_cols = imputing_nan_values(nan_cols, dist_file)            
        chunk.fillna(value=nan_cols, inplace=True)   
        
        chunk.drop_duplicates(inplace=True) # Follow this instruction!!                        
        logger.info('dropping invalid transitions and delinquency status, fill nan values, drop duplicates')                  
        log_file.write('Drop duplicates - new size : %d\r\n' % (chunk.shape[0]))
                               
        chunk.reset_index(drop=True, inplace=True)  #don't remove this line! otherwise NaN values appears.
        chunk['ORIGINATION_YEAR'][chunk['ORIGINATION_YEAR']<1995] = "B1995"
        for k,v in categorical_cols.items():
            # if (chunk[k].dtype=='O'):                
            chunk[k] = chunk[k].astype('str')
            chunk[k] = chunk[k].str.strip()
            chunk[k].replace(['\.0$'], [''], regex=True,  inplace=True)
            new_cols = oneHotDummies_column(chunk[k], v)
            if (chunk[k].value_counts().sum()!=new_cols.sum().sum()):
                print('Error at categorization, different sizes', k)
                print(chunk[k].value_counts(), new_cols.sum())                
                log_file.write('Error at categorization, different sizes %s\r\n' % str(k))
                chunk[new_cols.columns] = new_cols
            else:
                chunk[new_cols.columns] = new_cols
                log_file.write('New columns added: %s\r\n' % str(new_cols.columns.values))
            
                    
        allfeatures_drop_cols(chunk, descriptive_cols)                    
        #np.savetxt(log_file, descriptive_cols, header='descriptive_cols dropped:', newline=" ")
        log_file.write('descriptive_cols dropped: %s\r\n' % str(descriptive_cols))
        allfeatures_drop_cols(chunk, time_cols)
        #np.savetxt(log_file, time_cols, header='time_cols dropped:', newline=" ")
        log_file.write('time_cols dropped: %s\r\n' % str(time_cols))
        cat_list = list(categorical_cols.keys())
        cat_list.remove('DELINQUENCY_STATUS_NEXT')
        #np.savetxt(log_file, cat_list, header='categorical_cols dropped:', newline=" ")
        log_file.write('categorical_cols dropped: %s\r\n' % str(cat_list))
        allfeatures_drop_cols(chunk, cat_list)

        chunk.reset_index(drop=True, inplace=True)  
        chunk.set_index(['LOAN_ID', 'DELINQUENCY_STATUS_NEXT', 'PERIOD'], append=True, inplace=True) #4 indexes
        # np.savetxt(log_file, str(chunk.index.names), header='Indexes created:', newline=" ")
        log_file.write('Indexes created: %s\r\n' % str(chunk.index.names))
         
        
        
        if chunk.isnull().any().any(): raise ValueError('There are null values...File: ' + file_name)   
                
        
        if (refNorm==True):            
            chunk[robust_cols] = robust_normalizer.transform(chunk[robust_cols])
            chunk[minmax_cols] = minmax_normalizer.transform(chunk[minmax_cols])            
            #np.savetxt(log_file, robust_cols, header='robust_cols normalized:', newline=" ")
            log_file.write('robust_cols normalized: %s\r\n' % str(robust_cols))
            #np.savetxt(log_file, minmax_cols, header='minmax_cols normalized:', newline=" ")
            log_file.write('minmax_cols normalized: %s\r\n' % str(minmax_cols))
        
        if chunk.isnull().any().any(): raise ValueError('There are null values...File: ' + file_name)   
        
        chunk_periods = set(list(chunk.index.get_level_values('PERIOD')))
        print(tfrec)
        if (tfrec!=None):
            train_index = tag_chunk('train', label, chunk, chunk_periods, train_period, log_file, with_index, train_index, tfrec=tfrec[0])
            valid_index = tag_chunk('valid', label, chunk, chunk_periods, valid_period, log_file, with_index, valid_index, tfrec=tfrec[1])
            test_index = tag_chunk('test', label, chunk, chunk_periods, test_period, log_file, with_index, test_index, tfrec=tfrec[2])
        elif (hdf!=None):
            train_index = tag_chunk('train', label, chunk, chunk_periods, train_period, log_file, with_index, train_index, hdf=hdf)
            valid_index = tag_chunk('valid', label, chunk, chunk_periods, valid_period, log_file, with_index, valid_index, hdf=hdf)
            test_index = tag_chunk('test', label, chunk, chunk_periods, test_period, log_file, with_index, test_index, hdf=hdf)
                
        
        inter_periods = list(chunk_periods.intersection(set(range(test_period[1]+1,355))))    
        log_file.write('Periods greater than test_period: %s\r\n' % str(inter_periods))
        p_chunk = chunk.loc[(slice(None), slice(None), slice(None), inter_periods), :]
        log_file.write('Records greater than test_period - Number of rows: %d\r\n' % (p_chunk.shape[0]))
        
        if (hdf!=None): hdf.flush()
        elif (tfrec!=None): sys.stdout.flush()
        del chunk        
        i +=  1   
    
    return train_index, valid_index, test_index

def allfeatures_prepro_file(RAW_DIR, file_path, raw_dir, file_name, target_path, train_period, valid_period, test_period, log_file, dividing='percentage', chunksize=500000, 
                            refNorm=True, label='DELINQUENCY_STATUS_NEXT', with_index=True, output_hdf=True):
    descriptive_cols = [
#        'LOAN_ID',
        'ASOFMONTH',        
        'PERIOD_NEXT',
        'MOD_PER_FROM',
        'MOD_PER_TO',
        'PROPERTY_ZIP',
        'INVALID_TRANSITIONS'
        ]
        
    numeric_cols = ['MBA_DAYS_DELINQUENT', 'MBA_DAYS_DELINQUENT_NAN',
       'CURRENT_INTEREST_RATE', 'CURRENT_INTEREST_RATE_NAN', 'LOANAGE', 'LOANAGE_NAN',
       'CURRENT_BALANCE', 'CURRENT_BALANCE_NAN', 'SCHEDULED_PRINCIPAL',
       'SCHEDULED_PRINCIPAL_NAN', 'SCHEDULED_MONTHLY_PANDI',
       'SCHEDULED_MONTHLY_PANDI_NAN', 
       'LLMA2_CURRENT_INTEREST_SPREAD', 'LLMA2_CURRENT_INTEREST_SPREAD_NAN',  
       'LLMA2_C_IN_LAST_12_MONTHS',
       'LLMA2_30_IN_LAST_12_MONTHS', 'LLMA2_60_IN_LAST_12_MONTHS',
       'LLMA2_90_IN_LAST_12_MONTHS', 'LLMA2_FC_IN_LAST_12_MONTHS',
       'LLMA2_REO_IN_LAST_12_MONTHS', 'LLMA2_0_IN_LAST_12_MONTHS',
       'LLMA2_HIST_LAST_12_MONTHS_MIS', 
       'NUM_MODIF', 'NUM_MODIF_NAN', 'P_RATE_TO_MOD', 'P_RATE_TO_MOD_NAN', 'MOD_RATE',
       'MOD_RATE_NAN', 'DIF_RATE', 'DIF_RATE_NAN', 'P_MONTHLY_PAY',
       'P_MONTHLY_PAY_NAN', 'MOD_MONTHLY_PAY', 'MOD_MONTHLY_PAY_NAN',
       'DIF_MONTHLY_PAY', 'DIF_MONTHLY_PAY_NAN', 'CAPITALIZATION_AMT',
       'CAPITALIZATION_AMT_NAN', 'MORTGAGE_RATE', 'MORTGAGE_RATE_NAN',
       'FICO_SCORE_ORIGINATION', 'INITIAL_INTEREST_RATE', 'ORIGINAL_LTV',
       'ORIGINAL_BALANCE', 'BACKEND_RATIO', 'BACKEND_RATIO_NAN',
       'ORIGINAL_TERM', 'ORIGINAL_TERM_NAN', 'SALE_PRICE', 'SALE_PRICE_NAN', 	   
       'PREPAY_PENALTY_TERM', 'PREPAY_PENALTY_TERM_NAN', 
	    'NUMBER_OF_UNITS', 'NUMBER_OF_UNITS_NAN', 'MARGIN',
       'MARGIN_NAN', 'PERIODIC_RATE_CAP', 'PERIODIC_RATE_CAP_NAN',
       'PERIODIC_RATE_FLOOR', 'PERIODIC_RATE_FLOOR_NAN', 'LIFETIME_RATE_CAP',
       'LIFETIME_RATE_CAP_NAN', 'LIFETIME_RATE_FLOOR',
       'LIFETIME_RATE_FLOOR_NAN', 'RATE_RESET_FREQUENCY',
       'RATE_RESET_FREQUENCY_NAN', 'PAY_RESET_FREQUENCY',
       'PAY_RESET_FREQUENCY_NAN', 'FIRST_RATE_RESET_PERIOD',
       'FIRST_RATE_RESET_PERIOD_NAN', 	   
	    'LLMA2_PRIME',
       'LLMA2_SUBPRIME', 'LLMA2_APPVAL_LT_SALEPRICE', 'LLMA2_ORIG_RATE_SPREAD',
       'LLMA2_ORIG_RATE_SPREAD_NAN', 'AGI', 'AGI_NAN', 'UR', 'UR_NAN', 'LLMA2_ORIG_RATE_ORIG_MR_SPREAD', 
       'LLMA2_ORIG_RATE_ORIG_MR_SPREAD_NAN', 'COUNT_INT_RATE_LESS', 'NUM_PRIME_ZIP', 'NUM_PRIME_ZIP_NAN'
       ]
        
#    nan_cols = {'MBA_DAYS_DELINQUENT': 0, 'CURRENT_INTEREST_RATE': 0, 'LOANAGE': 0,
#                'CURRENT_BALANCE' : 0, 'SCHEDULED_PRINCIPAL': 0, 'SCHEDULED_MONTHLY_PANDI': 0,       
#                'LLMA2_CURRENT_INTEREST_SPREAD': 0, 'NUM_MODIF': 0, 'P_RATE_TO_MOD': 0, 'MOD_RATE': 0,
#                'DIF_RATE': 0, 'P_MONTHLY_PAY': 0, 'MOD_MONTHLY_PAY': 0, 'DIF_MONTHLY_PAY': 0, 'CAPITALIZATION_AMT': 0,
#                'MORTGAGE_RATE': 0, 'FICO_SCORE_ORIGINATION': 0, 'INITIAL_INTEREST_RATE': 0, 'ORIGINAL_LTV': 0,
#                'ORIGINAL_BALANCE': 0, 'BACKEND_RATIO': 0, 'ORIGINAL_TERM': 0, 'SALE_PRICE': 0, 'PREPAY_PENALTY_TERM': 0,
#                'NUMBER_OF_UNITS': 0, 'MARGIN': 0, 'PERIODIC_RATE_CAP': 0, 'PERIODIC_RATE_FLOOR': 0, 'LIFETIME_RATE_CAP': 0,
#                'LIFETIME_RATE_FLOOR': 0, 'RATE_RESET_FREQUENCY': 0, 'PAY_RESET_FREQUENCY': 0,
#                'FIRST_RATE_RESET_PERIOD': 0, 'LLMA2_ORIG_RATE_SPREAD': 0, 'AGI': 0, 'UR': 0,
#                'LLMA2_C_IN_LAST_12_MONTHS': 0, 'LLMA2_30_IN_LAST_12_MONTHS': 0, 'LLMA2_60_IN_LAST_12_MONTHS': 0,
#                'LLMA2_90_IN_LAST_12_MONTHS': 0, 'LLMA2_FC_IN_LAST_12_MONTHS': 0,
#                'LLMA2_REO_IN_LAST_12_MONTHS': 0, 'LLMA2_0_IN_LAST_12_MONTHS': 0}
    
    nan_cols = {'MBA_DAYS_DELINQUENT': 'median', 'CURRENT_INTEREST_RATE': 'median', 'LOANAGE': 'median',
                'CURRENT_BALANCE' : 'median', 'SCHEDULED_PRINCIPAL': 'median', 'SCHEDULED_MONTHLY_PANDI': 'median',       
                'LLMA2_CURRENT_INTEREST_SPREAD': 'median', 'NUM_MODIF': 0, 'P_RATE_TO_MOD': 0, 'MOD_RATE': 0,
                'DIF_RATE': 0, 'P_MONTHLY_PAY': 0, 'MOD_MONTHLY_PAY': 0, 'DIF_MONTHLY_PAY': 0, 'CAPITALIZATION_AMT': 0,
                'MORTGAGE_RATE': 'median', 'FICO_SCORE_ORIGINATION': 'median', 'INITIAL_INTEREST_RATE': 'median', 'ORIGINAL_LTV': 'median',
                'ORIGINAL_BALANCE': 'median', 'BACKEND_RATIO': 'median', 'ORIGINAL_TERM': 'median', 'SALE_PRICE': 'median', 'PREPAY_PENALTY_TERM': 'median',
                'NUMBER_OF_UNITS': 'median', 'MARGIN': 'median', 'PERIODIC_RATE_CAP': 'median', 'PERIODIC_RATE_FLOOR': 'median', 'LIFETIME_RATE_CAP': 'median',
                'LIFETIME_RATE_FLOOR': 'median', 'RATE_RESET_FREQUENCY': 'median', 'PAY_RESET_FREQUENCY': 'median',
                'FIRST_RATE_RESET_PERIOD': 'median', 'LLMA2_ORIG_RATE_SPREAD': 'median', 'AGI': 'median', 'UR': 'median',
                'LLMA2_C_IN_LAST_12_MONTHS': 'median', 'LLMA2_30_IN_LAST_12_MONTHS': 'median', 'LLMA2_60_IN_LAST_12_MONTHS': 'median',
                'LLMA2_90_IN_LAST_12_MONTHS': 'median', 'LLMA2_FC_IN_LAST_12_MONTHS': 'median',
                'LLMA2_REO_IN_LAST_12_MONTHS': 'median', 'LLMA2_0_IN_LAST_12_MONTHS': 'median', 
                'LLMA2_ORIG_RATE_ORIG_MR_SPREAD':0, 'COUNT_INT_RATE_LESS' :'median', 'NUM_PRIME_ZIP':'median'
                }
        
    categorical_cols = {'MBA_DELINQUENCY_STATUS':  ['0','3','6','9','C','F','R'], 'DELINQUENCY_STATUS_NEXT': ['0','3','6','9','C','F','R'],  #,'S','T','X'
                           'BUYDOWN_FLAG': ['N','U','Y'], 'NEGATIVE_AMORTIZATION_FLAG': ['N','U','Y'], 'PREPAY_PENALTY_FLAG': ['N','U','Y'],
                           'OCCUPANCY_TYPE': ['1','2','3','U'], 'PRODUCT_TYPE': ['10','20','30','40','50','51','52','53','54','5A','5Z',
                                            '60','61','62','63','6Z','70','80','81','82','83','84','8Z','U'], 
                           'PROPERTY_TYPE': ['1','2','3','4','5','6','7','8','9','M','U','Z'], 'LOAN_PURPOSE_CATEGORY': ['P','R','U'], 
                           'DOCUMENTATION_TYPE': ['1','2','3','U'], 'CHANNEL': ['1','2','3','4','5','6','7','8','9','A','B','C','D','U'], 
                           'LOAN_TYPE': ['1','2','3','4','5','6','U'], 'IO_FLAG': ['N','U','Y'], 
                           'CONVERTIBLE_FLAG': ['N','U','Y'], 'POOL_INSURANCE_FLAG': ['N','U','Y'], 'STATE': ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO',
                                               'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 
                                               'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 
                                               'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 
                                               'WA', 'WI', 'WV', 'WY'], 
                           'CURRENT_INVESTOR_CODE': ['240', '250', '253', 'U'], 'ORIGINATION_YEAR': ['B1995','1995','1996','1997','1998','1999','2000','2001','2002','2003',
                                                    '2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']}
      
    time_cols = ['YEAR', 'MONTH'] #, 'PERIOD'] #no nan values        
    pd.set_option('io.hdf.default_format','table')
    
    dist_file = pd.read_csv(os.path.join(RAW_DIR, "percentile features3.csv"), sep=';', low_memory=False)
    dist_file.columns = dist_file.columns.str.upper()
    
    ncols = [x for x in numeric_cols if x.find('NAN')<0]
    robust_cols, robust_normalizer = custom_robust_normalizer(ncols, dist_file, center_value='quantile', normalizer_type='percentile_scaler')    
    minmax_cols, minmax_normalizer = custom_minmax_normalizer(ncols, robust_normalizer.scale_, dist_file)            
   
    inters = set(robust_cols).intersection(minmax_cols)
    to_delete = [i for x,i in zip(minmax_cols,range(len(minmax_cols))) if x in inters]
    minmax_normalizer.scale_ = np.delete(minmax_normalizer.scale_,to_delete, 0)
    minmax_normalizer.center_ = np.delete(minmax_normalizer.center_,to_delete, 0)
    minmax_cols = np.delete(minmax_cols,to_delete, 0)            
    
    if (output_hdf == True):
        with  pd.HDFStore(target_path +'-pp.h5', complib='lzo', complevel=9) as hdf: #complib='lzo', complevel=9
            
            print('generating: ', target_path +'-pp.h5')
            train_index, valid_index, test_index = prepro_chunk(file_name, file_path, chunksize, label, log_file, nan_cols, categorical_cols, descriptive_cols, 
                                                                time_cols, robust_cols, minmax_cols, robust_normalizer, minmax_normalizer, dist_file, with_index, 
                                                                refNorm, train_period, valid_period, test_period, hdf=hdf, tfrec=None)            
            
            print(train_index, valid_index, test_index)
            
            if hdf.get_storer('train/features').nrows != hdf.get_storer('train/labels').nrows:
                    raise ValueError('Train-DataSet: Sizes should match!')  
            if hdf.get_storer('valid/features').nrows != hdf.get_storer('valid/labels').nrows:
                    raise ValueError('Valid-DataSet: Sizes should match!')  
            if hdf.get_storer('test/features').nrows != hdf.get_storer('test/labels').nrows:
                    raise ValueError('Test-DataSet: Sizes should match!')  
            
            print('train/features size: ', hdf.get_storer('train/features').nrows)
            print('valid/features size: ', hdf.get_storer('valid/features').nrows)
            print('test/features size: ', hdf.get_storer('test/features').nrows)
            
            log_file.write('***SUMMARY***\n')
            log_file.write('train/features size: %d\r\n' %(hdf.get_storer('train/features').nrows))
            log_file.write('valid/features size: %d\r\n' %(hdf.get_storer('valid/features').nrows))
            log_file.write('test/features size: %d\r\n' %(hdf.get_storer('test/features').nrows))
    
            logger.info('training, validation and testing set into .h5 file')        
    else:        
        train_writer = tf.python_io.TFRecordWriter(target_path +'-train-pp.tfrecords')
        valid_writer = tf.python_io.TFRecordWriter(target_path +'-valid-pp.tfrecords')
        test_writer = tf.python_io.TFRecordWriter(target_path +'-test-pp.tfrecords')
        train_index, valid_index, test_index = prepro_chunk(file_name, file_path, chunksize, label, log_file, nan_cols, categorical_cols, descriptive_cols, time_cols, 
                                                            robust_cols, minmax_cols, robust_normalizer, minmax_normalizer, dist_file, with_index, refNorm, train_period, 

                                                            valid_period, test_period, hdf=None, tfrec=[train_writer, valid_writer, test_writer]) 
        print(train_index, valid_index, test_index)
        train_writer.close()
        valid_writer.close()
        test_writer.close()
            


def get_other_set_slice(prep_dir, init_period, end_period, set_dir, file_name, chunk_size=8000000):
    
    pd.set_option('io.hdf.default_format','table')
    try:
        chunk_ind = 0
        target_path = os.path.join(PRO_DIR, set_dir,file_name+'_{:d}.h5'.format(chunk_ind))        
        hdf_target =  pd.HDFStore(target_path) 
        print('Target Path: ', target_path)
        total_rows = 0       
        for file_path in glob.glob(os.path.join(PRO_DIR, prep_dir, "*.h5")): 
            file_name = os.path.basename(file_path)        
            with pd.HDFStore(file_path) as hdf_input:
                # hdf_input.get['features'].                
                # temp_features = pd.read_hdf(self.h5_path, self.dtype + '/features', start=self._global_index, stop=self._global_index + batch_size)        
                # df = hdf_input.select('features', [ Term('index', '>', Timestamp('20010105') ])
                period_range =  set(range(init_period, end_period+1))
                period_features = set(list(hdf_input['features'].index.get_level_values(2)))
                period_inter = period_features.intersection(period_range)
                for i in list(period_inter):
                    df_features = hdf_input['features'].loc[(slice(None), slice(None), i), :]
                    df_labels = hdf_input['labels'].loc[(slice(None), slice(None), i), :]
                    hdf_target.put('features', df_features, append=True) 
                    hdf_target.put('labels', df_labels, append=True) 
                    hdf_target.flush()
                    total_rows += df_features.shape[0]
                    num_columns = len(df_features.columns.values)
                    del df_features
                    del df_labels
                    if (total_rows >= chunk_size or i==period_inter[-1]):
                        if hdf_target.get_storer('features').nrows != hdf_target.get_storer('labels').nrows:
                            raise ValueError('DataSet: Sizes should match!')
                        hdf_target.get_storer('features').attrs.num_columns = num_columns
                        hdf_target.close()
                        total_rows = 0
                        chunk_ind += 1
                        if (i!=period_inter[-1]):
                            target_path = os.path.join(PRO_DIR, set_dir,file_name+'_{:d}.h5'.format(chunk_ind))
                            hdf_target =  pd.HDFStore(target_path) 
                            print('Target Path: ', target_path)                        
        if hdf_target.is_open: hdf_target.close()
    except Exception as e:
        hdf_target.close()
        print(e)        
                    

def get_other_set(prep_dir, init_period, end_period, set_dir, chunk_size=8000000):
    
    pd.set_option('io.hdf.default_format','table')
    try:
        chunk_ind = 0        
        for file_path in glob.glob(os.path.join(PRO_DIR, prep_dir, "*.h5")): 
            file_name = os.path.basename(file_path)        
            print(file_name)
            with pd.HDFStore(file_path) as hdf_input:  
                file_index = 0
                for df_features in hdf_input.select('features', "PERIOD>=" + str(init_period) + ' & PERIOD<=' + str(end_period), chunksize = chunk_size):
                    try:
                        target_path = os.path.join(PRO_DIR, set_dir,file_name[:-4]+'_{:d}.h5'.format(chunk_ind))
                        hdf_target =  pd.HDFStore(target_path) 
                        print('Target Path: ', target_path)
    
                        if file_index + chunk_size <= hdf_input.get_storer('features').nrows:
                            df_labels = hdf_input.select('labels', "PERIOD>=" + str(init_period) + ' & PERIOD<=' + str(end_period), start = file_index, stop = file_index + chunk_size)
                            file_index += chunk_size
                        else:
                            df_labels = hdf_input.select('labels', "PERIOD>=" + str(init_period) + ' & PERIOD<=' + str(end_period), start = file_index)
                            file_index = 0
                        hdf_target.put('features', df_features, append=True) 
                        hdf_target.put('labels', df_labels, append=True) 
                        hdf_target.flush()
                        num_columns = len(df_features.columns.values)                    
                        hdf_target.get_storer('features').attrs.num_columns = num_columns
                        if hdf_target.get_storer('features').nrows != hdf_target.get_storer('labels').nrows:
                            raise ValueError('DataSet: Sizes should match!')
                        hdf_target.close()
                        del df_labels
                        del df_features                    
                        chunk_ind += 1
                    except Exception as e:
                        if hdf_target.is_open: hdf_target.close()
    except Exception as e:        
        print(e)        

def slice_fixed_sets(prep_dir, set_dir, tag, chunk_size=400000):
    
    pd.set_option('io.hdf.default_format','fixed') #'table')
    try:
        chunk_ind = 0        
        for file_path in glob.glob(os.path.join(PRO_DIR, prep_dir, "*.h5")): 
            file_name = os.path.basename(file_path)        
            print(file_name)
            with pd.HDFStore(file_path) as hdf_input:  
                file_index = 0
                for df_features in hdf_input.select(tag + '/features', chunksize = chunk_size):
                    try:
                        target_path = os.path.join(PRO_DIR, set_dir,file_name[:-4]+'_{:d}.h5'.format(chunk_ind))
                        hdf_target =  pd.HDFStore(target_path, complib='lzo', complevel=9, chunkshape='auto') 
                        print('Target Path: ', target_path)    
                        df_labels = hdf_input.select(tag + '/labels', start = file_index, stop = file_index + df_features.shape[0])
                        # df_labels = df_labels.reset_index(level='index', drop=True)                                                
                        # df_labels.set_index('index', range(0, chunk_size), append=True, inplace=True)                                                
                        df_features.index = pd.MultiIndex.from_tuples([(i, x[1], x[2],x[3]) for x,i in zip(df_features.index, range(0, df_features.shape[0]))])
                        df_labels.index = pd.MultiIndex.from_tuples([(i, x[1], x[2],x[3]) for x,i in zip(df_labels.index, range(0, df_labels.shape[0]))])
                        file_index += df_features.shape[0]
                        hdf_target.put(tag + '/features', df_features) 
                        hdf_target.put(tag + '/labels', df_labels)                        
                        hdf_target.flush()                        
                        if hdf_target.get_storer(tag+'/features').shape[0] != hdf_target.get_storer(tag + '/labels').shape[0]:
                            raise ValueError('DataSet: Sizes should match!')
                        hdf_target.close()
                        del df_labels
                        del df_features                    
                        chunk_ind += 1
                    except Exception as e:
                        if hdf_target.is_open: hdf_target.close()
    except Exception as e:        
        print(e)                    

def slice_table_sets(prep_dir, set_dir, tag, target_name, input_chunk_size=1200, target_size = 70000, with_index=True, index=0):
    '''The input directory must not be the same as the output directory, because the .h5 output files can be confused with the input files. '''
    pd.set_option('io.hdf.default_format', 'table')
    all_files = glob.glob(os.path.join(PRO_DIR, prep_dir, "*.h5"))        
    chunk_ind = index
    if (with_index==True):      
        target_path = os.path.join(PRO_DIR, set_dir,target_name+'_{:d}.h5'.format(chunk_ind))
    else:
        target_path = os.path.join(PRO_DIR, set_dir,target_name+'_non_index_{:d}.h5'.format(chunk_ind))
        
    hdf_target =  pd.HDFStore(target_path, complib='lzo', complevel=9)         
    print('Target Path: ', target_path)
    try:        
        total_rows = 0                
        target_index = 0
        for i, file_path in enumerate(all_files): 
            file_name = os.path.basename(file_path)        
            print('Input File: ', file_name)
            with pd.HDFStore(file_path) as hdf_input:  
                file_index = 0
                for df_features in hdf_input.select(tag + '/features', chunksize = input_chunk_size):
                    try:                            
                        df_labels = hdf_input.select(tag + '/labels', start = file_index, stop = file_index + df_features.shape[0])                        
                        # df_labels.set_index('index', range(0, chunk_size), append=True, inplace=True)              
                        if (with_index==True):
                            df_features.index = pd.MultiIndex.from_tuples([(i, x[1], x[2],x[3]) for x,i in zip(df_features.index, range(target_index, target_index + df_features.shape[0]))])
                            df_labels.index = pd.MultiIndex.from_tuples([(i, x[1], x[2],x[3]) for x,i in zip(df_labels.index, range(target_index, target_index + df_labels.shape[0]))])
                        else:
                            df_features.reset_index(drop=True, inplace=True)
                            df_labels.reset_index(drop=True, inplace=True)
                            
                        file_index += df_features.shape[0]
                        target_index += df_features.shape[0]
                        hdf_target.put(tag + '/features', df_features, append=True) 
                        hdf_target.put(tag + '/labels', df_labels, append=True)                        
                        hdf_target.flush()      
                        total_rows += df_features.shape[0]
                        print('total_rows: ', total_rows)
                        if (total_rows >= target_size):
                            if hdf_target.get_storer(tag+'/features').nrows != hdf_target.get_storer(tag + '/labels').nrows:
                                raise ValueError('DataSet: Sizes should match!')
                            hdf_target.close()
                            total_rows = 0
                            chunk_ind += 1
                            if ((i+1<len(all_files)) or (i+1==len(all_files) and df_features.shape[0]>=input_chunk_size)):
                                if (with_index==True):      
                                    target_path = os.path.join(PRO_DIR, set_dir,target_name+'_{:d}.h5'.format(chunk_ind))                                    
                                else:
                                    target_path = os.path.join(PRO_DIR, set_dir,target_name+'_non_index_{:d}.h5'.format(chunk_ind))                                                                
                                hdf_target =  pd.HDFStore(target_path, complib='lzo', complevel=9)
                                print('Target Path: ', target_path)  
                                target_index = 0                                                  
                        del df_labels
                        del df_features                                            
                    except Exception as e:
                        if hdf_target.is_open: hdf_target.close()        
        if hdf_target.is_open: hdf_target.close()
    except Exception as e:        
        if hdf_target.is_open: hdf_target.close()
        print(e)
                       
def get_h5_dataset(PRO_DIR, train_dir, valid_dir, test_dir, train_period=[121, 316], valid_period=[317,323], test_period=[324, 351]):
    train_path = os.path.join(PRO_DIR, train_dir)
    valid_path = os.path.join(PRO_DIR, valid_dir)
    test_path = os.path.join(PRO_DIR, test_dir)    
    DATA = data_classes.Dataset(train_path=train_path, valid_path=valid_path, test_path=test_path, 
                                train_period=train_period, valid_period=valid_period, test_period=test_period)
        
    return DATA

    
def allfeatures_preprocessing(RAW_DIR, PRO_DIR, raw_dir, train_num, valid_num, test_num, dividing='percentage', chunksize=500000, refNorm=True, with_index=True, output_hdf=True):            

    for file_path in glob.glob(os.path.join(RAW_DIR, raw_dir,"*.txt")):  
        file_name = os.path.basename(file_path)
        if with_index==True:
            target_path = os.path.join(PRO_DIR, raw_dir,file_name[:-4])        
        else:
            target_path = os.path.join(PRO_DIR, raw_dir,file_name[:-4]+'_non_index')
        log_file=open(target_path+'-log.txt', 'w+', 1)        
        print('Preprocessing File: ' + file_path)
        log_file.write('Preprocessing File:  %s\r\n' % file_path)
        startTime = datetime.now()        
        allfeatures_prepro_file(RAW_DIR, file_path, raw_dir, file_name, target_path, train_num, valid_num, test_num, log_file, dividing=dividing, chunksize=chunksize, 
                                refNorm=refNorm, with_index=with_index, output_hdf=output_hdf)          
        startTime = datetime.now() - startTime
        print('Preprocessing Time: ', startTime)     
        log_file.write('Preprocessing Time:  %s\r\n' % str(startTime))
        log_file.close()


def read_data_sets(num_examples, valid_num, test_num, weight_flag=False, stratified_flag=False, refNorm=True):
    """load the notMNIST dataset and apply get_datasets(...) function.    
    Args: 
        num_examples (Integer): Input Dataset.
        train_num (Integer): Number of Training examples.
        valid_num (Integer): Number of Validation samples.
        test_num (Integer): Number of Testing samples.
        weight_flag (boolean): Default False. True if it executes a pondered sampling.
    Returns: 
        data_classes.Dataset Object.
    Raises:        
    """
    print('Reading the data from disk....')    
    all_data = grd.read_df(45) 
    print('Size of the database:', all_data.shape)
    train, valid, test, feature_columns = get_datasets(all_data, num_examples, valid_num, test_num,
                            weight_flag=weight_flag, stratified_flag=stratified_flag, refNorm=refNorm)    
    return data_classes.Dataset(train, valid, test, feature_columns)

def update_parser(parser):
    """Parse the arguments from the CLI and update the parser."""    
    parser.add_argument(
        '--prepro_step',
        type=str,
        default='preprocessing', #'slicing', 'preprocessing'
        help='To execute a preprocessing method')    
    #this is for allfeatures_preprocessing:
    parser.add_argument(
        '--train_period',
        type=int,
        nargs='*',
        default=[121,279], #[156, 180], [121,143],  # 279],
        help='Training Period')
    parser.add_argument(
        '--valid_period',
        type=int,
        nargs='*',
        default=[280,285], #[181,185], [144,147],
        help='Validation Period')    
    parser.add_argument(
        '--test_period',
        type=int,
        nargs='*',
        default= [286, 304], # [186,191], [148, 155],
        help='Testing Period')    
    parser.add_argument(
        '--prepro_dir',
        type=str,
        default='chuncks_random_c1mill',
        help='Directory with raw data inside data/raw/ and it will be the output directory inside data/processed/')    
    parser.add_argument(
        '--prepro_chunksize',
        type=int,
        default=500000,
        help='Chunk size to put into the h5 file...')    
    parser.add_argument(
        '--prepro_with_index',
        type=bool,
        default=True,
        help='To keep indexes for each record')
    parser.add_argument(
        '--ref_norm',
        type=bool,
        default=True,
        help='To execute the normalization over the raw inputs')
    
    #to execute slice_table_sets:
    parser.add_argument(
        '--slice_input_dir',
        type=str,
        default='chuncks_random_c1mill',
        help='Input data directory')
    parser.add_argument(
        '--slice_output_dir',
        type=str,
        nargs='*',
        default=['chuncks_random_c1mill_train', 'chuncks_random_c1mill_valid', 'chuncks_random_c1mill_test'],
        help='Output data directory. Input and output could be the same per group, it is recommendable different directories...')
    parser.add_argument(
        '--slice_tag',
        type=str,
        nargs='*',
        default=['train', 'valid', 'test'],
        help='features group to be extracted')
    parser.add_argument(
        '--slice_target_name',
        type=str,
        nargs='*',
        default=['c1mill99-01_train', 'c1mill99-01_valid', 'c1mill99-01_test'],
        help='file name root inside output directory')
    parser.add_argument(
        '--slice_chunksize',
        type=int,
        default=1000,
        help='Chunk size to put into the h5 output files...')
    parser.add_argument(
        '--slice_target_size',
        type=int,
        default=36000000,
        help='Output file size')
    parser.add_argument(
        '--slice_with_index',
        type=bool,
        default=False,
        help='To keep indexes for each record')
    parser.add_argument(
        '--slice_index',
        type=int,
        default=0,
        help='index to label each output file')       
    
    return parser.parse_known_args()


def main(project_dir):
    """ 
    This module is in charge of::
        - Retrieving DataFrame from Raw Data .
        - Data Sampling.
        - Encode Categorical features.
        - Reformat and Normalize features.
        - Remove columns from dataset.
        - Split the dataset in training, validation and testing sets.
        - 
    """   
    logger.name ='__main__'     
    logger.info('Retrieving DataFrame from Raw Data, Data Sampling')
    print("Run the main program.")

    FLAGS, UNPARSED = update_parser(argparse.ArgumentParser())    
    print("UNPARSED", UNPARSED)    
        
    if FLAGS.prepro_step == 'preprocessing':
        startTime = datetime.now()
        #chuncks_random_c1mill chunks_all_800th
        #allfeatures_preprocessing('chuncks_random_c1mill', [121, 279], [280,285], [286, 304], dividing='percentage', chunksize=500000, refNorm=True, with_index=True)  
        if not os.path.exists(os.path.join(PRO_DIR, FLAGS.prepro_dir)): #os.path.exists
                os.makedirs(os.path.join(PRO_DIR, FLAGS.prepro_dir))
        allfeatures_preprocessing(RAW_DIR, PRO_DIR, FLAGS.prepro_dir, FLAGS.train_period, FLAGS.valid_period, FLAGS.test_period, dividing='percentage', 
                                  chunksize=FLAGS.prepro_chunksize, refNorm=FLAGS.ref_norm, with_index=FLAGS.prepro_with_index, output_hdf=True)        
        print('Preprocessing - Time: ', datetime.now() - startTime)
    elif FLAGS.prepro_step == 'slicing':        
        for i in range(len(FLAGS.slice_tag)):
            startTime = datetime.now()
            if not os.path.exists(os.path.join(PRO_DIR, FLAGS.slice_output_dir[i])): #os.path.exists
                os.makedirs(os.path.join(PRO_DIR, FLAGS.slice_output_dir[i]))
            slice_table_sets(FLAGS.slice_input_dir, FLAGS.slice_output_dir[i], FLAGS.slice_tag[i], FLAGS.slice_target_name[i], 
                         target_size=FLAGS.slice_target_size, with_index=FLAGS.slice_with_index, index=FLAGS.slice_index, input_chunk_size = FLAGS.slice_chunksize)        
            print('Dividing .h5 files - Time: ', datetime.now() - startTime)
        #slice_table_sets('chuncks_random_c1mill', 'chuncks_random_c1mill', 'train', 'chuncks_random_c1mill_train_cs1200', target_size=36000000, with_index=False, index=2)
        #slice_table_sets('chuncks_random_c1mill', 'chuncks_random_c1mill', 'valid', 'chuncks_random_c1mill_valid_cs1200', target_size=36000000, with_index=False, index=2)
        #slice_table_sets('chuncks_random_c1mill', 'chuncks_random_c1mill', 'test', 'chuncks_random_c1mill_test_cs1200', target_size=36000000, with_index=False, index=2)        
    else: 
        print('Invalid prepro_step...')



if __name__ == '__main__':        
    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    logger.propagate = False
    main(project_dir)
        