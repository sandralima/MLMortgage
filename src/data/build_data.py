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
import datetime

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder

DT_FLOAT = np.float32 
DT_BOOL = np.uint8
RANDOM_SEED = 123
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.propagate = False # it will not log to console.

RAW_DIR = os.path.join(Path(abspath(getsourcefile(lambda:0))).parents[2], 'data', 'raw') 


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


def oneHotDummies_column(column):
    '''Convert categorical variable into dummy/indicator variables.
    
    Args: 
        column (Series): Input String Categorical Column.
    Returns: 
        DataFrame. Integer Sparse binary matrix of categorical features.
    Raises:        
    '''    
    return pd.get_dummies(column)
    

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

def allfeatures_preprocessing(raw_dir, file_name, chunksize=500000, refNorm=True):
    
    for chunk in pd.read_csv(os.path.join(RAW_DIR, raw_dir, file_name+'-{:d}.csv'.format(chunk_ind)), chunksize = chunksize, sep=',', low_memory=False):    
        # Dropping paid-off and REO loans.
        drop_paidoff_reo(data)    
        drop_descrip_cols(data_df)
        print('Droppping low-variation variables.....')
        drop_low_variation(data_df, None)        
        print('Getting the numeric labels...')        
    
        labels = oneHotEncoder_np(labels_df)    
        
        if (refNorm==True):
            print('Reformating and normalizing the data.....')
            data = reformat(data_df)
            data = normalize(data)
            logger.name = 'get_datasets'       
        
        chunk.to_csv(os.path.join(RAW_DIR, raw_dir, dynamic_fname+'-STATIC-IT-{:d}.csv'.format(chunk_ind)), mode='a')    
        
        if df.isnull().any().any(): exception('There are null values...')
        
 

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
                            weight_flag, stratified_flag, refNorm=refNorm)    
    return data_classes.Dataset(train, valid, test, feature_columns)


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
    #all_data = grd.read_df(45)   
    #all_data['LLMA2_APPVAL_LT_SALEPRICE'] = reformat(all_data['LLMA2_APPVAL_LT_SALEPRICE'], typ=DT_BOOL)
    # s_data = grd.stratified_sample_data(all_data, 0.2)        
    # DATA = read_data_sets(220000, 20000, 20000, refNorm=False)
    # print(DATA.feature_columns)
    allfeatures_preprocessing()
        



if __name__ == '__main__':        
    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
        

#def invalid_transition(group):    
#    # np.dstack((arr_a, arr_b)) # doesn't work
#    transition_chain = np.array(list(zip(data['MBA_DELINQUENCY_STATUS_next'],data['MBA_DELINQUENCY_STATUS_next'][1:])))
#    date_diff = (data[['ASOFMONTH']].diff() <= datetime.timedelta(days=31)).values.ravel()
#    transition_chain = transition_chain[date_diff[1:]] 
#    invalid_transitions = np.array([[4, 2], [4, 3], [1, 3], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], 
#                           [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5]])
#    
#    uw = []
#    i = 0
#    for z in range(0,len(transition_chain)):        
#        # it doesnt work because 
#        if (transition_chain[z] in invalid_transitions): # other cycle More long time.
#            uw.append(i+1)

# drop_invalid_delinquency_status(new_data, invalid_transition, status=None)