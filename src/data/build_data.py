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
         indices = [i for i, elem in enumerate(data.columns) if columns in elem]
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
    cat_column = pd.get_dummies(cat_column)    
    cat_column = cat_column.add_prefix(column.name + '_')
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


def drop_invalid_delinquency_status(data, gflag):   
    
    logger.name = 'drop_invalid_delinquency_status'
    delinq_ids =  data[data['MBA_DELINQUENCY_STATUS'].isin(['S', 'T', 'X', 'Z'])]['LOAN_ID']
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
        iuw= iuw.union(group.index[group.index.get_loc(li[0]):])        
        
    if iuw!=[]:                        
        data.drop(iuw, inplace=True) 
        
    logger.info('invalid_delinquency_status dropped')     
    return gflag


def custom_robust_normalizer(ncols, dist_file, normalizer_type='robust_scaler_sk', center_value='median'):            
    norm_cols = []
    scales = []
    centers = []
    for i, x in enumerate (ncols):                        
        x_frame = dist_file.iloc[:, np.where(pd.DataFrame(dist_file.columns.values)[0].str.contains(x+'_'))[0]]    
        if not x_frame.empty:            
            scales.append(float(pd.to_numeric(x_frame[x+'_Q3'], errors='coerce').subtract(pd.to_numeric(x_frame[x+'_Q1'], errors='coerce'))))                    
            if center_value == 'median':
                centers.append( float(x_frame[x+'_MEDIAN']) )   
            else:
                centers.append( float(x_frame[x+'_Q1']) )           
            norm_cols.append(x)
    
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
    to_delete =[]
    for i, x in enumerate (ncols):  
        if scales[i] == 0:
            x_frame = dist_file.iloc[:, np.where(pd.DataFrame(dist_file.columns.values)[0].str.contains(x+'_'))[0]]    
            if not x_frame.empty:            
                minmax_scales.append(float(x_frame[x+'_MAX'].subtract(x_frame[x+'_MIN'])))                            
                centers.append( float(x_frame[x+'_MIN']))
                norm_cols.append(x)
                to_delete.append(i)
        
    normalizer = Normalizer.Normalizer(minmax_scales, centers)         
    
    return norm_cols, normalizer, to_delete

def imputing_nan_values(nan_dict, distribution):        
    new_dict = {}
    for k,v in nan_dict.items():
        if v=='median':
            new_dict[k] = float(distribution[k+'_MEDIAN'])    
        else:
            new_dict[k] = v
            
    return new_dict

def allfeatures_prepro_file(file_path, raw_dir, file_name, train_num, valid_num, test_num, dividing='percentage', chunksize=500000, 
                            refNorm=True, label='DELINQUENCY_STATUS_NEXT'):
    descriptive_cols = [
        'LOAN_ID',
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
       'LLMA2_ORIG_RATE_SPREAD_NAN', 'AGI', 'AGI_NAN', 'UR', 'UR_NAN']
        
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
                'LLMA2_REO_IN_LAST_12_MONTHS': 'median', 'LLMA2_0_IN_LAST_12_MONTHS': 'median'}
        
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
                           'CURRENT_INVESTOR_CODE': ['240', '250', '253', 'U']}
      
    time_cols = ['YEAR', 'MONTH', 'PERIOD'] #no nan values        
    pd.set_option('io.hdf.default_format','table')
    
    dist_file = pd.read_csv(os.path.join(RAW_DIR, "percentile features2.csv"), sep=';', low_memory=False)
    dist_file.columns = dist_file.columns.str.upper()
    
    ncols = [x for x in numeric_cols if x.find('NAN')<0]
    robust_cols, robust_normalizer = custom_robust_normalizer(ncols, dist_file, center_value='quantile', normalizer_type='percentile_scaler')    
    minmax_cols, minmax_normalizer, to_delete = custom_minmax_normalizer(robust_cols, robust_normalizer.scale_, dist_file)        
    robust_normalizer.scale_ = np.delete(robust_normalizer.scale_,to_delete, 0)
    robust_normalizer.center_ = np.delete(robust_normalizer.center_,to_delete, 0)
    robust_cols = np.delete(robust_cols,to_delete, 0)            
    
    target_path = os.path.join(PRO_DIR, raw_dir,file_name[:-4])
    with  pd.HDFStore(target_path +'-pp.h5') as hdf:
        gflag = ''    
        i = 1            
        for chunk in pd.read_csv(file_path, chunksize = chunksize, sep=',', low_memory=False):    
            print('chunk: ', i)
            chunk.columns = chunk.columns.str.upper()                
            
            chunk.drop(chunk.index[chunk[label].isnull()], axis=0, inplace=True)
            chunk.drop(chunk.index[chunk['INVALID_TRANSITIONS']==1], axis=0, inplace=True)        
            gflag = drop_invalid_delinquency_status(chunk, gflag)
            chunk = chunk.reset_index(drop=True)   
            
            nan_cols = imputing_nan_values(nan_cols, dist_file)
            chunk.fillna(value=nan_cols, inplace=True)   
            
            logger.info('dropping invalid transitions and delinquency status, fill nan values')                  
            
            for k,v in categorical_cols.items():
                new_cols = oneHotDummies_column(chunk[k], v)
                chunk[new_cols.columns] = new_cols
                
            allfeatures_drop_cols(chunk, descriptive_cols)        
            allfeatures_drop_cols(chunk, time_cols)
            allfeatures_drop_cols(chunk, categorical_cols.keys())
            
            if chunk.isnull().any().any(): raise ValueError('There are null values...File: ' + file_name)   
                    
            
            for _ in range(4):
                chunk = chunk.sample(frac=1, axis=0, replace=False)    
            logger.info('sampled data shuffling with non replacement by 4 times')                  
            
            chunk = chunk.reset_index(drop=True)               
            if (refNorm==True):            
                chunk[robust_cols] = robust_normalizer.transform(chunk[robust_cols])
                chunk[minmax_cols] = minmax_normalizer.transform(chunk[minmax_cols])            

            chunk.to_csv(target_path +'-pp.csv', mode='a', index=False) 
            labels = allfeatures_extract_labels(chunk, columns=label)
            
            total_rows = chunk.shape[0]
            if dividing == 'percentage':            
                v_num = int(round(total_rows*(float(valid_num)/100),0))
                t_num = int(round(total_rows*(float(test_num)/100),0))
                tr_num = total_rows - (v_num + t_num)
            else:
                v_num = valid_num
                t_num = test_num
                tr_num = train_num        
            
            hdf.put('train/features', chunk.iloc[:tr_num, ], append=True)
            hdf.put('train/labels', labels.iloc[:tr_num, ], append=True)                        
            
            hdf.put('valid/features', chunk.iloc[tr_num:tr_num + v_num, ], append=True)
            hdf.put('valid/labels', labels.iloc[tr_num:tr_num + v_num, ], append=True)                        
            
            hdf.put('test/features', chunk.iloc[tr_num + v_num:, ], append=True)
            hdf.put('test/labels', labels.iloc[tr_num + v_num:, ], append=True)                        
            
            logger.info('training, validation and testing set into .h5 file')    
            del chunk
            del labels
            i +=  1   

                        
def get_h5_dataset(raw_dir, file_name):
    target_path = os.path.join(PRO_DIR, raw_dir, file_name)
    with  pd.HDFStore(target_path) as hdf:
        DATA = data_classes.Dataset(h5_dataset=hdf)
        
    return DATA

    
def allfeatures_preprocessing(raw_dir, train_num, valid_num, test_num, dividing='percentage', chunksize=500000, refNorm=True):            
    for file_path in glob.glob(os.path.join(RAW_DIR, raw_dir,"*.txt")):  
        print('Preprocessing File: ' + file_path)
        startTime = datetime.now()
        file_name = os.path.basename(file_path)
        allfeatures_prepro_file(file_path, raw_dir, file_name, train_num, valid_num, test_num, dividing=dividing, chunksize=chunksize, refNorm=refNorm)          
        print('Preprocessing Time: ', datetime.now() - startTime)     


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
    allfeatures_preprocessing('chunks_all_c100th', 70, 10, 20, dividing='percentage', chunksize=500000, refNorm=True)        



if __name__ == '__main__':        
    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    logger.propagate = False
    main(project_dir)
        