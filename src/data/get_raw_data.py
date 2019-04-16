# pylint: disable=missing-docstring
'''
1) Retrieving Records from Vertica: It is slower than making directly from Vertica Editor.

$ python get_raw_data.py --help

usage: get_raw_data.py [-h] [--retrieve_step RETRIEVE_STEP]
                       [--period_from PERIOD_FROM] [--period_to PERIOD_TO]
                       [--loan_number_from LOAN_NUMBER_FROM]
                       [--loan_number_to LOAN_NUMBER_TO]
                       [--retrieve_dir RETRIEVE_DIR] [--filename FILENAME]
                       [--retrieve_chunksize RETRIEVE_CHUNKSIZE]

optional arguments:
  -h, --help            show this help message and exit
  --retrieve_step RETRIEVE_STEP
                        To execute a retrieveng method
  --period_from PERIOD_FROM
                        Init Period, the default value includes all periods
  --period_to PERIOD_TO
                        End Period, the default value includes all periods
  --loan_number_from LOAN_NUMBER_FROM
                        Init Loan Number to avoid repetitions
  --loan_number_to LOAN_NUMBER_TO
                        End Loan Number to avoid repetitions
  --retrieve_dir RETRIEVE_DIR
                        Directory to save raw data inside data/raw/. If it
                        does not exist, it will be created...
  --filename FILENAME   File name for raw data inside data/raw/[retrieve_dir].
                        If it does not exist, it will be created, otherwise it
                        will open to append
  --retrieve_chunksize RETRIEVE_CHUNKSIZE
                        Chunk size to put into the h5 file...
						
Example of usage:

$ python get_raw_data.py --period_from=151 --period_to=155 --loan_number_from=1 --loan_number_to=50 --retrieve_dir=chuncks_random_c1mill --filename=dynstat_random --retrieve_chunksize=50000
'''

import os
import logging
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
import vertica_python as vp
import sys
import gc
import csv
import glob
import xlrd
import ntpath
import math
import argparse


from inspect import getsourcefile
from os.path import abspath
from datetime import datetime



ECON_DIR = os.path.join(Path(abspath(getsourcefile(lambda:0))).parents[2], 'data', 'raw', 'chunks') 
DYNAMIC_DIR = os.path.join(Path(abspath(getsourcefile(lambda:0))).parents[2], 'data', 'raw', 'chunks') 
RAW_DIR = os.path.join(Path(abspath(getsourcefile(lambda:0))).parents[2], 'data', 'raw') 
INC_DIR = os.path.join(Path(abspath(getsourcefile(lambda:0))).parents[2], 'data', 'external', 'incomes')
UR_DIR = os.path.join(Path(abspath(getsourcefile(lambda:0))).parents[2], 'data', 'external', 'ur')
DT_FLOAT = np.float32  # pylint: disable=no-member
DT_BOOL = np.uint8
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.propagate = False #it will not log to console.

def vertica_connection():
    '''Establish a connection with the Vertica database a return a connection object.
    Args:         
        None
    Returns: 
        Connection Object: connection object to the database.
    Raises:        
    '''
    logger.name = 'vertica_connection'
    conn_info = {'host': os.environ.get("VERTICA_HOST"),
             'port': int(os.environ.get("VERTICA_PORT")),
             'user': os.environ.get("VERTICA_USER"),
             'password': os.environ.get("VERTICA_PASS"),
             'database': os.environ.get("VERTICA_DB"),
             # 10 minutes timeout on queries
             'read_timeout': 600,
             # default throw error on invalid UTF-8 results
             'unicode_error': 'replace', # 'unicode_error': 'strict',
             # SSL is disabled by default
             'ssl': False             
             # 'connection_timeout': 5
             # connection timeout is not enabled by default}
             }   
    try:
        conn = vp.connect(**conn_info)
        logger.info('connection stablished...')
        return conn    
    except Exception as e:
        logger.critical('Exception Error: ' + str(e))


def execute_query(connection, query,  query_file, chunksize=0):
    '''Execute a query from a database connection and return the result in a DataFrame.
    Args: 
        connection (Connection Object): connection object to a database.
        query (String): A string containing a sql query to be executed.
    Returns: 
        DataFrame. DataFrame composed by the columns of the query executed.
    Raises:        
    '''
    logger.name = 'execute_query'
    try:        
        cur = connection.cursor()
        cur.execute(query)
        logger.info('query Executed...')
        columnList = [d.name for d in cur.description]        
        i=0               
        if not query_file.exists():
            with open(query_file, 'w') as f: 
                # f.write(columnList)             
                # wr = csv.writer(f, dialect='excel')
                # wr.writerows(columnList)
                # for d in cur.description:
                #    f.write(d.name + ';')
                # f.write('\n')
                f.write(';'.join(columnList) + '\n')
        while (True):
            try:
                if (chunksize>0):
                    # for row in cur.iterate():                    
                    df = pd.DataFrame(cur.fetchmany(chunksize), columns=None, index=None)                
                    if df.empty: break                    
                    df.to_csv(query_file, index=False, sep=';', decimal='.', header=False, mode='a')                    
                    print(++i, '-Added: ', df.shape)
                    del df
                    gc.collect()
                else:
                    df = pd.DataFrame(cur.fetchall(), columns=None, index=None)                
                    df.to_csv(query_file, index=False, sep=';', decimal='.', header=False, mode='a')
                    print('Added: ', df.shape)
                    del df
                    gc.collect()
                    break;                
            except Exception as e:
                logger.critical('Error at Retrieving records: ' + sys.exc_info()[0])
                break;        
    except vp.ProgrammingError as pe:
        logger.critical('Error at Executing Query: ' + str(pe))        
    except Exception as e:
        logger.critical('Error at Executing Query: ' + sys.exc_info()[0])       
    
    return


def retrieve_df_from_csv(path, name, parcial_name=False, sep=';'):
    df = pd.DataFrame()
    if parcial_name == False:
        df = pd.read_csv(os.path.join(path,name), sep=sep)        
    return df
    

def merge_allfeatures(chunk_ind, raw_dir, dynamic_fname, static_fname, chunksize=500000):
    '''Merge static and dynamic loan data and economic data for loans.
    
    Args: 
        chunk_ind (int): chunk index to identify the files to be merged.
    Returns: 
        Nothing
    Raises:
        NameError at the path of files.
    '''
    logger.name = 'merge_allfeatures'        
    static_data = retrieve_df_from_csv(os.path.join(RAW_DIR, raw_dir), static_fname+'-{:d}.csv'.format(chunk_ind), sep=',')
    static_data.columns = static_data.columns.str.upper()    
    incomet_data = retrieve_df_from_csv(os.path.join(RAW_DIR, raw_dir), 'income_taxes.csv', sep=',')
    incomet_data.columns = incomet_data.columns.str.upper()    
    ur_data = retrieve_df_from_csv(os.path.join(RAW_DIR, raw_dir), 'temporalur.txt')
    ur_data.columns = ur_data.columns.str.upper()
        
    for chunk in pd.read_csv(os.path.join(RAW_DIR, raw_dir, dynamic_fname+'-{:d}.txt'.format(chunk_ind)), chunksize = chunksize, sep=',', low_memory=False):
        chunk.columns = chunk.columns.str.upper()                

        chunk = chunk.merge(
            static_data,
            how='inner',
            on='LOAN_ID',
            copy=False,
            suffixes=('', '_STATIC'))
        
        chunk = chunk.merge(
        incomet_data,
        how='left',
        left_on=['PROPERTY_ZIP', 'YEAR', 'STATE'],
        right_on=['GEO_CODE', 'YEAR', 'STATE'],
        copy=False,
        suffixes=('', '_IT'))      
        
        chunk = chunk.merge(
        ur_data,
        how='left',
        left_on=['PROPERTY_ZIP', 'YEAR', 'MONTH', 'STATE'],
        right_on=['ZIP_CODE', 'YEAR', 'MONTH', 'STATE'],        
        copy=False,
        suffixes=('', '_ur'))    
    
        chunk.to_csv(os.path.join(RAW_DIR, raw_dir, dynamic_fname+'-STATIC-IT-{:d}.csv'.format(chunk_ind)), mode='a')    
    
    
    logger.info(' static shape: '
                + str(static_data.shape) + ' Income Taxes Shape: '+ str(incomet_data.shape))


def read_df(chunk_ind):
    '''Merge static and dynamic loan data and economic data for loans.
    
    Args: 
        chunk_ind (int): chunk index to identify the files to be merged.
    Returns: 
        DataFrame. Dataset merged by 'LOAN_ID', 'TIME', 'STATE'.
    Raises:
        NameError at the path of files.
    '''
    logger.name = 'read_df'
    dynamic_data = pd.read_hdf(
        os.path.join(DYNAMIC_DIR, 'dynamic_{:d}.h5'.format(chunk_ind)),
        'df',
        mode='r',
        format='f')
    static_data = pd.read_hdf(
        os.path.join(DYNAMIC_DIR, 'static_{:d}.h5'.format(chunk_ind)),
        'df',
        mode='r',
        format='f')
    merged = dynamic_data.merge(
        static_data,
        how='inner',
        on='LOAN_ID',
        copy=False,
        suffixes=('', '_static'))
    merged.drop(
        ['FOLDER', 'FOLDER_static'], axis=1, inplace=True, errors='raise')
    logger.info('merged shape:' + str(merged.shape))
    unemployment_df = pd.read_hdf(
        os.path.join(ECON_DIR, 'economics.h5'), 'df', mode='r', format='f')
    merged = merged.merge(
        unemployment_df,
        how='inner',
        on=['TIME', 'STATE'],
        copy=False,
        suffixes=('', '_ur'))
    logger.info('dynamic shape:'+ str(dynamic_data.shape) + ' static shape:'
                + str(static_data.shape) + ' unemployment shape: '+ str(unemployment_df.shape)
                + ' merged shape: '+ str(merged.shape))
    return merged


def sample_data(all_data, total_num, weight_flag=False):    
    '''Samples 'all_data' according 'total_num'. 
    
    Args: 
        all_data (DataFrame): Input Dataset.
        total_num (int): .
        weight_flag (bool): Default False. If 'weight_flag' is True it is assigned a weight for each record ponderated by 'MBA_DELINQUENCY_STATUS_next' frequency.
    Returns: 
        DataFrame. sampled Dataset.
    Raises:        
        ValueError: There are not enough samples in the input database.
    '''
    logger.name = 'sample_data'
    if total_num > all_data.shape[0]:
        raise ValueError('There are not enough samples in the input database.')

    if weight_flag: # to allow sampling with weights assigned to each data point
        column_name = 'MBA_DELINQUENCY_STATUS_next'  # 'MBA_DELINQUENCY_STATUS'
        freqs = all_data[column_name].value_counts()
        # print("freqs: ", freqs)
        map_dict = (1 / freqs.shape[0] / freqs).to_dict()
        weights = all_data[column_name].map(map_dict)
        data_df = all_data.sample(
            n=total_num, axis=0, replace=True, weights=weights)  #replace=True
        logger.info('data sampled with weights with replace')
    else:
        data_df = all_data.sample(n=total_num, axis=0, replace=False) 
        logger.info('data sampled without replace')

    for _ in range(4):
        data_df = data_df.sample(frac=1, axis=0, replace=False)
    
    logger.info('sampled data shuffling by 4 times')
    return data_df


def stratified_sample_data(all_data, percentage):
    '''Samples 'all_data' by stratifying method, taking a percentage for each group of 'MBA_DELINQUENCY_STATUS_next' label. 
    
    Args: 
        all_data (DataFrame): Input Dataset.
        percentage (Decimal [0-1]): Percentage to keep for label.        
    Returns: 
        DataFrame. sampled Dataset.
    Raises:        
        ValueError: Percentile value must be between [0-1].
    '''
    logger.name = 'stratified_sample_data'
    if (percentage > 1) or (percentage < 0):
        raise ValueError('Percentile value must be between [0-1]')
    else:
        data_df = all_data.groupby('MBA_DELINQUENCY_STATUS_next', as_index=False).apply(lambda x: x.sample(frac=percentage))
        logger.info('data sampled without replace')

    for _ in range(4):
        data_df = data_df.sample(frac=1, axis=0, replace=False)
    logger.info('sampled data shuffling by 4 times')
    return data_df
 
    
def loan_modifications():              
    conn = vertica_connection()
    df_LML = pd.read_sql_query("select * from Servicing_LLMA2.LoanMaster limit 10000000", conn) # the fetch operation fails and chuncksize doesn't work.
    execute_query(conn, "select * from Servicing_LLMA2.LoanMaster limit 500", "LoanMaster")
    execute_query(conn, "select * from Servicing_LLMA2.LoanDynamic limit 500", "LoanDynamic")
    df_LM = retrieve_df_from_csv(RAW_DIR, "LoanMaster.csv")
    all_data = read_df(45)
    
    sql = "select * from Servicing_LLMA2.LoanModificationI as lmi, Servicing_LLMA2.LoanMaster as lm, Servicing_LLMA2.LoanDynamic as ld " \
           " where lmi.Loan_ID =  lm.Loan_ID and lm.Loan_ID = ld.Loan_ID and lmi.Mod_Period = ld.Period order by lmi.Loan_ID"
    

def calculate_AGI_STUB(x, agi_classes):    
    ac = agi_classes["AGI_STUB"][str(x).strip().upper()==agi_classes["DESCRIPTION"].str.strip().str.upper()]
    return '0' if ac.empty else ac.iloc[0]
    
           
def year_income_taxes(file_path, columns, col_key, year_dir, agi_classes, cols_dict):
    state_df = pd.read_excel(file_path)
    state_df.drop(columns[col_key]['frows'], axis=0, inplace=True)                   
            
    state_df.drop(state_df.columns[len(columns[col_key]['cols']):], axis=1, inplace=True)
    state_df.columns = columns[col_key]['cols']
    state_df.columns = state_df.columns.str.upper()    
    state_df.reset_index(inplace=True, drop=True)
    
    if columns[col_key]['first_col'] > 0 : 
        state_df["AGI_STUB"] = state_df["AGI"].map(lambda x: calculate_AGI_STUB(x, agi_classes)) 
        state_df["AGI"] = np.where(state_df["AGI"].astype('str').str.strip()=="nan", state_df["GEO_CODE"], state_df["AGI"])        
    else:
        state_df["AGI_STUB"] = state_df["GEO_CODE"].map(lambda x: calculate_AGI_STUB(x, agi_classes))
            
    nan_rows = state_df.index[(state_df[state_df.columns[0]].isnull()) | (state_df[state_df.columns[0]].astype('str').str.strip()=="")].tolist()
    
    if columns[col_key]['first_col'] > 0 :
        state_df.drop(["AGI"], axis=1, inplace=True)
    
    state_df["YEAR"] = col_key 
    state_df["GEO_AREA"] = "5DigitZIP"
    state_df.loc[0:nan_rows[0]-1,"GEO_AREA"] = "State"       
    file_name = ntpath.basename(file_path)
    state_df.loc[0:nan_rows[0]-1,"GEO_CODE"] = file_name[columns[col_key]['acr']:-4].upper()      
    state_df["STATE"] = file_name[columns[col_key]['acr']:-4].upper()
        
    for i in range(0,len(nan_rows)-1,1):
        state_df.loc[nan_rows[i]+1:nan_rows[i+1]-1,["GEO_CODE"]] = state_df.loc[nan_rows[i]+1]["GEO_CODE"]
    
    try:
        state_df.loc[nan_rows[i+1]+1:nan_rows[i+1]+agi_classes.shape[0],["GEO_CODE"]] = state_df.loc[nan_rows[i+1]+1]["GEO_CODE"]        
        state_df.drop(state_df.index[nan_rows[i+1]+agi_classes.shape[0]+1:], inplace=True)                
    except  Exception  as e:
        print(str(e))
            
    state_df = state_df.astype('str')
    state_df.replace(['\.0$', '^0.0001$'], ['', ''], regex=True,  inplace=True) 
    state_df.drop(nan_rows, inplace=True)
    state_df.drop_duplicates(inplace=True)                                      
    state_df.replace(['\*', '\,', 'nan'], '', inplace=True, regex=True)
    state_df = state_df.applymap(lambda x: x.strip() if type(x) is str else x)
    state_df.replace(['^\.$', '\s', '^nan$'], np.nan, regex=True, inplace=True)
    state_df.replace(['^\-\-$', '^\-$'], 0, regex=True, inplace=True)
    state_df.rename(columns=cols_dict, inplace=True)
    state_df.to_csv(os.path.join(year_dir, str(col_key)+"processed", file_name[:-4] + "-v2" + ".csv"), index=False)
    return state_df
            
            
def income_taxes_1998_2008(cols_dict):  
    # from 1998 to 2010:
    
    columns = {
               1998: {
                      'cols': ['GEO_CODE', 'N1', 'N2', 'NUMDEP',	'A00100',	 'N00200', 	'A00200', 'N00300', 'A00300', 'N59660',	
                               'A59660',	'N07100', 'A07100', 'N00900', 'A00900',  'N02100', 'A02100', 'N04470',	'A04470'],
                      'frows': [0, 1, 2, 3, 4, 5, 6],                                           
                      'acr': 6,
                      'first_col': 0
                     },
               2001: {
                      'cols': ['GEO_CODE', 'N1', 'N2', 'NUMDEP',	'A00100',	 'N00200', 
                               'A00200', 'N00300', 'A00300', 'N07100', 'A07100', 'N00900','N02100', 'N04470'],
                      'frows': [0, 1, 2, 3, 4, 5, 6],                      
                      'acr': 6,
                      'first_col': 0
                     },
               2002: {
                      'cols': ['GEO_CODE', 'N1', 'N2', 'NUMDEP',	'A00100',	 'N00200',
                               'A00200', 'N00300', 'A00300', 'N07100', 'A07100', 'N19700',
                               'A19700', 'N00900','N02100', 'N04470'],
                      'frows': [0, 1, 2, 3, 4, 5, 6],                      
                      'acr': 7,
                      'first_col': 0
                     },
                2004: {
                      'cols': ['GEO_CODE', 'N1', 'N2', 'NUMDEP',	'A00100',	 'N00200', 	'A00200', 'N00300', 'A00300', 
                               'N00600', 'A00600',  'N01000', 'A01000', 'N00900', 'A00900',  'N02100', 'A02100', 
                               'N03150', 'A03150', 'N03300', 'A03300', 'N04470', 'A00101','A04470', 
                               'N19700', 'AGI_19700', 'A19700', 'N18300', 'AGI_18300', 'A18300', 'N09600', 'A09600', 
                               'N05800', 'A05800', 'N07100', 'A07100', 'N59660', 'A59660', 'PREP'],
                      'frows': [0, 1, 2, 3, 4, 5, 6, 7, 8],                      
                      'acr': 14,
                      'first_col': 0
                     },
                2005: {
                      'cols': ['GEO_CODE', 'N1', 'N2', 'NUMDEP',	'A00100',	 'N00200', 	'A00200', 'N00300', 'A00300', 
                               'N00600', 'A00600',  'N01000', 'A01000', 'N00900', 'A00900',  'N02100', 'A02100', 
                               'N03150', 'A03150', 'N03300', 'A03300', 'N04470', 'A00101','A04470', 
                               'N19700', 'AGI_19700', 'A19700', 'N18300', 'AGI_18300', 'A18300', 'N09600', 'A09600', 
                               'N05800', 'A05800', 'N07100', 'A07100', 'N59660', 'A59660', 'PREP'],
                      'frows': [0, 1, 2, 3, 4, 5, 6, 7, 8],                      
                      'acr': 14,
                      'first_col': 0
                     },   
                2006: {
                      'cols': ['GEO_CODE', 'N1', 'N2', 'NUMDEP',	'A00100',	 'N00200', 	'A00200', 'N00300', 'A00300', 
                               'N00600', 'A00600',  'N01000', 'A01000', 'N00900', 'A00900',  'N02100', 'A02100', 
                               'N03150', 'A03150', 'N03300', 'A03300', 'N04470', 'A00101','A04470', 
                               'N19700', 'AGI_19700', 'A19700', 'N18300', 'AGI_18300', 'A18300', 'N09600', 'A09600', 
                               'N05800', 'A05800', 'N07100', 'A07100', 'N59660', 'A59660', 'PREP'],
                      'frows': [0, 1, 2, 3, 4, 5, 6, 7, 8],                      
                      'acr': 14,
                      'first_col': 1
                     },   
                2007: {
                      'cols': ['GEO_CODE', 'N1', 'MARS2', 'PREP', 'N2', 'NUMDEP',	'A00100',	 'N00200', 	'A00200', 'N00300', 'A00300', 
                               'N00600', 'A00600', 'N00900', 'A00900',  'SCHF', 'N01000', 'A01000', 
                               'N01400', 'A01400', 'N01700', 'A01700', 'N02300',	'A02300', 'N02500', 'A02500', 
                               'N03300',	 'A03300', 'N04470', 'A04470',	'N18425',	'A18425',	 'N18450',	'A18450',
                               'N18500', 'A18500',	'N18300',	'A18300',	'N19300',	'A19300',	'N19700',	'A19700',	'N04800',	'A04800',	
                               'N07100',	'A07100', 'nf5695',	'af5695',	'n07220',	'a07220',	'n07180',	'a07180',	'n59660',	'a59660',
                               'N59720',	'a59720',	'n09600',	'a09600',	'n06500',	'a06500',	'n10300',	'a10300',	'n11900gt0',	'a11900gt0',
                               'N11900lt0'	,'a11900lt0'],
                      'frows': [0, 1, 2, 3, 4, 5, 6, 7],
                      'acr': 14,
                      'first_col': 1
                     },                
                2008: {
                      'cols': ['GEO_CODE', 'n1',	'mars2',	'prep'	,'n2', 	'numdep',	'a00100',	'a00200','a00300',	'a00600',	
                               'a00900',	'A01000',	'a01400',	'a01700',	'a02300','a02500',	'a03300',	'a04470',	'a18425',	
                               'a18450',	'a18500',	'a18300',	'a19300',	'a19700',	'a04800',	'a07100',	'af5695',	'a07220',	
                               'a07180',	'a59660',	'a59720',	'a09600',	'a06500',	'a10300',	'a11900gt0',	'a11900lt0'],
                      'frows': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                      'acr': 6,
                      'first_col': 1
                     }
               }

    agi_file = retrieve_df_from_csv(INC_DIR, "AGI_STUB.csv", sep=',')     
    all_df = pd.DataFrame()
    bdir = os.listdir(os.path.join(INC_DIR, "detailed"))    
    for col_key, directory in zip (columns, bdir):
        year_dir = os.path.join(INC_DIR, "detailed", directory)
        agi_classes = agi_file[agi_file['YEARS_RANGE'].str.contains(str(col_key))]
        if columns[col_key]['first_col'] > 0 : 
            columns[col_key]['cols'] = np.insert(columns[col_key]['cols'], 0, 'AGI')
        if col_key == 2007:
            for file_path in glob.glob(os.path.join(year_dir, "*.xls")):            
                print(file_path)
                filter_df = year_income_taxes(file_path, columns, col_key, year_dir, agi_classes, cols_dict)
                all_df = pd.concat([all_df, filter_df], axis=0, ignore_index=True)
            print(col_key, all_df.shape)    
    all_df = all_df[all_df["GEO_CODE"]!=np.NaN]
    all_df.to_csv(os.path.join(INC_DIR, "zpallagi98-08" + ".csv"), index=False)
    return all_df
            
            
    
def income_taxes_2009_2015(cols_dict): 
    myFiles = ['09zpallagi', '09zpallnoagi', '10zpallagi', '10zpallnoagi', '11zpallagi', 
               '11zpallnoagi', '12zpallagi', '12zpallnoagi', '13zpallagi', '13zpallnoagi', 
               '14zpallagi' , '14zpallnoagi', '15zpallagi', '15zpallnoagi']    # 'zipcode05', ,
    
    for file in myFiles:
        df = retrieve_df_from_csv(INC_DIR, file + ".csv", sep=',')
        df.columns = df.columns.str.upper()
        yr = int(file[0:2])        
        df["YEAR"] = (1900 + yr) if yr > 80 else (2000 + yr)
        df["GEO_AREA"] = "5DigitZIP"
        df["GEO_AREA"][df.ZIPCODE == 0] = "State"
        df["GEO_CODE"] = df.ZIPCODE
        df["GEO_CODE"][df.ZIPCODE == 0] = df["STATE"]
        df.drop(['STATEFIPS', 'ZIPCODE'], axis=1, inplace= True)        
        df = df.astype('str')    
        df.replace(['\.0$', '^0.0001$'], ['', ''], regex=True,  inplace=True) # state_df.applymap(lambda x: x.replace('.0', ''))                                             
        df.replace(['\*', '\,'], '', inplace=True, regex=True)
        df = df.applymap(lambda x: x.strip() if type(x) is str else x)
        df.replace(['^\.$', ''], np.nan, regex=True, inplace=True)
        df.replace(['^\-\-$', '^\-$'], 0, regex=True, inplace=True)
        df.drop_duplicates(inplace=True)
        df.rename(columns=cols_dict, inplace=True)
        df.to_csv(os.path.join(INC_DIR, file + "-v2" + ".csv"), index=False)
        

def income_taxes_common_cols():
    cols_file = retrieve_df_from_csv(INC_DIR, "INCOME_TAXES_COLUMNS.csv", sep=',')
    cols_file["COL_CODE"] = cols_file["COL_CODE"].str.strip().str.upper()
    cols_file.drop_duplicates(subset='COL_CODE', keep='first', inplace=True)
    return cols_file
    

def income_taxes_consolidated():
    cols_file = retrieve_df_from_csv(INC_DIR, "INCOME_TAXES_COLUMNS_SUMMARY.csv", sep=',')
    cols_dict = dict(zip(cols_file.COL_CODE, cols_file.NEW_CODE))
    income_taxes_2009_2015(cols_dict)
    df_9808 = income_taxes_1998_2008(cols_dict)            
     
     
def area_fips_preprocessing(ur_dir, filename):
    area_file = retrieve_df_from_csv(ur_dir, filename + ".csv", sep=',')    
    area_file.drop(area_file.columns[2:], axis=1, inplace=True)    
    area_file.drop_duplicates(keep='first', inplace=True) 
    area_file = area_file.applymap(lambda x: str(int(x)).zfill(5))
    area_file['SOURCE'] = 'HUDUSER'
    area_file.to_csv(os.path.join(ur_dir, filename + "-v2" + ".csv"), index=False)    


def file_fromcsv_tovertica(directory, filename, sqltable):
     conn = vertica_connection()             
     cur = conn.cursor()             
     fs = open(os.path.join(directory,filename), 'rb') 
     cur.copy("COPY " + sqltable+  " from stdin DELIMITER ',' ",  fs)    
     conn.commit()
     conn.close()


def income_taxes_fromcsv_tovertica():         
     
     bdir = os.listdir(os.path.join(INC_DIR, "proc_incomes_taxes"))    
     for directory in  bdir:   
         if directory =='2007processed':
             year_dir = os.path.join(INC_DIR, "proc_incomes_taxes", directory)                                  
             for file_path in glob.glob(os.path.join(year_dir, "*.csv")):            
                conn = vertica_connection()
                print(file_path)
                cur = conn.cursor()               
                csvfile = open(file_path, 'r')                  
                agi_file = retrieve_df_from_csv("",file_path, sep=',')
                cols = str(agi_file.columns.tolist())
                cols = cols.replace('\'', '')
                cols = cols.replace('[', '(')
                cols = cols.replace(']', ')')        
                cur.copy("COPY IRSIncome.IRS_SOI" + cols + " from stdin DELIMITER ',' ",  csvfile)    
                csvfile.close()                
                conn.commit()     
                conn.close()
             
def winyear_query(period_from, period_to, loan_from, loan_to):
    
    mystring = """
     select ld.Loan_Id, ld.Period, ld.AsOfMonth, ld.MBA_Delinquency_Status, 
     ld.MBA_Days_Delinquent, CASE WHEN ld.MBA_Days_Delinquent IS NULL THEN 1 ELSE 0 END AS MBA_DAYS_DELINQUENT_nan,
     ld.Current_Interest_Rate, CASE WHEN ld.Current_Interest_Rate IS NULL THEN 1 ELSE 0 END AS CURRENT_INTEREST_RATE_nan,
     ld.Loanage, CASE WHEN ld.Loanage IS NULL THEN 1 ELSE 0 END AS Loanage_nan,
     ld.Current_Balance, CASE WHEN ld.Current_Balance IS NULL THEN 1 ELSE 0 END AS current_balance_nan, 
     ld.Scheduled_Principal, CASE WHEN ld.Scheduled_Principal IS NULL THEN 1 ELSE 0 END AS SCHEDULED_PRINCIPAL_nan,
     ld.Scheduled_Monthly_PandI, CASE WHEN ld.Scheduled_Monthly_PandI IS NULL THEN 1 ELSE 0 END AS SCHEDULED_MONTHLY_PANDI_nan, 
     ld.Current_Investor_Code,  DATE_PART('YEAR' , ((ld.AsofMonth)::VARCHAR)::Date) as year, DATE_PART('MONTH' , ((ld.AsofMonth)::VARCHAR)::Date) as month, 
     (ld.Current_Interest_Rate - mr.Mortgage_Rate) as LLMA2_CURRENT_INTEREST_SPREAD, CASE WHEN (ld.Current_Interest_Rate - mr.Mortgage_Rate) IS NULL THEN 1 ELSE 0 END AS LLMA2_CURRENT_INTEREST_SPREAD_nan, 
     LENGTH(ld.Delinquency_History_String) - LENGTH (REPLACE(ld.Delinquency_History_String, 'C' , '')) as LLMA2_C_IN_LAST_12_MONTHS, 
     LENGTH(ld.Delinquency_History_String) - LENGTH (REPLACE(ld.Delinquency_History_String, '3' , '')) as LLMA2_30_IN_LAST_12_MONTHS, 
     LENGTH(ld.Delinquency_History_String) - LENGTH (REPLACE(ld.Delinquency_History_String, '6' , '')) as LLMA2_60_IN_LAST_12_MONTHS, 
     LENGTH(ld.Delinquency_History_String) - LENGTH (REPLACE(ld.Delinquency_History_String, '9' , '')) as LLMA2_90_IN_LAST_12_MONTHS, 
     LENGTH(ld.Delinquency_History_String) - LENGTH (REPLACE(ld.Delinquency_History_String, 'F' , '')) as LLMA2_FC_IN_LAST_12_MONTHS, 
     LENGTH(ld.Delinquency_History_String) - LENGTH (REPLACE(ld.Delinquency_History_String, 'R' , '')) as LLMA2_REO_IN_LAST_12_MONTHS, 
     LENGTH(ld.Delinquency_History_String) - LENGTH (REPLACE(ld.Delinquency_History_String, '0' , '')) as LLMA2_0_IN_LAST_12_MONTHS, 
     case when ld.Delinquency_History_String is null THEN 1 ELSE 0 END as LLMA2_HIST_LAST_12_MONTHS_MIS,    
    LEAD (ld.Period, 1) OVER (partition by ld.loan_id order by ld.Period) as Period_next,
    CASE WHEN (LEAD (ld.Period, 1) OVER (partition by ld.loan_id order by ld.Period) <> ld.Period+1) or (LEAD( ld.MBA_Delinquency_Status, 1) OVER (partition by ld.loan_id order by ld.Period) in ('S', 'T', 'X', 'Z')) 
    THEN NULL ELSE LEAD( ld.MBA_Delinquency_Status, 1) OVER (partition by ld.loan_id order by ld.Period) END AS Delinquency_Status_next,
    CASE WHEN (
            ((ld.MBA_Delinquency_Status='C') and (LEAD( ld.MBA_Delinquency_Status, 1) OVER (partition by ld.loan_id order by ld.Period)='6'))
            or 
            ((ld.MBA_Delinquency_Status='C') and (LEAD( ld.MBA_Delinquency_Status, 1) OVER (partition by ld.loan_id order by ld.Period)='9'))
            or
            ((ld.MBA_Delinquency_Status='3') and (LEAD( ld.MBA_Delinquency_Status, 1) OVER (partition by ld.loan_id order by ld.Period)='9'))
            or
            ((ld.MBA_Delinquency_Status='R') and (LEAD( ld.MBA_Delinquency_Status, 1) OVER (partition by ld.loan_id order by ld.Period)<>'R'))
            or
            ((ld.MBA_Delinquency_Status='0') and (LEAD( ld.MBA_Delinquency_Status, 1) OVER (partition by ld.loan_id order by ld.Period)<>'0'))
            ) THEN 1 ELSE 0 END AS Invalid_transitions,
    im.num_modif as Num_Modif, CASE WHEN im.num_modif IS NULL THEN 1 ELSE 0 END AS num_modif_nan,
    im.mod_per_from, im.mod_per_to, 
    im.P_Rate_to_Mod, CASE WHEN im.P_Rate_to_Mod IS NULL THEN 1 ELSE 0 END AS P_Rate_to_Mod_nan,
    im.Mod_Rate, CASE WHEN im.Mod_Rate IS NULL THEN 1 ELSE 0 END AS Mod_Rate_nan,
    im.dif_rate, CASE WHEN im.dif_rate IS NULL THEN 1 ELSE 0 END AS dif_rate_nan,
    im.P_Monthly_Pay, CASE WHEN im.P_Monthly_Pay IS NULL THEN 1 ELSE 0 END AS P_Monthly_Pay_nan,
    im.Mod_Monthly_Pay, CASE WHEN im.Mod_Monthly_Pay IS NULL THEN 1 ELSE 0 END AS Mod_Monthly_Pay_nan,
    im.dif_monthly_pay, CASE WHEN im.dif_monthly_pay IS NULL THEN 1 ELSE 0 END AS dif_monthly_pay_nan,
    im.Capiatlization_Amt as CAPITALIZATION_AMT, CASE WHEN im.Capiatlization_Amt IS NULL THEN 1 ELSE 0 END AS Capitalization_Amt_nan,
    mr.Mortgage_Rate, CASE WHEN mr.Mortgage_Rate IS NULL THEN 1 ELSE 0 END AS Mortgage_Rate_nan,
    lm.FICO_Score_Origination, lm.Initial_Interest_Rate, lm.Original_LTV, lm.Original_Balance, 
    lm.BackEnd_Ratio, CASE WHEN lm.BackEnd_Ratio IS NULL THEN 1 ELSE 0 END AS BACKEND_RATIO_nan,
    lm.Original_Term, CASE WHEN lm.Original_Term IS NULL THEN 1 ELSE 0 END AS Original_Term_nan,
    lm.Sale_Price, CASE WHEN lm.Sale_Price IS NULL THEN 1 ELSE 0 END AS SALE_PRICE_nan,
    lm.Buydown_Flag, lm.Negative_Amortization_Flag, lm.Prepay_Penalty_Flag, 
    lm.Prepay_Penalty_Term, CASE WHEN lm.Prepay_Penalty_Term IS NULL THEN 1 ELSE 0 END as PREPAY_PENALTY_TERM_nan,
    lm.Occupancy_Type, lm.Product_Type, lm.Property_Type, lm.Loan_Purpose_Category,
    lm.Documentation_Type, lm.Channel, lm.Loan_Type, 
    lm.Number_of_Units, CASE WHEN lm.Number_of_Units IS NULL THEN 1 ELSE 0 END as NUMBER_OF_UNITS_nan,
    lm.IO_Flag, 
    lm.Margin, CASE WHEN lm.Margin IS NULL THEN 1 ELSE 0 END AS MARGIN_nan,
    lm.Periodic_Rate_Cap, CASE WHEN lm.Periodic_Rate_Cap IS NULL THEN 1 ELSE 0 END AS PERIODIC_RATE_CAP_nan,
    lm.Periodic_Rate_Floor, CASE WHEN lm.Periodic_Rate_Floor IS NULL THEN 1 ELSE 0 END AS PERIODIC_RATE_FLOOR_nan, 
    lm.Lifetime_Rate_Cap, CASE WHEN lm.Lifetime_Rate_Cap IS NULL THEN 1 ELSE 0 END AS LIFETIME_RATE_CAP_nan,
    lm.Lifetime_Rate_Floor, CASE WHEN lm.Lifetime_Rate_Floor IS NULL THEN 1 ELSE 0 END AS LIFETIME_RATE_FLOOR_nan,
    lm.Rate_Reset_Frequency, CASE WHEN lm.Rate_Reset_Frequency IS NULL THEN 1 ELSE 0 END AS RATE_RESET_FREQUENCY_nan,
    lm.Pay_Reset_Frequency, CASE WHEN lm.Pay_Reset_Frequency IS NULL THEN 1 ELSE 0 END AS PAY_RESET_FREQUENCY_nan,
    lm.First_Rate_Reset_Period, CASE WHEN lm.First_Rate_Reset_Period IS NULL THEN 1 ELSE 0 END AS FIRST_RATE_RESET_PERIOD_nan,
    lm.Convertible_Flag, lm.Pool_Insurance_Flag, lm.State, lm.Property_Zip,
    CASE lm.Inferred_Collateral_Type WHEN 'P' THEN 1 ELSE 0  END as  LLMA2_PRIME, 
    CASE lm.Inferred_Collateral_Type WHEN 'S' THEN 1 ELSE 0  END as  LLMA2_SUBPRIME,
    CASE WHEN lm.Appraised_Value < lm.Sale_Price THEN 1 ELSE 0 END as LLMA2_APPVAL_LT_SALEPRICE, 
    (lm.Initial_Interest_Rate - mr.Mortgage_Rate) as LLMA2_ORIG_RATE_SPREAD, CASE WHEN (lm.Initial_Interest_Rate - mr.Mortgage_Rate) IS NULL THEN 1 ELSE 0 END AS LLMA2_ORIG_RATE_SPREAD_nan,
    irsi.AGI, CASE WHEN irsi.AGI IS NULL THEN 1 ELSE 0 END AS AGI_nan,
    tur.ur, CASE WHEN tur.ur IS NULL THEN 1 ELSE 0 END AS UR_nan,
    Substring(lm.origination_date, 0, 5) as origination_year,
    (lm.Initial_Interest_Rate - mr1.Mortgage_Rate) as LLMA2_ORIG_RATE_ORIG_MR_SPREAD, 
    CASE WHEN (lm.Initial_Interest_Rate - mr1.Mortgage_Rate) IS NULL THEN 1 ELSE 0 END AS LLMA2_ORIG_RATE_ORIG_MR_SPREAD_nan,
    sum(case when (ld.Current_Interest_Rate < mr.Mortgage_Rate) then 1 else 0 end) OVER (partition by ld.loan_id order by ld.Period) as count_int_rate_less,
    lm1.num_prime_zip, CASE WHEN lm1.num_prime_zip IS NULL THEN 1 ELSE 0 END AS num_prime_zip_nan
    from "Servicing_LLMA2"."LoanDynamic" as ld 
    inner join (select frld.row_count, --row_number() over (order by lm.loan_id) as row_count, 
                lm.Loan_Id, lm."FICO_Score_Origination", lm."Initial_Interest_Rate", lm."Original_LTV", lm."Original_Balance", 
                lm."BackEnd_Ratio",
                lm."Original_Term", 
                lm."Sale_Price", 
                lm."Buydown_Flag", lm."Negative_Amortization_Flag", lm."Prepay_Penalty_Flag", lm."Prepay_Penalty_Term", lm."Occupancy_Type", lm."Product_Type", lm."Property_Type", lm."Loan_Purpose_Category",
                lm."Documentation_Type", lm."Channel", lm."Loan_Type", lm."Number_of_Units", lm."IO_Flag", 
                lm."Margin", 
                lm."Periodic_Rate_Cap", 
                lm."Periodic_Rate_Floor", 
                lm."Lifetime_Rate_Cap", 
                lm."Lifetime_Rate_Floor", 
                lm."Rate_Reset_Frequency", 
                lm."Pay_Reset_Frequency", 
                lm."First_Rate_Reset_Period", 
                lm."Convertible_Flag",  
                lm."Pool_Insurance_Flag", lm."State", lm.Property_Zip, lm."Inferred_Collateral_Type", lm."Appraised_Value",
                lm.origination_date --as original_origination_date,
                from "Servicing_LLMA2"."LoanMaster" as lm
                inner join public.TemporalFirstRecordLD as frld on (lm.Loan_Id = frld.Loan_Id) --and frld.last_period>251
                inner join "MacroEconomicData"."LAUS_STATE_FIPS" as lsf on (lm.state = lsf.state)
                where lm."FICO_Score_Origination">0 and lm."Original_Balance">0 and lm."Initial_Interest_Rate">0 and lm."Original_LTV">0
                and length(lm.property_zip)=5                
                )
    as lm on (ld.Loan_Id = lm.Loan_Id)
    left outer join public.interm_modif as im on (ld.Loan_Id = im.loan_id and ld."Period" >= im.mod_per_from and ld."Period" <= im.mod_per_to)
    left outer join (
            select irsi.geo_code, irsi.year, irsi."STATE", count(*) as nrows, min(agi) as agi_min, max(agi) as agi_max, avg(agi) agi_avg,
            case when (min(agi)=0) then max(agi) else avg(agi) end as agi
            from "IRSIncome"."IRS_SOI" as irsi where irsi."AGI_STUB"='0'
            and irsi.agi is not null
            group by geo_code, year, state) as irsi on (lm.Property_Zip = irsi.geo_code and DATE_PART('YEAR' , ((ld.AsofMonth)::VARCHAR)::Date)= irsi.year and irsi."STATE"= lm.state)
    left outer join (select * from "MacroEconomicData"."MortgageRates" as mr where mr."Rate_Name" = 'FRM30' and mr."Rate_Source"='FreddieMac') as mr on (ld."AsOfMonth"= mr."Asofmon")
    left outer join (select * from "MacroEconomicData"."MortgageRates" as mr where mr."Rate_Name" = 'FRM30' and mr."Rate_Source"='FreddieMac') as mr1 on (((lm.origination_date * 100 + 1)::VARCHAR)::Date = mr1."Asofmon")
    left outer join (
            select lm.property_zip, count(*) as num_prime_zip
            from "Servicing_LLMA2"."LoanMaster" as lm 
            where lm."Inferred_Collateral_Type" = 'P'  and
            lm."FICO_Score_Origination">0 and lm."Original_Balance">0 and lm."Initial_Interest_Rate">0 and lm."Original_LTV">0
            and length(lm.property_zip)=5
            group by lm.property_zip
            ) as lm1 on (lm.property_zip=lm1.property_zip)
    left join public."TemporalUR" as tur  on (tur.year=DATE_PART('YEAR' , ((ld.AsofMonth)::VARCHAR)::Date)::int and tur.month::int=DATE_PART('MONTH' , ((ld.AsofMonth)::VARCHAR)::Date) and lm.state=tur.state and lm.property_zip=tur.property_zip)
    """ 
    if (loan_from == 0 and period_from==0):  #default values, retrieves all from database...
        args_string = " order by ld.Loan_ID asc, ld.Period asc;"
    elif (loan_from > 0 and period_from>0):
        args_string = " where lm.row_count>=%s and lm.row_count<=%s " \
                      " and ld.period>=%s and ld.period<=%s" \
                      " order by ld.Loan_ID asc, ld.Period asc;" %(str(loan_from), str(loan_to), str(period_from), str(period_to+1))
    elif (loan_from > 0 and period_from==0):
        args_string = """ where lm.row_count>=%s and lm.row_count<=%s                  
                      order by ld.Loan_ID asc, ld.Period asc;""" %(str(loan_from), str(loan_to))
    elif (loan_from == 0 and period_from>0):
        args_string = " where ld.period>=%s and ld.period<=%s " \
                      " order by ld.Loan_ID asc, ld.Period asc;" %(str(period_from), str(period_to+1))

    return  mystring + args_string
        
def retrieve_windataset(period_from, period_to, loan_from, loan_to, retrieve_dir, filename, chunksize):    
    
    conn = vertica_connection()
    query = winyear_query(period_from, period_to, loan_from, loan_to)
    if not os.path.exists(os.path.join(RAW_DIR, retrieve_dir)): #os.path.exists
        os.makedirs(os.path.join(RAW_DIR, retrieve_dir))
    query_file = Path(os.path.join(RAW_DIR, retrieve_dir, filename + ".txt"))
    execute_query(conn, query,  query_file, chunksize=chunksize)
    # conn.commit()     
    conn.close()

def update_parser(parser):
    """Parse the arguments from the CLI and update the parser."""    
    parser.add_argument(
        '--retrieve_step',
        type=str,
        default='windataset', #
        help='To execute a retrieveng method')    
    
    #this is for retrieve_windataset:
    parser.add_argument(
        '--period_from',
        type=int,        
        default=151,
        help='Init Period, the default value includes all periods')
    parser.add_argument(
        '--period_to',
        type=int,        
        default=155,
        help='End Period, the default value includes all periods') 
    parser.add_argument(
        '--loan_number_from',
        type=int,        
        default=1,
        help='Init Loan Number to avoid repetitions')
    parser.add_argument(
        '--loan_number_to',
        type=int,        
        default=50,
        help='End Loan Number to avoid repetitions')
    parser.add_argument(
        '--retrieve_dir',
        type=str,
        default='chuncks_random_c1mill',
        help='Directory to save raw data inside data/raw/. If it does not exist, it will be created...')    
    parser.add_argument(
        '--filename',
        type=str,
        default='results',
        help='File name for raw data inside data/raw/[retrieve_dir]. If it does not exist, it will be created, otherwise it will open to append')    
    parser.add_argument(
        '--retrieve_chunksize',
        type=int,
        default=5000,
        help='Chunk size to put into the h5 file...')    
            
    return parser.parse_known_args()
    
def main(project_dir):
    """ 
    This module is in charge of::
        - Retrieving DataFrame from Raw Data .
        - Data Sampling.
    """        
    logger.info('Retrieving DataFrame from Raw Data, Data Sampling')
    FLAGS, UNPARSED = update_parser(argparse.ArgumentParser())    
    print("UNPARSED", UNPARSED)    
        
    if FLAGS.retrieve_step == 'windataset':
        startTime = datetime.now()                
        retrieve_windataset(FLAGS.period_from, FLAGS.period_to, FLAGS.loan_number_from, FLAGS.loan_number_to, FLAGS.retrieve_dir, FLAGS.filename, FLAGS.retrieve_chunksize)
        print('Retrieving - Time: ', datetime.now() - startTime)
    else: 
        print('Invalid retrieve_step...')


if __name__ == '__main__':        
    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)