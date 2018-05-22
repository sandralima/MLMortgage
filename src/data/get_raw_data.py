# pylint: disable=missing-docstring

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
    

def execute_query(connection, query,  name, chunksize=0):
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
        while (True):
            try:
                if (chunksize>0):
                    df = pd.DataFrame(cur.fetchmany(chunksize), columns=columnList)                
                    if df.empty: break
                    df.to_csv(os.path.join(RAW_DIR, name + "_" + str(++i) + ".csv"), index=False, sep=';', decimal='.')  #, chunksize=round(float(chunksize)/2)                    
                    del df
                    gc.collect()
                else:
                    df = pd.DataFrame(cur.fetchall(), columns=columnList)                
                    df.to_csv(os.path.join(RAW_DIR, name + ".csv"), index=False, sep=';', decimal='.') #, chunksize=round(float(chunksize)/2)
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
    
#    df_InvCod = execute_query(conn, "select * from Servicing_LLMA2.InvestorCode_Typelookup")
#    lt = 'U'
#    df_InvCod_F = execute_query(conn, "select * from Servicing_LLMA2.InvestorCode_Typelookup where InvestorCode='" + lt + "'")
#    df_LoanDyn = execute_query(conn, "select * from Servicing_LLMA2.LoanDynamic limit 500")    
#    df_LoanMaster = execute_query(conn, "select * from Servicing_LLMA2.LoanMaster")    
#    df_LoanDyn_Group = execute_query(conn, "select a.Period, count (a.Loan_ID) from Servicing_LLMA2.LoanDynamic as a group by a.Period")

def calculate_AGI_STUB(x, agi_classes):
    # return agi_classes["AGI_STUB"] if x.strip().upper()==agi_classes["DESCRIPTION"].str.strip().str.upper() else x
    ac = agi_classes["AGI_STUB"][str(x).strip().upper()==agi_classes["DESCRIPTION"].str.strip().str.upper()]
    return '0' if ac.empty else ac.iloc[0]
    
           
def year_income_taxes(file_path, columns, col_key, year_dir, agi_classes, cols_dict):
    state_df = pd.read_excel(file_path)
    state_df.drop(columns[col_key]['frows'], axis=0, inplace=True)            
        # state_df[[state_df.columns[columns[col_key]['first_col']:]]].columns = columns[col_key]['cols']
        # state_df.drop(state_df.columns[:columns[col_key]['first_col']], axis=1, inplace=True)    
            
    state_df.drop(state_df.columns[len(columns[col_key]['cols']):], axis=1, inplace=True)
    state_df.columns = columns[col_key]['cols']
    state_df.columns = state_df.columns.str.upper()    
    state_df.reset_index(inplace=True, drop=True)
    
    # filter_df = state_df.iloc[[i for i in range(0, len(state_df)-columns[col_key]['end_rows'], columns[col_key]['step'])]]
    # seq = state_df["GEO_CODE"].iloc[[i for i in range(0, len(state_df)-columns[col_key]['end_rows'], agi_classes.shape[0])]]    
    # seq = seq*agi_classes.shape[0]
    # filter_df["GEO_CODE"] = [next(seq) for count in range(filter_df.shape[0])]    
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
        
    
    # filter_df["GEO_CODE"]= filter_df["GEO_CODE"].apply(lambda x: x.strip().upper())            
    # filter_df.loc[0,"GEO_CODE"] = next(key for key, name in states.items() if name.strip().upper() in filter_df.loc[0,"GEO_CODE"].upper())                        
    # state_df["GEO_CODE"] = state_df["GEO_CODE"].astype('str')
    state_df = state_df.astype('str')
    # state_df["GEO_CODE"] = state_df["GEO_CODE"].map(lambda x: x.replace('.0', ''))    
    state_df.replace(['\.0$', '^0.0001$'], ['', ''], regex=True,  inplace=True) 
    # np.where(state_df["GEO_CODE"].astype('str').str.strip().str.find(".0")==-1, state_df["GEO_CODE"],state_df["GEO_CODE"].str.strip()[:-2])
    # state_df["GEO_CODE"].str.rstrip('.0')
    state_df.drop(nan_rows, inplace=True)
    state_df.drop_duplicates(inplace=True)                                      
    state_df.replace(['\*', '\,', 'nan'], '', inplace=True, regex=True)
    state_df = state_df.applymap(lambda x: x.strip() if type(x) is str else x)
#    state_df.replace(['.', ''], np.nan, inplace=True)
#    state_df.replace(['--', '-'], 0, inplace=True)    
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
    # all_df = pd.DataFrame()
    # myFiles = ['11zpallagi', '11zpallnoagi']
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
        # all_df = pd.concat([all_df, df], axis=0, ignore_index=True)
        
    # all_df.to_csv(os.path.join(INC_DIR, "zpallagi11-15" + ".csv"), index=False)
    
    # return all_df

def income_taxes_common_cols():
    cols_file = retrieve_df_from_csv(INC_DIR, "INCOME_TAXES_COLUMNS.csv", sep=',')
    cols_file["COL_CODE"] = cols_file["COL_CODE"].str.strip().str.upper()
    cols_file.drop_duplicates(subset='COL_CODE', keep='first', inplace=True)
    return cols_file
    
def income_taxes_consolidated():
    cols_file = retrieve_df_from_csv(INC_DIR, "INCOME_TAXES_COLUMNS_SUMMARY.csv", sep=',')
    cols_dict = dict(zip(cols_file.COL_CODE, cols_file.NEW_CODE))
    #df_1115 = income_taxes_2009_2015(cols_dict)
    df_9808 = income_taxes_1998_2008(cols_dict)       
    # all_df = pd.concat([df_1115, df_9808], axis=0, ignore_index=True)    
    # all_df.to_csv(os.path.join(INC_DIR, "zpallagi" + ".csv"), index=False)    
def area_fips_preprocessing(ur_dir, filename):
    area_file = retrieve_df_from_csv(ur_dir, filename + ".csv", sep=',')    
    area_file.drop(area_file.columns[2:], axis=1, inplace=True)    
    area_file.drop_duplicates(keep='first', inplace=True) 
    area_file = area_file.applymap(lambda x: str(int(x)).zfill(5))
    area_file['SOURCE'] = 'HUDUSER'
    area_file.to_csv(os.path.join(ur_dir, filename + "-v2" + ".csv"), index=False)    

def file_fromcsv_tovertica(directory, filename, sqltable):
    # insert into  the SQL Table:
     conn = vertica_connection()             
     cur = conn.cursor()        
     
     fs = open(os.path.join(directory,filename), 'rb') # INCOME_TAXES_COLUMNS_SUMMARY.csv
     #my_file = fs.read().decode('utf-8','ignore')
     # csvReader = csv.reader(csvfile)    
     # it only copies 4thousands of records??
     cur.copy("COPY " + sqltable+  " from stdin DELIMITER ',' ",  fs)    
     conn.commit()
     conn.close()

def income_taxes_fromcsv_tovertica():         
#     # df_LML = pd.read_sql_query("select * from MacroEconomicData.SOI_IRS", conn) # the fetch operation fails and chuncksize doesn't work.    
     
     bdir = os.listdir(os.path.join(INC_DIR, "proc_incomes_taxes"))    
     for directory in  bdir:   
         if directory =='2007processed':
             year_dir = os.path.join(INC_DIR, "proc_incomes_taxes", directory)                                  
             for file_path in glob.glob(os.path.join(year_dir, "*.csv")):            
                conn = vertica_connection()
                print(file_path)
                cur = conn.cursor()               
                csvfile = open(file_path, 'r')  
                
#                reader = csv.reader(csvfile)
#                header = next(reader)
#                headers = map((lambda x: '`'+x+'`'), header)
#                insert = 'INSERT INTO Table (' + ", ".join(headers) + ") VALUES "
#                for row in reader:
#                    values = map((lambda x: '"'+x+'"'), row)
#                    print (insert +"("+ ", ".join(values) +");" )
                    
                agi_file = retrieve_df_from_csv("",file_path, sep=',')
                cols = str(agi_file.columns.tolist())
                cols = cols.replace('\'', '')
                cols = cols.replace('[', '(')
                cols = cols.replace(']', ')')        
                cur.copy("COPY IRSIncome.IRS_SOI" + cols + " from stdin DELIMITER ',' ",  csvfile)    
                csvfile.close()                
                conn.commit()     
                conn.close()
             
        
    
        
def main(project_dir):
    """ 
    This module is in charge of::
        - Retrieving DataFrame from Raw Data .
        - Data Sampling.
    """        
    logger.info('Retrieving DataFrame from Raw Data, Data Sampling')
    #income_taxes_consolidated()    
    # income_taxes_fromcsv_tovertica()
    # file_fromcsv_tovertica(UR_DIR, 'FIPS_STATE_CODES.csv', 'MacroEconomicData.LAUS_STATE_FIPS(STATE,STATE_FIPS_CODE)')    
    # for k,v in zip2fips.items(): 
    #    if v=='01129': print(k)
    
    # make some tests:
    # conn = vertica_connection()
    # df_LML = pd.read_sql_query("select * from Servicing_LLMA2.LoanMaster limit 10000000", conn) # the fetch operation fails and chuncksize doesn't work.
    # execute_query(conn, "select * from Servicing_LLMA2.LoanMaster limit 500", "LoanMaster")
    # execute_query(conn, "select * from Servicing_LLMA2.LoanDynamic limit 500", "LoanDynamic")
    # df_LM = retrieve_df_from_csv(RAW_DIR, "LoanMaster.csv")
    # all_data = read_df(45)
    
#    df_InvCod = execute_query(conn, "select * from Servicing_LLMA2.InvestorCode_Typelookup")
#    lt = 'U'
#    df_InvCod_F = execute_query(conn, "select * from Servicing_LLMA2.InvestorCode_Typelookup where InvestorCode='" + lt + "'")
#    df_LoanDyn = execute_query(conn, "select * from Servicing_LLMA2.LoanDynamic limit 500")    
#    df_LoanMaster = execute_query(conn, "select * from Servicing_LLMA2.LoanMaster")    
#    df_LoanDyn_Group = execute_query(conn, "select a.Period, count (a.Loan_ID) from Servicing_LLMA2.LoanDynamic as a group by a.Period")
    startTime = datetime.now()
    merge_allfeatures(1, 'chunks_all_c100th', 'temporalloandynamic', 'static')
    print(datetime.now() - startTime)     
#    area_fips_preprocessing(UR_DIR, 'zcta_cbsa_rel_10')
#    file_fromcsv_tovertica(UR_DIR, 'zcta_cbsa_rel_10-v2.csv', 'MacroEconomicData.LAUS_AREA_ZIP(ZIP_CODE, AREA_FIPS_CODE)')    
#    area_fips_preprocessing(UR_DIR, 'zcta_necta_rel_10')
#    file_fromcsv_tovertica(UR_DIR, 'zcta_necta_rel_10-v2.csv', 'MacroEconomicData.LAUS_AREA_ZIP(ZIP_CODE, AREA_FIPS_CODE)')    
#    area_fips_preprocessing(UR_DIR, 'ZIP_CBSA_092HD017')
#    file_fromcsv_tovertica(UR_DIR, 'ZIP_CBSA_092017-v2.csv', 'MacroEconomicData.LAUS_AREA_ZIP(AREA_FIPS_CODE, ZIP_CODE, SOURCE)')    
    


if __name__ == '__main__':        
    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
