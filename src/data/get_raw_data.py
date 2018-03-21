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


ECON_DIR = os.path.join(Path(os.getcwd()).parents[1], 'data', 'raw', 'chunks') 
DYNAMIC_DIR = os.path.join(Path(os.getcwd()).parents[1], 'data', 'raw', 'chunks') 
RAW_DIR = os.path.join(Path(os.getcwd()).parents[1], 'data', 'raw') 
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


def retrieve_df_from_csv(path, name, parcial_name=False):
    df = pd.DataFrame()
    if parcial_name == False:
        df = pd.read_csv(os.path.join(path,name), sep=';')
    return df
    
    
    

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
            
def main(project_dir):
    """ 
    This module is in charge of::
        - Retrieving DataFrame from Raw Data .
        - Data Sampling.
    """        
    logger.info('Retrieving DataFrame from Raw Data, Data Sampling')
    
    # make some tests:
    conn = vertica_connection()
    # df_LML = pd.read_sql_query("select * from Servicing_LLMA2.LoanMaster limit 10000000", conn) # the fetch operation fails and chuncksize doesn't work.
    execute_query(conn, "select * from Servicing_LLMA2.LoanMaster limit 500", "LoanMaster")
    execute_query(conn, "select * from Servicing_LLMA2.LoanDynamic limit 500", "LoanDynamic")
    df_LM = retrieve_df_from_csv(RAW_DIR, "LoanMaster.csv")
    all_data = read_df(45)
    
#    df_InvCod = execute_query(conn, "select * from Servicing_LLMA2.InvestorCode_Typelookup")
#    lt = 'U'
#    df_InvCod_F = execute_query(conn, "select * from Servicing_LLMA2.InvestorCode_Typelookup where InvestorCode='" + lt + "'")
#    df_LoanDyn = execute_query(conn, "select * from Servicing_LLMA2.LoanDynamic limit 500")    
#    df_LoanMaster = execute_query(conn, "select * from Servicing_LLMA2.LoanMaster")    
#    df_LoanDyn_Group = execute_query(conn, "select a.Period, count (a.Loan_ID) from Servicing_LLMA2.LoanDynamic as a group by a.Period")
    


if __name__ == '__main__':        
    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
