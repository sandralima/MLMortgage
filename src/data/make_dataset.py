# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv
import build_data as bd


def get_data(num_examples, valid_num, test_num, weight_flag=False, dataset_name='MORT', stratified_flag=False, refNorm=True):
    """Load the data into the program."""
    # Loading the dataset.
    if dataset_name.upper() == 'MORT':
        
        data = bd.read_data_sets(num_examples, valid_num, test_num, #(20000,1000,1000,
                                        weight_flag=weight_flag, stratified_flag=stratified_flag, refNorm=refNorm)
#    elif dataset_name.upper() == 'MNIST':
#        data = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
#    else:
#        data = not_mnist.read_data_sets(55000)
    print('Loaded all the data....')
    return data


def get_h5_data(PRO_DIR, architecture, train_dir, valid_dir, test_dir, train_period=[121, 279], valid_period=[280,285], test_period=[286, 304]):
    try:              
        return bd.get_h5_dataset(PRO_DIR, architecture, train_dir, valid_dir, test_dir, train_period=train_period, valid_period=valid_period, test_period=test_period)
    except  Exception  as e:        
        raise ValueError('Error in retrieving the DATA object: ' + train_dir + ' , ' + valid_dir + ' , ' +  test_dir + ' ' + str(e))


def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    # dataset = get_h5_data()
    # print(dataset.train.num_examples, dataset.validation.num_examples, dataset.test.num_examples)
    datasets = get_data(55000, 10000, 10000)
    main(project_dir)    
