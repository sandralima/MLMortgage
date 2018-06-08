# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv
import build_data as bd


def get_data(num_examples, valid_num, test_num, weighted_sampling, dataset_name='MORT', stratified_flag=False, refNorm=True):
    """Load the data into the program."""
    # Loading the dataset.
    if dataset_name.upper() == 'MORT':
        
        data = bd.read_data_sets(num_examples, valid_num, test_num, #(20000,1000,1000,
                                        weighted_sampling, stratified_flag=stratified_flag, refNorm=refNorm)
#    elif dataset_name.upper() == 'MNIST':
#        data = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
#    else:
#        data = not_mnist.read_data_sets(55000)
    print('Loaded all the data....')
    return data


def get_h5_data(raw_dir, file_name):
    return bd.get_h5_dataset(raw_dir, file_name)
    

def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    # datasets = bd.read_data_sets(55000, 10000, 10000)
    # dataset = get_h5_data('chunks_all_c100th', 'temporalloandynmodifmrstaticitur1-pp.h5'):
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)    
