
# coding: utf-8

# ## Beginning Steps

# In[1]:


import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import psutil
import numpy as np
import logging
import numpy as np
import tensorflow as tf
import time
import glob
from tensorflow.python.framework import ops
import math
from dotenv import find_dotenv, load_dotenv
import ftplib


nb_dir = os.path.join(Path(os.getcwd()).parents[0], 'src', 'data')
if nb_dir not in sys.path:
    sys.path.insert(0, nb_dir)
print(sys.path)
import features_selection as fs
import make_dataset as md
import build_data as bd
import get_raw_data as grd
import data_classes

models_dir = os.path.join(Path(os.getcwd()).parents[0], 'src', 'models')
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
import nn_real as nn

try:
    import horovod.tensorflow as hvd
except:
    print("Failed to import horovod module. "
          "%s is intended for use with Uber's Horovod distributed training "
          "framework. To create a Docker image with Horovod support see "
          "docker-examples/Dockerfile.horovod." % __file__)
    raise

load_dotenv(find_dotenv())


# In[2]:


RAW_DIR = os.path.join(Path(os.getcwd()).parents[0], 'data', 'raw') 
PRO_DIR = os.path.join(Path(os.getcwd()).parents[0], 'data', 'processed')
RANDOM_SEED = 123  # Set the seed to get reproducable results.
DT_FLOAT = tf.float32
NP_FLOAT = np.dtype('float32')

print(RAW_DIR, PRO_DIR)


# In[3]:


hvd.init()


# ## Defining FLAGS

# In[4]:


def FLAGS_setting(FLAGS, net_number):
    # To determine an optimal set of hyperparameters, see Section 11.4.2 of the
    # deep learning book. Has (1) grid, (2) random, and (3) Bayesian
    # model-based search methods.Swersky et al. have a paper mentioned in that
    # section (published in 2014).

    # Hyperparameters
    # FLAGS.epoch_num = 2  # 14  # 17  # 35  # 15
    #print("FLAGS.epoch_num", FLAGS.epoch_num)
    # FLAGS.batch_size = 141600 # 4425 # 4000  
    FLAGS.dropout_keep = 0.9  # 0.9  # 0.95  # .75  # .6
    # ### parameters for training optimizer.
    #FLAGS.learning_rate = .1  # .075  # .15  # .25
    FLAGS.momentum = .5  # used by the momentum SGD.

    # ### parameters for inverse_time_decay
    FLAGS.decay_rate = 1
    FLAGS.decay_step = 800 * 4400 #steps_per_epoch 1 * 80000 #according to paper: 800 epochs
    FLAGS.rate_min = .0015
    # ### parameters for exponential_decay
    # FLAGS.decay_base = .96  # .96
    # FLAGS.decay_step = 15000  # 12320  # 4 * 8700

    # ### parameters for regularization
    FLAGS.reg_rate = .01 * 1e-3  # * 1e-3

    FLAGS.batch_norm = True  # False  #
    FLAGS.dropout = True
    # A flag to show the results on the held-out test set. Keep this at False.
    FLAGS.test_flag = True
    FLAGS.xla = True  # False
    FLAGS.stratified_flag = False
    #FLAGS.batch_type = 'batch'    
    FLAGS.weighted_sampling = False  # True  #
    # FLAGS.logdir =  os.path.join(Path.home(), 'real_summaries')  # 
    #FLAGS.n_hidden = 3
    #FLAGS.s_hidden = [200, 140, 140]
    # FLAGS.allow_summaries = False
    FLAGS.epoch_flag = 0    
    
    #FLAGS.max_epoch_size = 141600*70 #137 # -1
    
    FLAGS.valid_batch_size = 150000
    FLAGS.test_batch_size = 100000
    
    FLAGS.train_dir = 'chuncks_random_c1millx2_train'
    FLAGS.valid_dir = 'chuncks_random_c1millx2_valid'
    FLAGS.test_dir = 'chuncks_random_c1millx2_test'
    FLAGS.train_period=[121,279] #[121, 143] 
    FLAGS.valid_period=[280,285] #[144, 147] 
    FLAGS.test_period=[286,304] #[148, 155]
    FLAGS.epoch_num=15 
    FLAGS.max_epoch_size=-1 
    FLAGS.batch_size=4425*2 # two files!
    FLAGS.lr_decay_policy       = 'time'
    FLAGS.lr_decay_epochs       = 30
    FLAGS.lr_decay_rate         = 0.1
    FLAGS.lr_poly_power         = 2.
    FLAGS.eval = False # True=Evaluation else Training
    FLAGS.save_interval = 450
    FLAGS.nstep_burnin = 20 # step from to count consuming time for a batch
    FLAGS.summary_interval = 1800 # Time in seconds between saves of summary statistics
    FLAGS.display_every = 100 # How often (in iterations) to print out running information
    FLAGS.total_examples = 38500000 #-1 to training all dataset, otherwise the training will have a fixed length
    
    #Retrieveng from ftp:
    FLAGS.ftp_dir = 'processed/c1mill'
    
    
    if FLAGS.n_hidden < 0 : raise ValueError('The size of hidden layer must be at least 0')
    if (FLAGS.n_hidden > 0) and (FLAGS.n_hidden != len(FLAGS.s_hidden)) : raise ValueError('Sizes in hidden layers should match!')
    
    if (net_number==0):
        FLAGS.name ='default_settings'        
    elif (net_number==1):
        FLAGS.name ='2workers_1mill'
        FLAGS.batch_layer_type = 'batch'        
        
    return FLAGS


# In[5]:


import tensorflow as tf

FLAGS, UNPARSED = nn.update_parser(argparse.ArgumentParser())
print("UNPARSED", UNPARSED)
FLAGS.logdir = Path(str('/home/ubuntu/summ_4425-15ep_2wrk/'))
if not os.path.exists(os.path.join(FLAGS.logdir)): #os.path.exists
    os.makedirs(os.path.join(FLAGS.logdir))
FLAGS = FLAGS_setting(FLAGS, 1)


# In[6]:


print("FLAGS", FLAGS) #you can change the FLAGS by adding the setting before this line.


# ## Network Builder, Trainer

# In[7]:


class GPUNetworkBuilder(object):
    """This class provides convenient methods for constructing feed-forward
    networks with internal data layout of 'NCHW'.
    """
    def __init__(self,
                 # is_training,
                 dtype=DT_FLOAT,
                 activation='RELU',
                 use_batch_norm=True,
                 batch_norm_config = {'decay':   0.9,
                                      'epsilon': 1e-4,
                                      'scale':   True,
                                      'zero_debias_moving_mean': False}):
        self.dtype             = dtype
        self.activation_func   = activation
        # self.is_training       = is_training
        self.use_batch_norm    = use_batch_norm
        self.batch_norm_config = batch_norm_config
        #self._layer_counts     = defaultdict(lambda: 0)        
        
    def variable_summaries(self, name, var, allow_summaries):
        """Create summaries for the given Tensor (for TensorBoard visualization (TB graphs)).
            Calculate the mean, min, max, histogram and standardeviation for 'var' variable and save the information
            in tf.summary.

        Args: 
             name (String): the of the scope for summaring. For min, max and standardeviation 'calculate_std' is used as sub-scope.
             var (Tensor): This is the tensor variable for building summaries.
        Returns: 
            None
        Raises:        
        """
        if allow_summaries:
            with tf.name_scope(name):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('calculate_std'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.histogram('histogram', var)
                
    def _variable_on_cpu(self, name,
                     shape,
                     initializer=None,
                     regularizer=None,
                     dtype=DT_FLOAT):
        """Create a Variable or get an existing one stored on CPU memory.    

        Args:
            name (String): name of the variable.
            shape (list of ints): Shape of the variable.
            initializer: Default None. Initializer for Variable.
            regularizer (A (Tensor -> Tensor or None) function): Default None. Regularizer for Variable.
            dtype (TYPE): Type of the new variable.
        Returns:
            Variable Tensor
        """
        with tf.device('/gpu:1'): # this operation is assigned to this device, but this make a copy of data when is transferred on and off the device, which is expensive.
            var = tf.get_variable(
                name,
                shape,
                initializer=initializer,
                regularizer=regularizer,
                dtype=dtype)
        return var

    def _create_variable(self, name,
                         shape, allow_summaries, 
                         initializer=None,
                         regularizer=None,
                         dtype=DT_FLOAT):
        """Call _variable_on_cpu methods and variable_summaries for the 'name' tensor variable. 

        Args:
            name (String): name of the variable.
            shape (list of ints): Shape of the variable.
            initializer: Default None. Initializer for Variable.
            regularizer (A (Tensor -> Tensor or None) function): Default None. Regularizer for Variable.
            dtype (TYPE): Type of the new variable.
        Returns:
            Variable Tensor
        """
        var = self._variable_on_cpu(name, shape, initializer, regularizer, dtype)
        self.variable_summaries(name + '/summaries', var, allow_summaries)
        return var

    def create_weights(self, name, shape, reg_rate, allow_summaries):
        """Create a Variable initialized with weights which are truncated normal distribution and regularized by
        l1_regularizer (L1 regularization encourages sparsity, Regularization can help prevent overfitting).    

        Args:
            name (String): name of the variable.
            shape (list of ints): Shape of the variable.
        Returns:
            Variable Tensor
        """    
        dtype = DT_FLOAT
        # kernel_initializer = tf.uniform_unit_scaling_initializer(
        #     factor=1.43, dtype=DT_FLOAT)
        # kernel_initializer = tf.contrib.layers.xavier_initializer(
        #     uniform=True, dtype=DT_FLOAT)
        kernel_initializer = tf.truncated_normal_initializer(
            stddev=(1.0 / np.sqrt(shape[0])), dtype=dtype)

        regularizer = tf.contrib.layers.l2_regularizer(
            np.float32(reg_rate), 'penalty')
        return self._create_variable(name, shape, allow_summaries, kernel_initializer, regularizer,
                                dtype)

    def bias_variable(self, name, shape, layer_name, weighted_sampling): # FLAGS.weighted_sampling
        """Create a bias variable with appropriate initialization. In case of FLAGS.weighted_sampling==False
        and layer_name contains 'soft' the bias variable will contain a np.array of Negative values. Otherwise
        the bias variable will be initialized in zero.

        Args:
            name (String): name of the variable.
            shape (list of ints): Shape of the variable.
            layer_name (String): name of the layer.
        Returns:
            Variable Tensor.
        """
        def initial_bias(layer_name):
            """Get the initial value for the bias of the layer with layer_name."""
            if (not weighted_sampling) and 'soft' in layer_name:
                return np.array(
                    [-4.66, -3.81, -4.81, -3.90, -0.08, -3.90, -7.51],
                    dtype=NP_FLOAT) + NP_FLOAT(4.1)
            return 0.0

        initial_value = initial_bias(layer_name)
        with tf.name_scope(name) as scope:
            initial = tf.constant(initial_value, shape=shape)
            bias = tf.Variable(initial, name=scope)
            self.variable_summaries('summaries', bias)
        return bias        
    
    def dropout_layer(self, name, tensor_before, FLAGS):
        """Compute dropout to tensor_before with name scoping and a placeholder for keep_prob. 
        With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0.

        Args:
            name (String): name of the scope.
            tensor_before (Tensor): Variable Tensor.        
        Returns:
            Variable Tensor of the same shape of tensor_before.
        """   
        if not FLAGS.dropout:
            print('There is not dropout for' + name)
            return tensor_before
        with tf.name_scope(name) as scope:
            keep_prob = tf.placeholder(DT_FLOAT, None, name='keep_proba')
            tf.summary.scalar('keep_probability', keep_prob)
            dropped = tf.nn.dropout(tensor_before, keep_prob=keep_prob, name=scope)
            self.variable_summaries('input_dropped_out', dropped, FLAGS.allow_summaries)
        return dropped

    def batch_normalization(self, name, input_tensor, train_flag, FLAGS):
        """Perform batch normalization over the input tensor.
        Batch normalization helps avoid overfitting and we're able to use more
        aggressive (larger) learning rates, resulting in faster convergence.
        training parameter: Either a Python boolean, or a TensorFlow boolean scalar tensor (e.g. a placeholder). 
        Whether to return the output in training mode (normalized with statistics of the current batch) or in 
        inference mode (normalized with moving statistics). NOTE: make sure to set this parameter correctly, 
        or else your training/inference will not work properly.

        Args:
            name (String): name of the scope and the name of the layer.
            input_tensor (Tensor): Variable Tensor.        
        Returns:
            Variable Tensor # the same shape of input_tensor??.
        """
        # if not FLAGS.batch_norm:
        #     return input_tensor
        # train_flag = tf.get_default_graph().get_tensor_by_name('train_flag:0')
        with tf.name_scope(name):
            normalized = tf.layers.batch_normalization(
                input_tensor,
                center=True,
                scale=True,
                training=train_flag,
                name=name)  # renorm=True, renorm_momentum=0.99)
            self.variable_summaries('normalized_batch', normalized, FLAGS.allow_summaries)
        return normalized

    def layer_normalization(self, name, input_tensor, FLAGS):
        """Perform layer normalization.

        Layer normalization helps avoid overfitting and we're able to use more
        aggressive (larger) learning rates, resulting in faster convergence.
        Can be used as a normalizer function for conv2d and fully_connected.

        Given a tensor inputs of rank R, moments are calculated and normalization 
        is performed over axes begin_norm_axis ... R - 1. 
        Scaling and centering, if requested, is performed over axes begin_params_axis .. R - 1.
        """
        # if not FLAGS.batch_norm:
        #     return input_tensor
        with tf.name_scope(name):
            normalized = tf.contrib.layers.layer_norm(
                input_tensor, center=True, scale=True, scope=name)
            self.variable_summaries('normalized_layer', normalized, FLAGS.allow_summaries)
        return normalized


    def normalize(self, name, input_tensor, train_flag, FLAGS):
        """Perform either type (batch/layer) of normalization."""
        if not FLAGS.batch_norm:
            return input_tensor
        if FLAGS.batch_type.lower() == 'batch':
            return self.batch_normalization(name, input_tensor, train_flag, FLAGS)
        if FLAGS.batch_type.lower() == 'layer':
            return self.layer_normalization(name, input_tensor, FLAGS)
        raise ValueError('Invalid value for batch_type: ' + FLAGS.batch_type)

    def nn_layer(self, input_tensor, output_dim, layer_name, FLAGS, act, train_flag):
        """Create a simple neural net layer.

        It performs the affine transformation and uses the activation function to
        nonlinearize. It further sets up name scoping so that the resultant graph
        is easy to read, and adds a number of summary ops.
        """
        input_dim = input_tensor.shape[1].value    
        with tf.variable_scope(layer_name): # A context manager for defining ops that creates variables (layers).
            weights = self.create_weights('weights', [input_dim, output_dim], FLAGS.reg_rate, FLAGS.allow_summaries)
            # This is outdated and no longer applies: Do not change the order of
            # batch normalization and drop out. batch # normalization has to stay
            # __before__ the drop out layer.
            self.variable_summaries('input', input_tensor, FLAGS.allow_summaries)
            input_tensor = self.dropout_layer('dropout', input_tensor, FLAGS)
            with tf.name_scope('mix'):
                mixed = tf.matmul(input_tensor, weights)
                tf.summary.histogram('maybe_guassian', mixed)
            # Batch or layer normalization has to stay __after__ the affine
            # transformation (the bias term doens't really matter because of the
            # beta term in the normalization equation).
            # See pp. 5 of the batch normalization paper:
            # ```We add the BN transform immediately before the nonlinearity, by
            # normalizing x = W u + b```
            # biases = bias_variable('biases', [output_dim], layer_name)
            preactivate = self.normalize('layer_normalization', mixed, train_flag, FLAGS)  # + biases
            # tf.summary.histogram('pre_activations', preactivate)
            # preactivate = dropout_layer('dropout', preactivate)
            with tf.name_scope('activation') as scope:
                activations = self.activate(preactivate, funcname=act)
                tf.summary.histogram('activations', activations)
        return activations        
    
    def activate(self, input_layer, funcname=None):
        """Applies an activation function"""
        if isinstance(funcname, tuple):
            funcname = funcname[0]
            params = funcname[1:]
        if funcname is None:
            funcname = self.activation_func
        if funcname == 'LINEAR':
            return input_layer
        activation_map = {
            'IDENT':   tf.identity,
            'RELU':    tf.nn.relu,
            'RELU6':   tf.nn.relu6,
            'ELU':     tf.nn.elu,
            'SIGMOID': tf.nn.sigmoid,
            'TANH':    tf.nn.tanh,
            'LRELU':   lambda x, name: tf.maximum(params[0]*x, x, name=name)
        }
        return activation_map[funcname](input_layer, name=funcname.lower())
    
    def add_hidden_layers(self, features, architecture, FLAGS, train_flag, act=None):
        """Add hidden layers to the model using the architecture parameters."""
        hidden_out = features
        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope #JIT compiler compiles and runs parts of TF graphs via XLA, fusing multiple operators (kernel fusion) nto a small number of compiled kernels.
        with jit_scope(): #this operation will be compiled with XLA.
            for hid_i in range(1, FLAGS.n_hidden + 1):
                hidden_out = self.nn_layer(hidden_out,
                                      architecture['n_hidden_{:1d}'.format(hid_i)],
                                      '{:1d}_hidden'.format(hid_i), FLAGS, act, train_flag)
        return hidden_out        
    


# In[8]:


class FeedForwardTrainer(object):
    def __init__(self, loss_func, nstep_per_epoch=None):        
        self.loss_func = loss_func
        self.nstep_per_epoch = nstep_per_epoch
        #self.architecture = architecture
        #self.FLAGS = FLAGS
        with tf.device('/cpu:0'):
            #self.global_step = tf.contrib.framework.get_or_create_global_step()
            # tf.train.get_global_step()
            self.global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0),
                dtype=tf.int64,
                trainable=False)

    def get_learning_rate(self, initial_learning_rate):
        """Get the learning rate."""
        with tf.name_scope('learning_rate') as scope:
            if FLAGS.lr_decay_policy == 'poly':
                return tf.train.polynomial_decay(
                                        initial_learning_rate,
                                        self.global_step,
                                        decay_steps=FLAGS.epoch_num*self.nstep_per_epoch,
                                        end_learning_rate=0.,
                                        power=FLAGS.lr_poly_power,
                                        cycle=False)
            elif FLAGS.lr_decay_policy == 'exp':
                return tf.train.exponential_decay(
                                        initial_learning_rate,
                                        self.global_step,
                                        decay_steps=FLAGS.lr_decay_epochs*self.nstep_per_epoch,
                                        decay_rate=FLAGS.lr_decay_rate,
                                        staircase=True)
            else:            
                # decayed_lr = tf.train.exponential_decay(
                #     initial_learning_rate,
                #     global_step,
                #     FLAGS.decay_step,
                #     FLAGS.decay_base,
                #     staircase=False)
                decayed_lr = tf.train.inverse_time_decay(
                    initial_learning_rate,
                    self.global_step,
                    decay_steps=FLAGS.decay_step,
                    decay_rate=FLAGS.decay_rate)
                final_lr = tf.clip_by_value(
                    decayed_lr, FLAGS.rate_min, 1000, name=scope)
                tf.summary.scalar('value', final_lr)
                return final_lr
        # return self.learning_rate 

    def get_accuracy(self, labels_int, logits, name):
        """Get the accuracy tensor."""
        with tf.name_scope(name) as scope:
            # For a classifier model, we can use the in_top_k Op.
            # It returns a bool tensor with shape [batch_size] that is true for
            # the examples where the label is in the top k (here k=1)
            # of all logits for that example.
            correct = tf.nn.in_top_k(
                logits, labels_int, 1, name='correct_prediction') # returns a tensor of type bool.
            return tf.reduce_mean(tf.cast(correct, DT_FLOAT), name=scope)

    # auc = get_auc(labels, probs, True, 'metrics/auc')
    def get_auc(self, labels, scores, hist_flag, name):
        """Calculate the AUC of the two-way classifier for the given class."""

        def get_auc_using_histogram(labels, scores, class_, name):
            """Calculate the AUC."""
            class_ind = class_dict[class_.upper()]
            with tf.name_scope(name) as scope:
                auc, update_op = tf.contrib.metrics.auc_using_histogram( # his Op maintains Variables containing histograms of the scores associated with True and False labels. 
                    tf.cast(labels[:, class_ind], tf.bool),
                    scores[:, class_ind],
                    score_range=[0.0, 1.0],
                    nbins=200,
                    collections=None,
                    name=scope)
            ops.add_to_collections(ops.GraphKeys.UPDATE_OPS, update_op)
            # print(update_op.name)
            # print(auc) # it doesn't work because FailedPreconditionError (see above for traceback): Attempting to use uninitialized value metrics/auc/0//hist_accumulate/hist_true_acc
            # aucp = tf.Print(auc,[auc], message='AUC the label: ' + class_) # it doesnt work because it doesnt run in a session
            # print(aucp)
            return auc

        def get_auc_metric(labels, scores, class_, name):
            """Determine the AUC using conventional methods."""
            class_ind = class_dict[class_.upper()]
            with tf.name_scope(name) as scope:
                auc, _ = tf.metrics.auc( # Computes the approximate AUC via a Riemann sum.
                    tf.cast(labels[:, class_ind], tf.bool), # ?? Print out!!
                    scores[:, class_ind],
                    weights=None,
                    num_thresholds=200,
                    metrics_collections=None,
                    updates_collections=ops.GraphKeys.UPDATE_OPS,
                    curve='ROC',
                    name=scope)
            # print(auc.op.name)
            return auc

        classes = ['0', '3', '6', '9', 'C', 'F', 'R']
        class_dict = {classes[ind]: ind for ind in range(len(classes))}
        if hist_flag:
            auc_func = get_auc_using_histogram
        else:
            auc_func = get_auc_metric
        with tf.name_scope(name) as scope:
            aucv = [
                    auc_func(labels, scores, class_, str(ind)) for ind, class_ in enumerate(classes) # pair (index ej. 0, value ej. '0')
                   ]      
            auc_values = tf.stack( # Pack along first dim
                aucv,
                axis=0,
                name=scope)
            # aucv = tf.Print(auc_values,[auc_values], message='AUC for all labels: ')
            # print(aucv) # or maybe aucv.eval() or var = tf.Variable(aucv) and then var.eval(session=sess), or ovar = sess.run(var) but Attempting to use uninitialized value metrics/auc/Variable
            return auc_values


    # conf_mtx = get_confusion_matrix(labels_int, predictions, len(classes), 'metrics/confusion')
    def get_confusion_matrix(self, labels_int, predictions, num_classes, name):
        """Get the confusion matrix.
        Both prediction and labels must be 1-D arrays of the same shape in order for 
        this function to work.
        """
        with tf.name_scope(name) as scope:
            conf = tf.confusion_matrix(
                labels_int,
                predictions=predictions,
                num_classes=num_classes,
                dtype=tf.int32,
                name=scope,
                weights=None)
        # print(conf.op.name)
        return conf #return a K x K Matriz K = num_classes


    def get_m_hand(self, labels, scores, name):
        """Implement the M measure described in Hand.

        See ```A Simple Generalisation of the Area Under the ROC Curve for Multiple
        Class Classification Problems``` Hand, Till 2001.    

        """
        def get_auc_using_histogram(labels, scores, first_ind, second_ind, scope):
            """Calculate the AUC.
            Calculate the AUC value by maintainig histograms of boolean variables (labels and 
            scores masked by the First-Second Individuals rule).
            """
            mask = (labels[:, first_ind] + labels[:, second_ind]) > 0 #one in at least one column.
            auc, update_op = tf.contrib.metrics.auc_using_histogram( # maintains variables containing histograms of the scores associated with True, False labels. 
                tf.cast(tf.boolean_mask(labels[:, first_ind], mask), tf.bool), # tf.boolean_mask: Apply boolean mask to tensor. Numpy equivalent is tensor[mask].
                tf.boolean_mask(scores[:, first_ind], mask),
                score_range=[0.0, 1.0],
                nbins=500,
                collections=None,
                name=scope)
            ops.add_to_collections(ops.GraphKeys.UPDATE_OPS, update_op)
            # print(update_op.name)
            return auc

        temp_array = []
        with tf.name_scope(name) as main_scope:
            for first_ind in range(7):
                for second_ind in range(7):
                    if first_ind != second_ind:
                        final_name = '{:d}{:d}'.format(first_ind, second_ind)
                        with tf.name_scope(final_name) as scope:
                            auc = get_auc_using_histogram(
                                labels, scores, first_ind, second_ind, scope)
                        temp_array.append(auc)
            return tf.stack(temp_array, axis=0, name=main_scope) # Stacks a list of rank-R tensors into one rank-(R+1) tensor.


    def get_auc_pr_curve(self, labels, scores, name, num_thresholds):    
        with tf.name_scope(name) as scope:                             
            AUC_PR = []
            AUC_data = []
            for i in range(7):  
                data, update_op = tf.contrib.metrics.precision_recall_at_equal_thresholds(
                                name='pr_data',
                                predictions=scores[:, i],
                                labels=tf.cast(labels[:, i], tf.bool),
                                num_thresholds=10, use_locking=True)
                ops.add_to_collections(ops.GraphKeys.UPDATE_OPS, update_op)
                AUC_data.append((tf.stack(data.recall), tf.stack(data.precision), tf.stack(data.thresholds)))   # we cant use sklearn with tensorflow definition!
                auc, _ = tf.metrics.auc(labels[:, i], scores[:, i], weights=None, num_thresholds=10, 
                                        curve='PR', updates_collections=ops.GraphKeys.UPDATE_OPS, metrics_collections=None, summation_method='careful_interpolation') # 
                # ops.add_to_collections(ops.GraphKeys.UPDATE_OPS, update_op)
                AUC_PR.append(auc)
            # print(AUC_data)
            return tf.stack( # Pack the array of scalar tensor along one dim tensor
                AUC_PR,
                axis=0,
                name=scope), AUC_data

    def log_loss(self, labels, probs, name):
        """
        Args:
            labels: Labels tensor, int32 - [batch_size, n_classes], with one-hot
            encoded values.
            logits: Probabilities tensor, float32 - [batch_size, n_classes].
        """
        with tf.name_scope(name) as scope:
            total_loss = 0
            for j in range(probs.shape[1].value):
                loss = tf.losses.log_loss(labels[:, j], probs[:, j], loss_collection=None)
                total_loss += loss

            return tf.div(total_loss, np.float32(probs.shape[1].value), name=scope)
    
    def calculate_metrics(self, labels, logits):
        """Evaluate the quality of the logits at predicting the label.

        Args:
            labels: Labels tensor, int32 - [batch_size, n_classes], with one-hot
            encoded values.
            logits: Logits tensor, float32 - [batch_size, n_classes].
        Returns:
            A scalar float32 tensor with the fraction of examples (out of
            batch_size) that were predicted correctly.
        """
        classes = ['0', '3', '6', '9', 'C', 'F', 'R']
        with tf.name_scope('metrics'):
            labels_int = tf.argmax(labels, 1, name='intlabels') #tf.argmax: Returns the index with the largest value across axes=1 of a tensor.		
            predictions = tf.argmax(logits, 1, name='predictions')        
            probs = tf.nn.softmax(logits, name='probs') # Computes softmax activations. softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)        

        m_list = self.get_m_hand(labels, probs, 'metrics/m_measure')
        accuracy = self.get_accuracy(labels_int, logits, 'metrics/accuracy')    
        auc = self.get_auc(labels, probs, True, 'metrics/auc')    
        conf_mtx = self.get_confusion_matrix(labels_int, predictions,
                                        len(classes), 'metrics/confusion')
        loss = self.log_loss(labels, probs, 'metrics/log_loss')
        pr_auc, pr_data = self.get_auc_pr_curve(labels, probs, 'metrics/auc_pr', 200)

        # this is for the definition of the graph:
        return accuracy, conf_mtx, auc, m_list, loss, pr_auc, pr_data
    
    def training_step(self, architecture, FLAGS):        
        features = tf.placeholder(
            DT_FLOAT, [None, architecture['n_input']], name='features')
        labels = tf.placeholder(
            DT_FLOAT, [None, architecture['n_classes']], name='targets')
        # epoch_flag = tf.placeholder(tf.int32, None, name='epoch_flag')
        example_weights = tf.placeholder(
            DT_FLOAT, [None], name='example_weights')
        with tf.device('/gpu:0'):
            # Evaluate the loss and compute the gradients            
            loss, logits = self.loss_func(features, labels, architecture, FLAGS)

        with tf.device('/cpu:0'): # No in_top_k implem on GPU
            accuracy, conf_mtx, auc_list, m_list, lloss, auc_pr, auc_data = self.calculate_metrics(labels, logits)
            better_acc = tf.reduce_mean(tf.diag_part(conf_mtx / tf.reduce_sum(conf_mtx, axis=1, keepdims=True)))
            auc_mean = tf.reduce_mean(auc_list)
            m_list_mean = tf.reduce_mean(m_list)
            auc_pr_mean = tf.reduce_mean(auc_pr)
            
            with tf.name_scope('0_performance'):
                # Scalar summaries to track the loss and accuracy over time in TB.
                tf.summary.scalar('0accuracy', accuracy)
                tf.summary.scalar('1better_accuracy', better_acc)
                tf.summary.scalar('2auc_aoc', auc_mean)
                tf.summary.scalar('3m_measure', m_list_mean)
                tf.summary.scalar('4loss', loss)
                tf.summary.scalar('5log_loss', lloss)
                tf.summary.scalar('6auc_pr', auc_pr_mean)

        # Apply the gradients to optimize the loss function
        with tf.device('/gpu:0'):            
            update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
            # print(update_ops)
            with ops.control_dependencies(update_ops):
                with tf.name_scope('train') as scope:
                    # print_loss = tf.Print(loss, [loss], name='print_loss') 

                    # Create a variable to track the global step.
        #            global_step = tf.get_variable(
        #                'train/global_step',
        #                shape=[],
        #                initializer=tf.constant_initializer(0, dtype=tf.int32),
        #                trainable=False)            
                    # Horovod: adjust learning rate based on number of GPUs.
                    # optimizer = tf.train.GradientDescentOptimizer(1.0 * hvd.size())
                    final_learning_rate = self.get_learning_rate(FLAGS.learning_rate * hvd.size())

                    # optimizer = tf.train.GradientDescentOptimizer(final_learning_rate)
                    optimizer = tf.train.MomentumOptimizer(final_learning_rate, FLAGS.momentum, use_nesterov=True)
                    # optimizer = tf.train.AdagradOptimizer(final_learning_rate)

                    # Use the optimizer to apply the gradients that minimize the loss
                    # (and increment the global step counter) as a single training step.
        #            return optimizer.minimize(
        #                loss, global_step=global_step, name=scope)
                    optimizer = hvd.DistributedOptimizer(optimizer) #HVD!!
                    train_op = optimizer.minimize(loss, global_step=self.global_step, name=scope)
            
                        
        return train_op, final_learning_rate, conf_mtx, accuracy, better_acc, auc_list, auc_mean, m_list, m_list_mean, loss, lloss, auc_pr, auc_pr_mean, auc_data
    
    def init(self):
        # init_op = tf.global_variables_initializer()
        # sess.run(init_op)        
        """Add an Op to the graph to initialize the global and local variables."""
        with tf.name_scope('init') as scope:
            with tf.name_scope('global'):
                global_init = tf.global_variables_initializer()
            with tf.name_scope('local'):
                local_init = tf.local_variables_initializer()
                # print(local_init.name)
            #init_op = tf.group(global_init, local_init, name=scope)
        return global_init, local_init
        
    def sync(self, sess):
        sync_op = hvd.broadcast_global_variables(0)
        sess.run(sync_op)


# In[9]:


def loss_func(features, labels, architecture, FLAGS):
    # Build the forward model
    net = GPUNetworkBuilder(dtype=DT_FLOAT)
    train_flag = tf.placeholder(tf.bool, None, name='train_flag')
    with tf.name_scope('input_normalization') as scope:
        feature_norm = features
        net.variable_summaries('input_normalized', feature_norm, FLAGS.allow_summaries)
    hidden_out = net.add_hidden_layers(feature_norm, architecture, FLAGS, train_flag)
    # Linear output layer for the logits
    logits = (net.nn_layer(hidden_out, architecture['n_classes'],'9_softmax_linear', FLAGS, 'IDENT', train_flag))
    
    with tf.name_scope('loss') as scope:
        with tf.name_scope('regularization'):
            penalty = tf.losses.get_regularization_loss(name='penalty') #Gets the total regularization loss from an optional scope name (sum for ol + 3h + 2h + 1h).
            tf.summary.scalar('weight_norm', penalty / (1e-8 + FLAGS.reg_rate)) #for printing out
        with tf.name_scope('cross_entropy') as xentropy_scope:
            weighted_cross_entropy = tf.losses.softmax_cross_entropy(
                onehot_labels=labels,
                logits=logits,
                weights=1.0,  # weights,  #
                scope=xentropy_scope,
                loss_collection=ops.GraphKeys.LOSSES)
            tf.summary.scalar('weighted_cross_entropy', weighted_cross_entropy)
        loss= tf.add(weighted_cross_entropy, penalty, name=scope) # Returns x + y element-wise.    
            
    return loss, logits


# ## Main 

# In[10]:


global_start_time = time.time()
tf.set_random_seed(1234+hvd.rank())
np.random.seed(4321+hvd.rank())

# create logger:
log_name = FLAGS.name + '_' + str(hvd.rank())
logger = logging.getLogger(log_name)
logger.setLevel(logging.DEBUG)  # INFO, ERROR
# file handler which logs debug messages
fh = logging.FileHandler(os.path.join(FLAGS.logdir, log_name + '.log'))
fh.setLevel(logging.DEBUG)
# console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# add formatter to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add handlers to logger
logger.addHandler(fh)
logger.addHandler(ch)


# #### Download the data if it has not been downloaded

# In[11]:


def download_data(rank, FLAGS):
    server = ftplib.FTP()
    server.connect(str(os.environ.get("FTP_HOST")), int(os.environ.get("FTP_PORT")))
    server.login(os.environ.get("FTP_USER"), os.environ.get("FTP_PASS"))

    server.cwd(FLAGS.ftp_dir)               # change into ftp_dir directory
    logger.info("FTP connection stablished by worker:  {}".format(rank))
    
    filenames = server.nlst() # get filenames within the directory
        
    train_suffix = 'train_%d.h5' % rank
    if (rank==0):
        if FLAGS.eval:        
            fname_suffix = 'test_%d.h5' % rank
            filenames = [elem for elem in filenames if fname_suffix in elem]        
        else:
            valid_suffix = 'valid_%d.h5' % rank
            filenames = [elem for elem in filenames if (train_suffix in elem or valid_suffix in elem)]                
    else:
        filenames = [elem for elem in filenames if (train_suffix in elem)]

    for filename in filenames:        
        if FLAGS.eval:
            local_path = os.path.join(PRO_DIR, FLAGS.test_dir, filename)    
        else:
            if (str('train') in filename[-10:-5]):
                local_path = os.path.join(PRO_DIR, FLAGS.train_dir, filename)    
            elif (str('valid') in filename[-10:-5]):
                local_path = os.path.join(PRO_DIR, FLAGS.valid_dir, filename)   
            else: 
                continue
                
        if not os.path.exists(local_path):            
            file = open(local_path, 'wb')
            server.retrbinary('RETR '+ filename, file.write, 8*1024)            
            file.close()
            logger.info("file downloaded:  {}".format(filename))

    server.quit() # This is the “polite” way to close a connection
    logger.info("FTP connection closed by worker:  {}".format(rank))

def download_data_by_rank(rank, FLAGS):
    server = ftplib.FTP()
    server.connect(str(os.environ.get("FTP_HOST")), int(os.environ.get("FTP_PORT")))
    server.login(os.environ.get("FTP_USER"), os.environ.get("FTP_PASS"))

    server.cwd(FLAGS.ftp_dir)               # change into ftp_dir directory
    logger.info("FTP connection stablished by worker:  {}".format(rank))
    
    filenames = server.nlst() # get filenames within the directory
    fname_suffix = '_%d.h5' % rank
    filenames = [elem for elem in filenames if fname_suffix in elem]        
    
    for filename in filenames:            
        local_path = os.path.join(PRO_DIR, FLAGS.train_dir, filename)                    
        if not os.path.exists(local_path):            
            file = open(local_path, 'wb')
            server.retrbinary('RETR '+ filename, file.write, 8*1024)            
            file.close()
            logger.info("file downloaded:  {}".format(filename))

    server.quit() 
    logger.info("FTP connection closed by worker:  {}".format(rank))


#download_data_by_rank(hvd.rank(), FLAGS)
download_data(hvd.rank(), FLAGS)


# In[12]:


def get_num_records(tf_record_pattern):
    def count_records(file_name):
        count = 0
        for _ in tf.python_io.tf_record_iterator(tf_record_filename):
            count += 1
        return count
    filenames = sorted(tf.gfile.Glob(tf_record_pattern))
    nfile = len(filenames)
    return (count_records(filenames[0])*(nfile-1) +
            count_records(filenames[-1]))

def get_files_dict(FLAGS):        
    ext = "*.h5"

    if (hvd.rank()==0):
        files_dict = {'train': glob.glob(os.path.join(PRO_DIR, FLAGS.train_dir, ext)), 
                      'valid': glob.glob(os.path.join(PRO_DIR, FLAGS.valid_dir, ext)), 
                      'test': glob.glob(os.path.join(PRO_DIR, FLAGS.test_dir, ext))}
    else:
        files_dict = {'train': glob.glob(os.path.join(PRO_DIR, FLAGS.train_dir, ext))}

    return files_dict

def architecture_settings(files_dict, FLAGS):
    architecture = {}
    ok_inputs = True
    for key in files_dict.keys():
        total_records = 0
        for file in files_dict[key]:                                
            with pd.HDFStore(file) as dataset_file:
                if (ok_inputs): 
                    index_length = len(dataset_file.get_storer(key+'/features').attrs.data_columns)
                    architecture['n_input'] = dataset_file.get_storer(key+ '/features').ncols - index_length
                    architecture['n_classes'] = dataset_file.get_storer(key+'/labels').ncols - index_length
                    ok_inputs = False                
                total_records += dataset_file.get_storer(key + '/features').nrows
        architecture[key + '_num_examples'] = total_records                            
    
    if FLAGS.eval:
        architecture['total_num_examples'] = architecture['test_num_examples']
    else:
        if FLAGS.total_examples == -1:
            architecture['total_num_examples'] = architecture['train_num_examples']
        else:
            architecture['total_num_examples'] = FLAGS.total_examples 
    
    for hid_i in range(1, FLAGS.n_hidden+1):
        architecture['n_hidden_{:1d}'.format(hid_i)] = FLAGS.s_hidden[hid_i-1]
    # print('rank: ', hvd.rank(), 'architecture', architecture)   
    time.sleep(5)
    return architecture


# In[13]:


#To sum up the dataset per worker (assuming the same size of files per worker approximately):
files_dict = get_files_dict(FLAGS)
architecture = architecture_settings(files_dict, FLAGS)

nrecord = architecture['total_num_examples']


# In[14]:


logger.info("Num ranks:  {}".format(hvd.size()))
logger.info("Num of records: {}".format(nrecord))
logger.info("Total batch size: {}".format(FLAGS.batch_size * hvd.size()))
logger.info("{}, per device".format(FLAGS.batch_size))
logger.info("Data type: {}".format(DT_FLOAT)) 
logger.info("architecture: {}".format(architecture)) 
# time.sleep(5)


# In[15]:


nstep = 0
if FLAGS.epoch_num is not None:
    if (nrecord <= 0):
        logger.error("num_epochs requires data_dir to be specified")
        raise ValueError("num_epochs requires data_dir to be specified")
    nstep = nrecord * FLAGS.epoch_num //  FLAGS.batch_size # if it is kwnow the total size: (FLAGS.batch_size * hvd.size())


# In[16]:


logger.info('METRICS:  %s\r\n' % str(FLAGS))
logger.info('Number of total steps: %d' % nstep)
#logger.info('training files:  %s\r\n' % str(DATA.train._dict))
#logger.info('validation files:  %s\r\n' % str(DATA.validation._dict))
#logger.info('testing files:  %s\r\n' % str(DATA.test._dict))        


# #### TF Operations

# In[17]:


if FLAGS.eval:
    if FLAGS.test_dir is None:
        logger.error("eval requires data_dir to be specified")
        raise ValueError("eval requires data_dir to be specified")
    if hvd.size() > 1:
        logger.error("Multi-GPU evaluation is not supported")
        raise ValueError("Multi-GPU evaluation is not supported")
    #evaluator = FeedForwardEvaluator(preprocessor, eval_func)
    #logger.info("Building evaluation graph")
    #top1_op, top5_op, enqueue_ops = evaluator.evaluation_step(batch_size)
else:    
    nstep_per_epoch = nrecord // FLAGS.batch_size # if it is kwnow the total size: (FLAGS.batch_size * hvd.size())
    logger.info("Number of steps per epoch: %d" % nstep_per_epoch)
    # model_func = lambda features, labels, architecture, FLAGS: loss_func(features, labels, architecture, FLAGS) # inference_vgg(net, images, nlayer)
    trainer = FeedForwardTrainer(loss_func, nstep_per_epoch=nstep_per_epoch)
    logger.info("Building training graph")    
    train_ops, learning_rate_op, conf_mtx_op, accuracy_op, better_acc_op, auc_list_op, auc_mean_op, m_list_op, m_list_mean_op, total_loss_op, lloss_op, auc_pr_op, auc_pr_mean_op, auc_data_op = trainer.training_step(architecture, FLAGS)
    logger.info("Graph building completed....")

logger.info("Creating session")
config = tf.ConfigProto(allow_soft_placement = True)
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 10
config.gpu_options.force_gpu_compatible = True
config.gpu_options.visible_device_list = str(hvd.local_rank())


# In[18]:


print(trainer)


# #### Defining summary (writer) and checkpoint (saver) files

# In[19]:


sess = tf.Session(config=config)
global_init, local_init = trainer.init()

train_writer = None
valid_writer = None
saver = None
summary_ops = None
if hvd.rank() == 0 and FLAGS.logdir is not None:
    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir), sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'valid'), graph=None)
    summary_ops = tf.summary.merge_all()
    last_summary_time = time.time()
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    last_save_time = time.time()

if not FLAGS.eval:
    logger.info("Initializing variables")    
    sess.run([global_init, local_init])

restored = False
if hvd.rank() == 0 and saver is not None:
    ckpt = tf.train.get_checkpoint_state(FLAGS.logdir)
    checkpoint_file = os.path.join(FLAGS.logdir, "checkpoint")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        restored = True
        logger.info("Restored session from checkpoint {}".format(ckpt.model_checkpoint_path))
    else:
        if not os.path.exists(FLAGS.logdir):
            os.mkdir(FLAGS.logdir)


# #### Running evaluation from a checkpoint file

# In[20]:


# It is lefting:
if FLAGS.eval:
    if not restored:
        logger.error("No checkpoint found for evaluation")
        raise ValueError("No checkpoint found for evaluation")
    else:
        logger.info("Pre-filling input pipeline")        
        nstep = nrecord // FLAGS.test_batch_size #??
        run_evaluation(nstep, sess, top1_op, top5_op, enqueue_ops)
        sys.exit(0) #the following instructiones will not be  executed


# In[21]:


# broadcast_global_variables from hvd
trainer.sync(sess)


# #### Running Training 

# In[22]:


# Trying to restore for training:
if hvd.rank() == 0 and not restored:
    if saver is not None:
        save_path = saver.save(sess, checkpoint_file, global_step=0)
        print("Checkpoint written to", save_path)

logger.info("Writing summaries to {}".format(FLAGS.logdir))
logger.info("Training")
logger.info("  Step; Epoch; time-per-record(sec);  batchtime/worker(sec);  Loss;   Learning Rate; Accuracy; better_acc")


# #### Getting Dataset

# In[23]:


if not FLAGS.eval:

    if (hvd.rank()==0):
        DATA = md.get_h5_data(PRO_DIR, architecture, FLAGS.train_dir, FLAGS.valid_dir, None, train_period=FLAGS.train_period, valid_period=FLAGS.valid_period, test_period=FLAGS.test_period)         
    else:
        DATA = md.get_h5_data(PRO_DIR, architecture, FLAGS.train_dir, None, None, train_period=FLAGS.train_period, valid_period=FLAGS.valid_period, test_period=FLAGS.test_period) 
        
    logger.info('Features List: {}'.format(DATA.train.features_list))
    logger.info('Labels List: {}'.format(DATA.train.labels_list))
    
else:
    DATA = md.get_h5_data(PRO_DIR, architecture, None, None, FLAGS.test_dir, train_period=FLAGS.train_period, valid_period=FLAGS.valid_period, test_period=FLAGS.test_period) 
    logger.info('Features List: {}'.format(DATA.test.features_list))
    logger.info('Labels List: {}'.format(DATA.test.labels_list))


# In[24]:


def create_feed_dict(tag, DATA, FLAGS):
    """Create the feed dictionary for mapping data onto placeholders in the graph."""
    if tag == 'batch':
        features, targets, example_weights = DATA.train.next_random_batch(FLAGS.batch_size)        
    elif tag == 'train':
        features = DATA.train.orig.features
        targets = DATA.train.orig.labels
        example_weights = np.ones_like(targets.iloc[:, 1].values)
    elif tag == 'valid':
        features, targets, example_weights = DATA.validation.next_sequential_batch(FLAGS.valid_batch_size)
    else:
        features, targets, example_weights = DATA.test.next_sequential_batch(FLAGS.test_batch_size)

    # features[:, :7] = targets
    if tag == 'batch':
        k_prob_input = 0.9  # 0.9  # .85  # .75  # 0.8  # 0.6
        k_prob = FLAGS.dropout_keep
        t_flag = True
    else:
        k_prob_input = 1.0
        k_prob = 1.0
        t_flag = False

    # Change the python dictionary to an io-buffer for a better performance.
    # See here:
    # https://www.tensorflow.org/performance/performance_guide
    feed_d = {
        'features:0': features,
        'targets:0': targets,
        'example_weights:0': example_weights,        
        'train_flag:0': t_flag,
        #'epoch_flag:0': FLAGS.epoch_flag,
        #'1_hidden/dropout/keep_proba:0': k_prob_input,
        #'2_hidden/dropout/keep_proba:0': k_prob,
        # '3_hidden/dropout/keep_proba:0': k_prob,
        # '4_hidden/dropout/keep_proba:0': k_prob,
        # '5_hidden/dropout/keep_proba:0': k_prob,
        '9_softmax_linear/dropout/keep_proba:0': k_prob
    }
	
	# for any tag:
    if (FLAGS.n_hidden > 0) :
        # print ('k_prob_input', k_prob_input, type(k_prob_input))
        feed_d['1_hidden/dropout/keep_proba:0'] = k_prob_input
        for hid_i in range(2, FLAGS.n_hidden+1):
            feed_d['{:1d}_hidden/dropout/keep_proba:0'.format(hid_i)] = k_prob
    # print('feed_d', feed_d)
    # print('batch shape: ', features.shape)    
    return feed_d


# In[25]:


def reset_and_update(sess, local_init, feed_dict):
    """Reset the local variables and update the necessary update ops."""
    # sess.run(local_init) # this is necesary in each batch??check out the local variables!
        
    update_names_list = [
        'metrics/auc/{:d}/hist_accumulate/update_op'.format(i)
        for i in range(7)
    ]

    update_names_list.extend([
        'metrics/m_measure/' + str(i) + str(j) + '/hist_accumulate/update_op'
        for i in range(7) for j in range(7) if i != j
    ])

    sess.run(update_names_list, feed_dict=feed_dict)
    return


# In[26]:


def reshape_m_mtx(mtx):
    """Reshape the python list into a np array."""
    new_mtx = [0]
    for i in range(6):
        new_mtx.extend(mtx[i * 7:(i + 1) * 7])
        new_mtx.append(0)
    temp = np.array(new_mtx).reshape(7, 7)

    return temp

def calculate_better_acc(conf_mtx):
    cfsum = conf_mtx.sum(axis=1, keepdims=True)
    conf_mtx1 = np.divide(conf_mtx, cfsum, out=np.zeros_like(conf_mtx, dtype=np.float32), where=(cfsum!=0), dtype=np.float32)    
    bett_acc = conf_mtx1.diagonal().mean()
    return bett_acc

def print_stats(name, conf_mtx, accuracy, better_acc, auc_list, auc_mean, m_list, m_list_mean, lloss, auc_pr, auc_pr_mean, loss):
    """Print to logger the given stats."""        
                
    m_mtx = np.nan_to_num(reshape_m_mtx(m_list)) 
    auc_list = np.nan_to_num(auc_list)
    conf_mtx = np.array(conf_mtx, dtype=int)
            
    stdout = 'Loss in ' + name +': {:.5f}\n'.format(loss)        
    stdout = stdout + ' Avg Log_Loss in ' + name +': {:.5f}\n'.format(lloss)
    stdout = stdout +  '{:s}:'.format(name) + ' (Silly) Global-ACC={:.5f}, Better ACC={:.5f},'.format(accuracy, better_acc) +         ' Avg M-Measure={:.4f},'.format(m_list_mean) +         ' Avg AUC_AOC={:.4f}'.format(auc_mean) + ' Avg AUC_PR={:.4f}\n'.format(auc_pr_mean)
    stdout = stdout + (';').join(['Total Confusion Matrix', 'Total M-Measure Matrix', 'Total AUC_AOC', 'Total AUC_PR\n'])
    for conf_row, row, auc, auc_pr in zip(conf_mtx, m_mtx, auc_list, auc_pr):
        for conf_value in conf_row:
            stdout = stdout + '{}'.format(conf_value) + ';'
        stdout = stdout + ';'
        for value in row:
            stdout = stdout + '{:.4f}'.format(value) + ';'
        stdout = stdout + ';{:.4f}'.format(auc) + ' ;{:.4f}'.format(auc_pr) + '\n'
    stdout = stdout + '---------------------------------------------------------------------'
              
    logger.info('METRICS each %s (secs):  %s\r\n' % (FLAGS.summary_interval, stdout))


# In[27]:


#for validation set:
def acc_metrics_init(DATA):    
    acc_conf_mtx=np.zeros((DATA.train.num_classes, DATA.train.num_classes))
    acc_acc = 0
    acc_auc_list = np.zeros((DATA.train.num_classes))
    acc_m_mtx = np.zeros((DATA.train.num_classes, DATA.train.num_classes))
    acc_loss = 0
    acc_log_loss = 0
    acc_auc_pr_list = np.zeros((DATA.train.num_classes))
    epoch_metrics = list([acc_conf_mtx, acc_acc, acc_auc_list, acc_m_mtx, acc_loss, acc_log_loss, acc_auc_pr_list])
    
    return epoch_metrics

def init_metrics(metrics):
    metrics[0].fill(0)
    metrics[1] = 0
    metrics[2].fill(0)
    metrics[3].fill(0)
    metrics[4] = 0
    metrics[5] = 0
    metrics[6].fill(0)

def batch_stats(global_stats, current_stats):
    """Print the given stats."""
    acc_conf_mtx, acc_acc, acc_auc_list, acc_m_mtx, acc_loss, acc_log_loss, acc_auc_pr_list = global_stats
    conf_mtx, acc, auc_list, m_mtx_list, loss, log_loss, auc_pr_list = current_stats          
    m_mtx = reshape_m_mtx(m_mtx_list)
        
    acc_conf_mtx += conf_mtx
    acc_acc += acc
    auc_list = np.nan_to_num(auc_list)
    acc_auc_list += auc_list
    m_mtx = np.nan_to_num(m_mtx)
    acc_m_mtx += m_mtx
    acc_loss += loss
    acc_log_loss += log_loss
    acc_auc_pr_list += auc_pr_list
    
    return (acc_conf_mtx, acc_acc, acc_auc_list, acc_m_mtx, acc_loss, acc_log_loss, acc_auc_pr_list)

def batching_dataset(sess, writer, tag, DATA, FLAGS):
    if tag =='valid':
        batch_num = DATA.validation.total_num_batch(FLAGS.valid_batch_size) 
        
    # metrics = acc_metrics_init(DATA)
    sess.run(local_init)
    start_time = datetime.now()
    acc_conf_mtx=np.zeros((DATA.train.num_classes, DATA.train.num_classes))        
    metrics = [0.0] * 4
    for batch_i in range(batch_num):
        feed = create_feed_dict(tag, DATA, FLAGS)     
        reset_and_update(sess, local_init, feed)
        summary, conf_mtx, accuracy, better_acc,  lloss, loss = sess.run([summary_ops, conf_mtx_op, accuracy_op, better_acc_op, lloss_op, total_loss_op], feed_dict=feed)
        if (math.isnan(better_acc)):
            better_acc = calculate_better_acc(conf_mtx)
        acc_conf_mtx = np.add(acc_conf_mtx, conf_mtx)
        metrics = np.add(metrics, np.array([accuracy, better_acc,  lloss, loss]))
        writer.add_summary(summary, batch_i)
        writer.flush()
    
    metrics[:] = [x / batch_num for x in metrics]
    valid_time = datetime.now() - start_time
    logger.info('%s - Number of batches: %d; batch_size: %d; Total Time: %s' %(tag, batch_num,  FLAGS.valid_batch_size, valid_time))
    return acc_conf_mtx, valid_time, metrics
    


# In[28]:


if hvd.rank() == 0:
    if not FLAGS.eval:
        dtype = ['step','epoch','batch_time','Loss','LogLoss','Accuracy','Better-Accuracy','M-Measure Mean','AUC_AOC Mean','AUC_PR Mean']
        train_file = os.path.join(FLAGS.logdir, FLAGS.name + "_train.csv")
        valid_file = os.path.join(FLAGS.logdir, FLAGS.name + "_valid.csv")        
        
        if not Path(train_file).exists():
            df_train = pd.DataFrame(columns=dtype)                
            df_train.to_csv(train_file, sep=';', index=False)
        else:
            df_train = pd.read_csv(train_file, sep=';')

        if not Path(valid_file).exists():
            df_valid = pd.DataFrame(columns=dtype[:7])
            df_valid.to_csv(valid_file, sep=';', index=False)            
        else:
            df_valid = pd.read_csv(valid_file, sep=';')
        
        print('df_train: \n', df_train)
        print('df_valid: \n', df_valid)
        
    else:  # validation set:
        dtype = ['NN_name', 'NN_Number','Total Epochs', 'Execute Epochs', 'Total Training Time', 'Loss','LogLoss','Accuracy','Better-Accuracy','M-Measure Mean','AUC_AOC Mean','AUC_PR Mean']


# In[29]:


ops_to_run = [learning_rate_op, train_ops]
ops_stats = [conf_mtx_op, accuracy_op, better_acc_op, auc_list_op, auc_mean_op, m_list_op, m_list_mean_op, 
             lloss_op, auc_pr_op, auc_pr_mean_op, total_loss_op]
                    
oom = False
step0 = int(sess.run(trainer.global_step))
for step in range(step0, nstep):    
    try:
        start_time = time.time()
        epoch = step*FLAGS.batch_size // nrecord #*hvd.size()
        batch_dict= create_feed_dict('batch', DATA, FLAGS)        
        
        if (hvd.rank() == 0 and summary_ops is not None and
            (step == 0 or step+1 == nstep or
             time.time() - last_summary_time > FLAGS.summary_interval)):
            
            if step != 0:
                last_summary_time += FLAGS.summary_interval                        
                
            reset_and_update(sess, local_init, batch_dict)
            summary, conf_mtx, accuracy, better_acc, auc_list, auc_mean, m_list, m_list_mean, lloss, auc_pr, auc_pr_mean, loss, lr, _ = sess.run([summary_ops] + ops_stats + ops_to_run, feed_dict=batch_dict)                        
            train_writer.add_summary(summary, step)            
            train_writer.flush()
            if (math.isnan(better_acc)):
                better_acc = calculate_better_acc(conf_mtx)        
            elapsed = time.time() - start_time            
            #this not necessarily matches with the display at console not even with validation set, due the summary_interval!
            print_stats('---Training in Summary---', conf_mtx, accuracy, better_acc, auc_list, auc_mean, m_list, m_list_mean, lloss, auc_pr, auc_pr_mean, loss)                         
            df_train.loc[len(df_train)] = [step+1, epoch+1, elapsed, loss, lloss, accuracy, better_acc, m_list_mean, auc_mean, auc_pr_mean]                        
            
                
        else:
            accuracy, conf_mtx, better_acc, loss, lr, _ = sess.run([accuracy_op, conf_mtx_op, better_acc_op, total_loss_op] + ops_to_run, feed_dict=batch_dict)
            if (math.isnan(better_acc)):
                better_acc = calculate_better_acc(conf_mtx)        
            elapsed = time.time() - start_time
        
        if step == 0 or (step+1) % FLAGS.display_every == 0:                    
            feature_per_sec = FLAGS.batch_size / elapsed                        
            logger.info("%6i; %5i; %7.1f; %7.3f; %7.5f; %7.5f; %7.5f; %7.5f" % (
                step+1, epoch+1, feature_per_sec*hvd.size(), elapsed, loss, lr, accuracy, better_acc))        

        if (hvd.rank() == 0 and (step == 0 or step+1 == nstep or (step+1) % nstep_per_epoch == 0)):        
            #Running validation set:
            valid_conf_mtx, valid_time, metrics = batching_dataset(sess, valid_writer, 'valid', DATA, FLAGS)
            #valid_conf_mtx = np.array2string(valid_conf_mtx, formatter={'int_type':lambda x: "int(%)" % x})
            valid_conf_mtx = np.array(valid_conf_mtx, dtype=int)
            logger.info("---Validation--- Training Step: %d; Training Epoch: %d; \n Confusion Matrix:\n %s" % (step+1, epoch+1, str(valid_conf_mtx)))
            df_valid.loc[len(df_valid)] = [step+1, epoch+1, valid_time, metrics[3], metrics[2], metrics[0], metrics[1]]            
            logger.info("(Training Step, Training Epoch, loss, accuracy, better accuracy) in Validation: %6i; %5i; %7.5f; %7.5f; %7.5f" % (
                step+1, epoch+1, metrics[3], metrics[0], metrics[1]))    
            sess.run(local_init)

                    
    except KeyboardInterrupt:
        if hvd.rank() == 0:
            df_train.to_csv(train_file, index=False, mode='a', sep =';', header=False)
            df_valid.to_csv(valid_file, index=False, mode='a', sep =';', header=False)
        logger.info("Keyboard interrupt")
        break
    except tf.errors.ResourceExhaustedError:
        elapsed = -1.
        loss    = 0.
        lr      = -1
        if hvd.rank() == 0:
            df_train.to_csv(train_file, index=False, mode='a', sep =';', header=False)
            df_valid.to_csv(valid_file, index=False, mode='a', sep =';', header=False)
        oom = True
    
    if (hvd.rank() == 0 and saver is not None and
        (time.time() - last_save_time > FLAGS.save_interval or step+1 == nstep)):
        last_save_time += FLAGS.save_interval
        save_path = saver.save(sess, checkpoint_file, global_step=trainer.global_step)
        print("Checkpoint written to", save_path)
    
    if oom:
        break

if hvd.rank() == 0:                               
    df_train.to_csv(train_file, index=False, mode='a', sep =';', header=False)
    df_valid.to_csv(valid_file, index=False, mode='a', sep =';', header=False)
                               
if train_writer is not None:
    train_writer.close()

if valid_writer is not None:
    valid_writer.close()    
    
global_end_time = time.time()
#logger.info("start time is {}, end time is {}".format(global_start_time, global_end_time))
logger.info('Time used in total: %.1f seconds' % (global_end_time - global_start_time))

if oom:
    print("Out of memory error detected, exiting")
    sys.exit(-2)
        
