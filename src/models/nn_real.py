# coding: utf-8
#
# Author: Vahid Montazerhodjat,
# Summer 2017
#
"""Create a multilayer perceptron model using low-level tensorflow modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-name-in-module
# pylint: disable=not-context-manager
# pylint: disable=unused-import
import argparse
import os

import sys
from pathlib import Path
sys.path.insert(0, os.path.join(Path(os.getcwd()).parents[1], 'src', 'data'))
# from build_data import encode_sparsematrix
import make_dataset as md

import numpy as np
import tensorflow as tf
from tensorboard import summary as summary_lib
# from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.client import timeline
from tensorflow.python.framework import ops
from sklearn import metrics
import math
import pandas as pd
from datetime import datetime
import glob
from os.path import abspath
from pathlib import Path
from inspect import getsourcefile
from io import IOBase
import psutil


RANDOM_SEED = 123  # Set the seed to get reproducable results.
DT_FLOAT = tf.float32
NP_FLOAT = np.dtype('float32')
PRO_DIR = os.path.join(Path(abspath(getsourcefile(lambda:0))).parents[2], 'data', 'processed') 

# TO1DO(vahid):
# 1.   Change the M-measure to H-measure proposed by Hand et al.
# 2.   Explore SavedModel API in TF and see if we can use it for
#          bagging/evaluation.
# 3.   Add lagged default/prepayment rates per prime/subprime per zip-code
#          to the database.
# 4.   Explore the bagging method proposed by Wallace to address the
#          imabalance characteristic of the data. (bootstrap the original
#          database, create copies by downsampling the majority to balance
#          the minority, train on each copy, then merge the classifiers).
# 5.   Look into the duplicate update ops in train from auc calculation.
# 6.   Look into queue-runner (from Stanford) to read the data in.


def variable_summaries(name, var, allow_summaries):
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


def _variable_on_cpu(name,
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


def _create_variable(name,
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
    var = _variable_on_cpu(name, shape, initializer, regularizer, dtype)
    variable_summaries(name + '/summaries', var, allow_summaries)
    return var


def create_weights(name, shape, reg_rate, allow_summaries):
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
    return _create_variable(name, shape, allow_summaries, kernel_initializer, regularizer,
                            dtype)


def bias_variable(name, shape, layer_name, weighted_sampling): # FLAGS.weighted_sampling
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
        variable_summaries('summaries', bias)
    return bias


def dropout_layer(name, tensor_before, FLAGS):
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
        variable_summaries('input_dropped_out', dropped, FLAGS.allow_summaries)
    return dropped


def batch_normalization(name, input_tensor, train_flag, FLAGS):
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
        variable_summaries('normalized_batch', normalized, FLAGS.allow_summaries)
    return normalized


def layer_normalization(name, input_tensor, FLAGS):
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
        variable_summaries('normalized_layer', normalized, FLAGS.allow_summaries)
    return normalized


def normalize(name, input_tensor, train_flag, FLAGS):
    """Perform either type (batch/layer) of normalization."""
    if not FLAGS.batch_norm:
        return input_tensor
    if FLAGS.batch_type.lower() == 'batch':
        return batch_normalization(name, input_tensor, train_flag, FLAGS)
    if FLAGS.batch_type.lower() == 'layer':
        return layer_normalization(name, input_tensor, FLAGS)
    raise ValueError('Invalid value for batch_type: ' + FLAGS.batch_type)


def nn_layer(input_tensor, output_dim, layer_name, FLAGS, act, train_flag):
    """Create a simple neural net layer.

    It performs the affine transformation and uses the activation function to
    nonlinearize. It further sets up name scoping so that the resultant graph
    is easy to read, and adds a number of summary ops.
    """
    input_dim = input_tensor.shape[1].value    
    with tf.variable_scope(layer_name): # A context manager for defining ops that creates variables (layers).
        weights = create_weights('weights', [input_dim, output_dim], FLAGS.reg_rate, FLAGS.allow_summaries)
        # This is outdated and no longer applies: Do not change the order of
        # batch normalization and drop out. batch # normalization has to stay
        # __before__ the drop out layer.
        variable_summaries('input', input_tensor, FLAGS.allow_summaries)
        input_tensor = dropout_layer('dropout', input_tensor, FLAGS)
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
        preactivate = normalize('layer_normalization', mixed, train_flag, FLAGS)  # + biases
        # tf.summary.histogram('pre_activations', preactivate)
        # preactivate = dropout_layer('dropout', preactivate)
        with tf.name_scope('activation') as scope:
            activations = act(preactivate, name=scope)
            tf.summary.histogram('activations', activations)
    return activations


def calculate_loss(labels, logits, weights, reg_rate):
    """Calculate the loss from the logits and the labels.
    Returns a list of regularization losses as Tensors and Creates a cross-entropy 
    loss using tf.nn.softmax_cross_entropy_with_logits.

    Args:
        labels: Labels tensor, int32 - [batch_size].
        logits: Logits tensor, float32 - [batch_size, n_classes].
        weights: List of weights tensors, each of dtype=float32.
    Returns:
        loss: Loss tensor of the same type as logits.
    """
    with tf.name_scope('loss') as scope:
        # print_labels = tf.Print(labels, [labels], name='print_labels') 
        # print_logits = tf.Print(logits, [logits], name='print_logits')  
        # tf.print('calculate_loss:logits: ', logits)  # only at design level
        with tf.name_scope('regularization'):
            penalty = tf.losses.get_regularization_loss(name='penalty') #Gets the total regularization loss from an optional scope name (sum for ol + 3h + 2h + 1h).
            # print_penalty = tf.Print(penalty, [penalty], name='print_penalty') # penalty is equal to print_penalty, it is a scalar, I guess from only the output layer
            tf.summary.scalar('weight_norm', penalty / (1e-8 + reg_rate)) #for printing out
        with tf.name_scope('cross_entropy') as xentropy_scope:
            # ## hard (sparse) softmax: only one class can be active.
            # labels = tf.to_int64(labels)
            # cross_entropy = tf.reduce_mean(
            #     tf.nn.sparse_softmax_cross_entropy_with_logits(
            #         labels=labels, logits=logits),
            #     name=xentropy_scope)

            # # ## soft softmax (with proper weights) this is from the
            # # `losses` submodule of TF instead of the `nn` submodule.
            weighted_cross_entropy = tf.losses.softmax_cross_entropy(
                onehot_labels=labels,
                logits=logits,
                weights=1.0,  # weights,  #
                scope=xentropy_scope,
                loss_collection=ops.GraphKeys.LOSSES)
            # print_weighted_cross_entropy = tf.Print(weighted_cross_entropy, [weighted_cross_entropy], name='print_weighted_cross_entropy')
            tf.summary.scalar('weighted_cross_entropy', weighted_cross_entropy)

            # cross_entropy = tf.losses.softmax_cross_entropy(
            #     onehot_labels=labels,
            #     logits=logits,
            #     weights=1.0,
            #     loss_collection=None)
            # tf.summary.scalar('cross_entropy', cross_entropy)
        return tf.add(weighted_cross_entropy, penalty, name=scope) # Returns x + y element-wise.


def train(loss, FLAGS, epoch_flag):
    """Set up the training Ops.

    Create an optimizer and apply the gradients to all trainable variables. The
    Op this function returns is what must be passed to the `sess.run()` call to
    have the model execute training.

    Args:
        loss: Loss tensor that calculate_loss returns.
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
    """

    def get_learning_rate(initial_learning_rate, epoch_flag):
        """Get the learning rate."""
        with tf.name_scope('learning_rate') as scope:
            # decayed_lr = tf.train.exponential_decay(
            #     initial_learning_rate,
            #     global_step,
            #     FLAGS.decay_step,
            #     FLAGS.decay_base,
            #     staircase=False)
            decayed_lr = tf.train.inverse_time_decay(
                initial_learning_rate,
                epoch_flag,
                decay_steps=FLAGS.decay_step,
                decay_rate=FLAGS.decay_rate)
            final_lr = tf.clip_by_value(
                decayed_lr, FLAGS.rate_min, 1000, name=scope)
            tf.summary.scalar('value', final_lr)
        return final_lr

    # Because of batch normalization, we need to update the control
    # dependencies of the train op. So, do __not__ remove the following two
    # lines.
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
            
            final_learning_rate = get_learning_rate(FLAGS.learning_rate, epoch_flag)

            # optimizer = tf.train.GradientDescentOptimizer(final_learning_rate)
            optimizer = tf.train.MomentumOptimizer(final_learning_rate, FLAGS.momentum, use_nesterov=True)
            # optimizer = tf.train.AdagradOptimizer(final_learning_rate)

            # Use the optimizer to apply the gradients that minimize the loss
            # (and increment the global step counter) as a single training step.
#            return optimizer.minimize(
#                loss, global_step=global_step, name=scope)
            return optimizer.minimize(
                loss, global_step=None, name=scope)



def get_accuracy(labels_int, logits, name):
    """Get the accuracy tensor."""
    # tf.metrics.accuracy(
    #     labels,
    #     predictions,
    #     weights=None,
    #     metrics_collections=None,
    #     updates_collections=None,
    #     name=None)

    with tf.name_scope(name) as scope:
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        correct = tf.nn.in_top_k(
            logits, labels_int, 1, name='correct_prediction') # returns a tensor of type bool.
        # correct = tf.equal(
        #     tf.argmax(labels, 1),
        #     tf.argmax(logits, 1),
        #     name='correct_prediction')
        # return the fraction of true entries.
        return tf.reduce_mean(tf.cast(correct, DT_FLOAT), name=scope)

# auc = get_auc(labels, probs, True, 'metrics/auc')
def get_auc(labels, scores, hist_flag, name):
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
        # for ind, class_ in enumerate(classes):            
        #    aucp = auc_func(labels, scores, class_, str(ind))
            # tf.Print(aucp)
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
def get_confusion_matrix(labels_int, predictions, num_classes, name):
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


def get_m_hand(labels, scores, name):
    """Implement the M measure described in Hand.

    See ```A Simple Generalisation of the Area Under the ROC Curve for Multiple
    Class Classification Problems``` Hand, Till 2001.    

    """

    def get_auc_using_histogram(labels, scores, first_ind, second_ind, scope):
        """Calculate the AUC.
        Calculate the AUC value by maintainig histograms of boolean variables (labels and 
        scores masked by the First-Second Individuals rule).
        """
        
        # tf.boolean_mask(scores[:, first_ind] /
        #                 (scores[:, first_ind] + scores[:, second_ind]),
        #                 mask)
        # Why the second individual?? to speed the execution??
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
#                        if (auc==np.nan): 
#                            auc=0
#                            print('auc with nan value')
                    temp_array.append(auc)
        return tf.stack(temp_array, axis=0, name=main_scope) # Stacks a list of rank-R tensors into one rank-(R+1) tensor.




def get_auc_pr_curve(labels, scores, name, num_thresholds):    
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
        
#            summary_lib.pr_curve_raw_data_op(
#                                name='curve',
#                                true_positive_counts=data.tp,
#                                false_positive_counts=data.fp,
#                                true_negative_counts=data.tn,
#                                false_negative_counts=data.fn,
#                                precision=data.precision,
#                                recall=data.recall,
#                                num_thresholds=num_thresholds,
#                                display_name='Precision-Recall Curve',
#                                description='Predictions must be in the range [0-1]')
#    
#            summary_lib.scalar(
#                                'f1_max',
#                                tf.reduce_max(
#                                    2.0 * data.precision * data.recall / tf.maximum(
#                                        data.precision + data.recall, 1e-7)))
            
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

    
# at running level:  
def log_loss(labels, probs):
    """
    Args:
        labels: Labels tensor, int32 - [batch_size, n_classes], with one-hot
        encoded values.
        logits: Probabilities tensor, float32 - [batch_size, n_classes].
    """    
    total_loss = 0
    for j in range(probs.shape[1]):
        loss = metrics.log_loss(labels[:, j], probs[:, j])
        total_loss += loss

    total_loss /= np.float32(probs.shape[1])
    
    return total_loss

# at graph level:
def log_loss(labels, probs, name):
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
                      

def calculate_metrics(labels, logits):
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

    m_list = get_m_hand(labels, probs, 'metrics/m_measure')
    accuracy = get_accuracy(labels_int, logits, 'metrics/accuracy')    
    auc = get_auc(labels, probs, True, 'metrics/auc')    
    conf_mtx = get_confusion_matrix(labels_int, predictions,
                                    len(classes), 'metrics/confusion')
    loss = log_loss(labels, probs, 'metrics/log_loss')
    pr_auc, pr_data = get_auc_pr_curve(labels, probs, 'metrics/auc_pr', 200)
    
    # this is for the definition of the graph:
    return accuracy, conf_mtx, auc, m_list, loss, pr_auc, pr_data 


def add_hidden_layers(features, architecture, FLAGS, train_flag, act=tf.nn.relu):
    """Add hidden layers to the model using the architecture parameters."""
    hidden_out = features
    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope #JIT compiler compiles and runs parts of TF graphs via XLA, fusing multiple operators (kernel fusion) nto a small number of compiled kernels.
    with jit_scope(): #this operation will be compiled with XLA.
        for hid_i in range(1, FLAGS.n_hidden + 1):
            hidden_out = nn_layer(hidden_out,
                                  architecture['n_hidden_{:1d}'.format(hid_i)],
                                  '{:1d}_hidden'.format(hid_i), FLAGS, act, train_flag)
    return hidden_out


def inference(features, architecture, FLAGS):
    """Build the forward model and return the logits and labels placeholder."""
    train_flag = tf.placeholder(tf.bool, None, name='train_flag')
    with tf.name_scope('input_normalization') as scope:
        feature_norm = features
#        feature_norm = tf.contrib.layers.layer_norm( # 
#            features, center=True, scale=True, scope=scope)
        # feature_norm = tf.layers.batch_normalization(
        #     features,
        #     center=True,  # False,
        #     scale=True,  # False,
        #     training=train_flag,
        #     name='input_normalization/norm')
        variable_summaries('input_normalized', feature_norm, FLAGS.allow_summaries)

    hidden_out = add_hidden_layers(feature_norm, architecture, FLAGS, train_flag)
    # Linear output layer for the logits
    logits = (nn_layer(hidden_out, architecture['n_classes'],'9_softmax_linear', FLAGS, tf.identity, train_flag))    
    return logits


def initialize():
    """Add an Op to the graph to initialize the global and local variables."""
    with tf.name_scope('init') as scope:
        with tf.name_scope('global'):
            global_init = tf.global_variables_initializer()
        with tf.name_scope('local'):
            local_init = tf.local_variables_initializer()
            # print(local_init.name)
        tf.group(global_init, local_init, name=scope)
    return


##########################
# ## GRAPH DEFINITION ## #
##########################
def build_graph(architecture, FLAGS):
    """Build the computation graph for the neural net."""
    print('Building the computation graph in TF....')
    tf.set_random_seed(RANDOM_SEED)
    with tf.Graph().as_default() as comp_graph:
        features = tf.placeholder(
            DT_FLOAT, [None, architecture['n_input']], name='features')
        labels = tf.placeholder(
            DT_FLOAT, [None, architecture['n_classes']], name='targets')
        epoch_flag = tf.placeholder(tf.int32, None, name='epoch_flag')
        example_weights = tf.placeholder(
            DT_FLOAT, [None], name='example_weights')
        logits = inference(features, architecture, FLAGS) #makes all processing from input (features) to output (nn_layer) but with tf.placeholders
        loss = calculate_loss(labels, logits, example_weights, FLAGS.reg_rate)
        # Accuracy is only for reporting purposes, won't be used to train.
        accuracy, conf_mtx, auc_list, m_list, lloss, auc_pr, auc_data = calculate_metrics( # ---, labels_int, predictions, probs, pr_auc
            labels, logits)
        train(loss, FLAGS, epoch_flag)
        with tf.name_scope('0_performance'):
            # Scalar summaries to track the loss and accuracy over time in TB.
            tf.summary.scalar('0accuracy', accuracy)
            tf.summary.scalar('1better_accuracy',
                              tf.reduce_mean(
                                  tf.diag_part(conf_mtx / tf.reduce_sum(
                                      conf_mtx, axis=1, keepdims=True))))
            tf.summary.scalar('2auc_aoc', tf.reduce_mean(auc_list))
            tf.summary.scalar('3m_measure', tf.reduce_mean(m_list))
            tf.summary.scalar('4loss', loss)
            tf.summary.scalar('5log_loss', lloss)
            tf.summary.scalar('6auc_pr', tf.reduce_mean(auc_pr))
        initialize()
        # print(ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES))
        # FLAGS.reset_op = [
        #     tf.variables_initializer(
        #         ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES))
        # ]
        # print((FLAGS.reset_op[0]).name)
        # for xyxx in ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES):
        #     print(xyxx)
        #     print(xyxx.initializer)

        # Create a scalar tensor of string type merging all the summaries.
        # The name of this variable is 'Merge/MergeSummary:0'
        merged = tf.summary.merge_all()
        assert merged.name == 'Merge/MergeSummary:0'
    print('Graph building completed....')
    return comp_graph


###############################
# ## TRAINING & EVALUATION ## #
###############################
def run_model(comp_graph, name, net_number, FLAGS, DATA):
    """Run the model represented by the input computation graph."""
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True) # tf.ConfigProto()
    # Turns on XLA JIT compilation if the XLA flag is on.
    jit_level = tf.OptimizerOptions.ON_1 if FLAGS.xla else 0  # pylint: disable=no-member  # tf.OptimizerOptions.ON_1: IT compilation is turned on at the session level
    config.graph_options.optimizer_options.global_jit_level = jit_level  # pylint: disable=no-member
    config.gpu_options.allow_growth = True
    return_epoch = 0
    with tf.Session(graph=comp_graph, config=config) as sess:
        writers = {
            'batch': tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'batch'),
                                           sess.graph),
            'train': tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'train'), graph=None),
            'valid': tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'valid'), graph=None)
        }
        try:
            start_time = datetime.now()
            return_epoch = batch_training(sess, writers, name, net_number, FLAGS, DATA)
            end_time = datetime.now() - start_time
            print('Total Training Time for: ' + FLAGS.name, end_time)
        except Exception as inst:            
            print('Exception on run_model: ', inst)
        finally:
            print('Finally on run_model...')            
            for mode in writers:
                writers[mode].close()
            if FLAGS.test_flag:
                end_time = datetime.now() - start_time                                
                test_metrics = batching_dataset(sess, writers, 'test', DATA, FLAGS)
                bett_acc_test, m_mtx_mean_test, auc_aoc_mean_test, auc_pr_mean_test = print_stats(test_metrics, 'test', FLAGS.log_file)                
                test_file = Path(os.path.join(FLAGS.logdir, name + "_test_history.csv"))
                if test_file.exists():
                    dtype=None
                else: 
                    dtype = ['NN_name', 'NN_Number','Total Epochs', 'Execute Epochs', 'Total Training Time', 'Loss','LogLoss','Accuracy','Better-Accuracy','M-Measure Mean','AUC_AOC Mean','AUC_PR Mean']
                
                pd.DataFrame(data=[(name, net_number, FLAGS.epoch_num, return_epoch, end_time,test_metrics[4], test_metrics[5], test_metrics[1], bett_acc_test, m_mtx_mean_test, auc_aoc_mean_test, auc_pr_mean_test)], 
                             columns=dtype, index=None).to_csv(test_file, index=False, mode='a', sep =';')   
                
                # DATA.train.__exit__(dtype(inst), inst, inst.__traceback__)
                if (not FLAGS.log_file == None): FLAGS.log_file.close()
                DATA.train.__exit__(None, None, None)
                DATA.validation.__exit__(None, None, None)
                DATA.test.__exit__(None, None, None)



def acc_metrics_init(DATA):    
    acc_conf_mtx=np.zeros((DATA.train.num_classes, DATA.train.num_classes))
    acc_acc = 0
    acc_auc_list = np.zeros((DATA.train.num_classes))
    acc_m_mtx = np.zeros((DATA.train.num_classes, DATA.train.num_classes))
    acc_loss = 0
    acc_log_loss = 0
    acc_auc_pr_list = np.zeros((DATA.train.num_classes))
    epoch_metrics = (acc_conf_mtx, acc_acc, acc_auc_list, acc_m_mtx, acc_loss, acc_log_loss, acc_auc_pr_list)
    
    return epoch_metrics

def batching_dataset(sess, writers, tag, DATA, FLAGS):

    if tag =='valid':
        batch_num = DATA.validation.total_num_batch(FLAGS.valid_batch_size) # math.ceil(np.float32(DATA.validation.num_examples / FLAGS.batch_size))
    else: #test
        batch_num = DATA.test.total_num_batch(FLAGS.test_batch_size) # math.ceil(np.float32(DATA.test.num_examples / FLAGS.batch_size))
        
    metrics = acc_metrics_init(DATA)
    start_time = datetime.now()
    for batch_i in range(batch_num):
        feed = create_feed_dict(tag, DATA, FLAGS)     
        batch_metrics = get_metrics(sess, feed)        
        metrics = batch_stats(metrics, batch_metrics) 
        if tag == 'valid': write_summaries(sess, writers, tag, batch_i, feed)
    
    metrics = (metrics[0], metrics[1]/batch_num, metrics[2]/batch_num, metrics[3]/batch_num, 
               metrics[4]/batch_num, metrics[5]/batch_num, metrics[6]/batch_num)
        
    print( tag + ' Time: ', datetime.now() - start_time)
    return metrics
            

def batch_training(sess, writers, name, net_number, FLAGS, DATA):
    """Iterate over the dataset based on the number of epochs and train."""

    def train_one_epoch(epoch, batch_size, batch_num):
        
        epoch_metrics_train = acc_metrics_init(DATA)        
        try:
            print("Epoch: ", epoch)		
            FLAGS.log_file.write('Epoch Number:  %d\r\n' % epoch) 
            for batch_i in range(batch_num):
                batch_time = datetime.now()
                print("batch Number: ", batch_i)
                FLAGS.log_file.write('batch Number:  %d\r\n' % batch_i)            
                batch_dict= create_feed_dict('batch', DATA, FLAGS)
                step = epoch * batch_num + batch_i    # total steps for all epochs        
                if step > 0 and (step * batch_size) % (DATA.train.total_num_examples) < batch_size:                
                    # print ('(step * batch_size) % (DATA.train.num_examples): ', (step * batch_size) % (DATA.train.num_examples))                    
                    train_and_summarize(sess, writers, step, batch_dict)# for saving space on disk.        
                else:                
                    _ = sess.run(['train'], feed_dict=batch_dict)                
                    
                batch_metrics_train = get_metrics(sess, batch_dict)        
                epoch_metrics_train = batch_stats(epoch_metrics_train, batch_metrics_train)             
                batch_time = datetime.now() - batch_time
                print('Batch Time: ', batch_time) 
                FLAGS.log_file.write('Batch Time:  %s\r\n' % str(batch_time))
            print("Calculating Epoch Metrics")
        except Exception as e:
            print('Exception on train_one_epoch', e)
            raise ValueError(e)
        finally:              
            epoch_metrics = (epoch_metrics_train[0], epoch_metrics_train[1]/batch_num, epoch_metrics_train[2]/batch_num, epoch_metrics_train[3]/batch_num, 
                             epoch_metrics_train[4]/batch_num, epoch_metrics_train[5]/batch_num, epoch_metrics_train[6]/batch_num)            
            print(epoch_metrics)
            FLAGS.log_file.write('epoch_metrics:  %s\r\n' % str(epoch_metrics))                                    
            
        return epoch_metrics 


    return_epoch = 0
    try:
        # # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(var_list=None, max_to_keep=1)
        # # Initialize all the variables in the graph.
        print('before init tf variables: ', psutil.virtual_memory()) #  physical memory usage
        FLAGS.log_file.write('RAM before init tf variables: %s\r\n' % str(psutil.virtual_memory()))
        sess.run('init')
        print('after init tf variables: ', psutil.virtual_memory()) #  physical memory usage
        FLAGS.log_file.write('RAM after init tf variables: %s\r\n' % str(psutil.virtual_memory()))
        print('Initialized all the local and global variables in the graph....')
        
        best_loss = -1
        best_epoch = 0 
        # best_weights = None
        train_history =[]
        valid_history=[]            
        dtype = ['epoch','epoch_time','Loss','LogLoss','Accuracy','Better-Accuracy','M-Measure Mean','AUC_AOC Mean','AUC_PR Mean']
        checkpoint_file = os.path.join(FLAGS.logdir, name[:-4] + '_' + FLAGS.name + '_' + str(net_number)) #'/model.ckpt' # os.path.join(FLAGS.logdir, 'model.ckpt')

        if FLAGS.max_epoch_size==-1:
            batch_num = math.ceil(np.float32(DATA.train.total_num_examples / FLAGS.batch_size))        
        elif (FLAGS.max_epoch_size > DATA.train.total_num_examples):
            raise ValueError('ERROR: The max_epoch_size must not be greater than the train.total_num_examples')
        else:
            batch_num = math.ceil(np.float32( FLAGS.max_epoch_size / FLAGS.batch_size))
        
        for epoch in range(FLAGS.epoch_num):
#            try:
            FLAGS.epoch_flag = epoch
            epoch_time = datetime.now()
            epoch_metrics = train_one_epoch(epoch, FLAGS.batch_size, batch_num)
            epoch_time = datetime.now() - epoch_time
            FLAGS.log_file.write('Epoch Time:  %s\r\n' % str(epoch_time))
            bett_acc_train, m_mtx_mean_train, auc_aoc_mean_train, auc_pr_mean_train = print_stats(epoch_metrics, 'train', FLAGS.log_file)        
            # Validation set: 
            valid_time = datetime.now()                               
            valid_metrics = batching_dataset(sess, writers, 'valid', DATA, FLAGS)
            valid_time = datetime.now() - valid_time
            FLAGS.log_file.write('Valid Time:  %s\r\n' % str(valid_time))
            bett_acc_valid, m_mtx_mean_valid, auc_aoc_mean_valid, auc_pr_mean_valid = print_stats(valid_metrics, 'valid', FLAGS.log_file)
            # model.set_weights(best_weights) ?? How to do that with tensorflow??
            # Do __not__ delete the following 2 lines; they periodically save the model.                
            saver.save(sess, checkpoint_file) # global_step=epoch
            train_history += [(epoch, epoch_time, epoch_metrics[4], epoch_metrics[5], epoch_metrics[1], bett_acc_train, m_mtx_mean_train, auc_aoc_mean_train, auc_pr_mean_train)]
            valid_history += [(epoch, valid_time, valid_metrics[4], valid_metrics[5], valid_metrics[1], bett_acc_valid, m_mtx_mean_valid, auc_aoc_mean_valid, auc_pr_mean_valid)]
            
            # Early Stopping:
            if valid_metrics[4] < best_loss or best_loss == -1:
                best_loss = valid_metrics[4]
                # only_weights = [layer for layer in tf.trainable_variables() if layer.op.name.find('weights')>0 ]            
                # if weights: best_weights =  weights[0].eval()  
                # model.set_weights(best_weights) ?? How to do that with tensorflow??
                # Do __not__ delete the following 2 lines; they periodically save the model.                
                # saver.save(sess, checkpoint_file) # global_step=epoch
                best_epoch = epoch
            else:
                if epoch - best_epoch == 5:
                    print('Stopping: Not Improving in Validation Set after 5 epochs')
                    FLAGS.log_file.write('Stopping: Not Improving in Validation Set after 5 epochs  %d\r\n' % best_epoch)
                    return_epoch = epoch
                    break       
                
            if valid_metrics[4] <= FLAGS.loss_tolerance:
                print('Stopping: The loss tolerance was reached: ', FLAGS.loss_tolerance)
                FLAGS.log_file.write('Stopping: The loss tolerance (%f) was reached \r\n' % FLAGS.loss_tolerance)
                return_epoch = epoch
                break
            print('memory usage per epoch: ', psutil.virtual_memory()) #  physical memory usage
            FLAGS.log_file.write('RAM memory usage per epoch: %s\r\n' % str(psutil.virtual_memory()))
#            finally:
#                pd.DataFrame(data=train_history, columns=dtype, index=None).to_csv(os.path.join(FLAGS.logdir, name[:-4] + '_' + FLAGS.name + '_' + str(net_number) +"_train_history.csv"), index=False)
#                pd.DataFrame(data=valid_history, columns=dtype, index=None).to_csv(os.path.join(FLAGS.logdir, name[:-4] + '_' + FLAGS.name + '_' + str(net_number) +"_valid_history.csv"), index=False)                        
    except Exception as inst:                    
        print('Exception on batch_training', inst)
        raise ValueError(inst)        

    finally:
        print('batch_training finally instance!')
        if return_epoch == 0: return_epoch = FLAGS.epoch_num    
        pd.DataFrame(data=train_history, columns=dtype, index=None).to_csv(os.path.join(FLAGS.logdir, name[:-4] + '_' + FLAGS.name + '_' + str(net_number) +"_train_history.csv"), index=False, sep =';')
        pd.DataFrame(data=valid_history, columns=dtype, index=None).to_csv(os.path.join(FLAGS.logdir, name[:-4] + '_' + FLAGS.name + '_' + str(net_number) +"_valid_history.csv"), index=False, sep =';')
        
    return return_epoch


def reshape_m_mtx(mtx):
    """Reshape the python list into a np array."""
    new_mtx = [0]
    for i in range(6):
        new_mtx.extend(mtx[i * 7:(i + 1) * 7])
        new_mtx.append(0)
    temp = np.array(new_mtx).reshape(7, 7)
#    temp = .5 * (temp + temp.T)
    # # temp = np.triu(temp)
    return temp


def print_stats(stats, name, log_file):
    """Print the given stats."""
        
    conf_mtx, acc, auc_list, m_mtx_list, loss, log_loss, auc_pr_list = stats          
               
    # conf_mtx1 = conf_mtx / conf_mtx.sum(axis=1, keepdims=True)       
    cfsum = conf_mtx.sum(axis=1, keepdims=True)
    conf_mtx1 = np.divide(conf_mtx, cfsum, out=np.zeros_like(conf_mtx), where=cfsum!=0)
    bett_acc = conf_mtx1.diagonal().mean()
    auc_aoc_mean = auc_list.mean()
    auc_pr_mean = auc_pr_list.mean()
    # global_acc = 0.
    m_mtx_mean = 0.
    
#    if (name!='train'):
    m_mtx = m_mtx_list # reshape_m_mtx(m_mtx_list)
    m_mtx_mean = np.float32(m_mtx.sum()) / (49 - 7)
    stdout = 'Avg Cost in ' + name +': {:.5f}\n'.format(loss)        
    stdout = stdout + ' Avg Log_Cost in ' + name +': {:.5f}\n'.format(log_loss)
    stdout = stdout +  '{:s}:'.format(name) + ' (Silly) Global-ACC={:.4f}, Better ACC={:.4f},'.format(acc, bett_acc) + \
        ' Avg M-Measure={:.4f},'.format(m_mtx_mean) + \
        ' Avg AUC_AOC={:.4f}'.format(auc_aoc_mean) + ' Avg AUC_PR={:.4f}\n'.format(auc_pr_mean)
    stdout = stdout + ('\t' * 6).join(['Total Confusion Matrix', 'Total M-Measure Matrix', 'Total AUC_AOC', 'Total AUC_PR\n'])
    for conf_row, row, auc, auc_pr in zip(conf_mtx, m_mtx, auc_list, auc_pr_list):
        for conf_value in conf_row:
            stdout = stdout + '{:.4f}'.format(conf_value) + '\t'
        stdout = stdout + '|\t'
        for value in row:
            stdout = stdout + '{:.4f}'.format(value) + '\t'
        stdout = stdout + '| \t{:.4f}'.format(auc) + ' | \t{:.4f}'.format(auc_pr) + '\n'
    stdout = stdout + '---------------------------------------------------------------------'
          
    print(stdout)
    log_file.write('METRICS:  %s\r\n' % stdout)
    
    return bett_acc, m_mtx_mean, auc_aoc_mean, auc_pr_mean
    

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

def reset_and_update(sess, feed_dict):
    """Reset the local variables and update the necessary update ops."""
    sess.run('init/local/init')
    
#    update_names_list = [
#        'metrics/m_measure/' + str(i) + str(j) + '/hist_accumulate/update_op'
#        for i in range(7) for j in range(7) if i != j
#    ]
    
    update_names_list = [
        'metrics/auc/{:d}/hist_accumulate/update_op'.format(i)
        for i in range(7)
    ]
#    update_names_list.extend([
#        'metrics/auc_pr/{:d}/hist_accumulate/update_op'.format(i)
#        for i in range(7)
#    ])
    update_names_list.extend([
        'metrics/m_measure/' + str(i) + str(j) + '/hist_accumulate/update_op'
        for i in range(7) for j in range(7) if i != j
    ])

    sess.run(update_names_list, feed_dict=feed_dict)
    return


def get_metrics(sess, feed_dict):
    """Get the accuracy over the dataset corresponding to the given mode."""
    # feed_dict = create_feed_dict(mode, DATA, FLAGS)
    reset_and_update(sess, feed_dict)
    
    model_metrics = sess.run(
        [
            'metrics/confusion/SparseTensorDenseAdd:0',
            'metrics/accuracy:0', 
            'metrics/auc:0', 
            'metrics/m_measure:0',
            'loss:0',
            'metrics/log_loss:0',
            'metrics/auc_pr:0',            
        ],    
        feed_dict=feed_dict)    
    #pmetrics = tf.Print(metrics, [metrics], message='Metrics: ')
    # print(pmetrics.eval(Session=sess))
    # print('get_metrics: SparseTensorDenseAdd, accuracy, auc_roc, m_measure, loss, log_loss, auc_pr: ', model_metrics)
    # output = sess.run('input_normalization/9_softmax_linear', feed_dict=feed_dict)
    return model_metrics


def train_and_summarize(sess, writers, step, feed_dict):
    """Train and record execution metadata for use in TB."""
    batch_writer = writers['batch']
    summary, _ = sess.run(
        ['Merge/MergeSummary:0', 'train'], feed_dict=feed_dict)
    # Do __not__ delete the following lines. I've disabled these just to make
    # the program faster; however, these would write metadata to TB.

    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  # pylint: disable=no-member
    # run_metadata = tf.RunMetadata()
    # summary, _ = sess.run(
    #     ['Merge/MergeSummary:0', 'train'],
    #     feed_dict=create_feed_dict('batch'),
    #     options=run_options,
    #     run_metadata=run_metadata)
    # batch_writer.add_run_metadata(run_metadata, 'step{:06d}'.format(step))
    # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    # with open('timeline.ctf.json', 'w') as trace_file:
    #     trace_file.write(trace.generate_chrome_trace_format())

    batch_writer.add_summary(summary, step)
    batch_writer.flush()
    return


def write_summaries(sess, writers, mode, step, feed_dict):
    """Get the summary protobuf and write it to the Summary.FileWriter."""
    writer = writers[mode]    
    reset_and_update(sess, feed_dict)
    summary = sess.run('Merge/MergeSummary:0', feed_dict=feed_dict)
    writer.add_summary(summary, step)
    writer.flush()
    return


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
        'epoch_flag:0': FLAGS.epoch_flag,
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
    
    return feed_d

def retrieve_tf_model(name, net_number, FLAGS):
    checkpoint_file = os.path.join(Path(FLAGS.logdir), name + '_' + str(net_number) +'.meta')
    print(checkpoint_file)
    # with tf.Session() as sess:    
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(checkpoint_file)
    new_saver.restore(sess, tf.train.latest_checkpoint(Path(FLAGS.logdir)))

##########################
# ## SETTINGS
##########################
    
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
    FLAGS.decay_step = 800 # 1 * 80000 #according to paper: 800 epochs
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
    
    FLAGS.valid_batch_size = 100000
    FLAGS.test_batch_size = 100000
    
    if FLAGS.n_hidden < 0 : raise ValueError('The size of hidden layer must be at least 0')
    if (FLAGS.n_hidden > 0) and (FLAGS.n_hidden != len(FLAGS.s_hidden)) : raise ValueError('Sizes in hidden layers should match!')
    
    if (net_number==0):
        FLAGS.name ='default_settings'        
    elif (net_number==1):
        FLAGS.name ='batch_layer_type'
        FLAGS.batch_layer_type = 'batch'        
    
    FLAGS.log_file = open(os.path.join(FLAGS.logdir, FLAGS.name + '_' + str(net_number) +"_log.txt"), 'w+', 1)
    return FLAGS

def architecture_settings(DATA, FLAGS):
    # Architecture	
    architecture = {
        'n_input': DATA.train.num_columns,
        'n_classes': DATA.validation.num_classes
    }
    for hid_i in range(1, FLAGS.n_hidden+1):
        architecture['n_hidden_{:1d}'.format(hid_i)] = FLAGS.s_hidden[hid_i-1]
    print('architecture', architecture)    
    return architecture

def main(_):
    print("Run the main program.")

    FLAGS, UNPARSED = update_parser(argparse.ArgumentParser())    
    print("UNPARSED", UNPARSED)    
    
    if tf.gfile.Exists(FLAGS.logdir):
       tf.gfile.DeleteRecursively(FLAGS.logdir)
    tf.gfile.MakeDirs(FLAGS.logdir)    

    conf_number = 1    
    #FLAGS.train_dir = 'train_set_1millx30mill' #'train_set_1millx30mill' # 'chuncks_random_c1mill_train'#'chunks_all_c1millx3' # 'train_set_1millx30mill' # 'c100th_train_set' # 'train_set_800th'
    #FLAGS.valid_dir = 'c100th_valid_set' #'c100th_valid_set' # 'chuncks_random_c1mill_valid' # chunks_all_800th 'valid_set_800th'
    #FLAGS.test_dir = 'c100th_test_set' #'c100th_test_set' # 'chuncks_random_c1mill_test' # 'test_set_800th'
    # training_dict = md.get_dataset_metadata(train_dir)
    DATA = md.get_h5_data(FLAGS.train_dir, FLAGS.valid_dir, FLAGS.test_dir) 
    
#    for file_path in glob.glob(os.path.join(PRO_DIR, subdir_name,"*.h5")):  
#        file_name = os.path.basename(file_path)
#        print('Dataset: ' + file_name)    
    
    for i in range(conf_number):                        
        FLAGS = FLAGS_setting(FLAGS, i)
        print("FLAGS", FLAGS)
        FLAGS.log_file.write('METRICS:  %s\r\n' % str(FLAGS))
        FLAGS.log_file.write('training files:  %s\r\n' % str(DATA.train._dict))
        FLAGS.log_file.write('validation files:  %s\r\n' % str(DATA.validation._dict))
        FLAGS.log_file.write('testing files:  %s\r\n' % str(DATA.test._dict))        
        architecture = architecture_settings(DATA, FLAGS)
        print('RAM before build: ', psutil.virtual_memory()) #  physical memory usage
        FLAGS.log_file.write('RAM  before build: %s\r\n' % str(psutil.virtual_memory()))
        graph = build_graph(architecture, FLAGS)        
        print('RAM after build', psutil.virtual_memory()) #  physical memory usage
        FLAGS.log_file.write('RAM  after build: %s\r\n' % str(psutil.virtual_memory()))
        run_model(graph, 'testing_data', i,  FLAGS, DATA)      
    #FLAGS = FLAGS_setting(FLAGS, 1)
    #graph = build_graph(architecture, FLAGS)        
    #run_model(graph, 'Data1-100_' + FLAGS.name, 1,  FLAGS, DATA)    


def update_parser(parser):
    """Parse the arguments from the CLI and update the parser."""
    parser.add_argument(
        '--epoch_num',
        type=int,
        default=15,
        help='Number of epochs to run trainer on the dataset.')
    parser.add_argument(
        '--dropout_keep',
        type=float,
        default=0.5,
        help='Keep probability for training dropout.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='Initial learning rate')
    parser.add_argument(
        '--xla', type=bool, default=True, help='Turn xla via JIT on')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/input_data',
        help='Directory for storing input data')
    parser.add_argument(
        '--logdir',
        type=str,
        default=os.path.join(Path.home(), 'real_summaries'),
        help='Summaries log directory')
    parser.add_argument(
        '--n_hidden',
        type=int,
        default=3,
        help='Number of hidden layers.')
    parser.add_argument(
        '--s_hidden',
        type=int,
        nargs='*',
        default=[200, 140, 140],
        help='Size of each hidden layer.')
    parser.add_argument(
        '--stratified_flag',
        type=bool,
        default=False,
        help='Execute Stratified Sampling of the DataSet by Delinquency Status')
    parser.add_argument(
        '--batch_type',
        type=str,
        default='layer',
        help='Select the layer type for batch normalization')
    parser.add_argument(
        '--name',
        type=str,
        default='',
        help='Hyperparameters configuration')
    parser.add_argument(
        '--allow_summaries',
        type=bool,
        default=True,
        help='Recording Summaries by variable')
    parser.add_argument(
        '--max_epoch_size',
        type=int,
        default=-1,
        help='How many observations will be passed per epoch. default value=-1 (all training set)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4425,
        help='How many observations will be passed per batch')
    parser.add_argument(
        '--log_file',
        type=IOBase,
        default=None,
        help='file path for the log trace')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='chuncks_random_c1mill_train',
        help='Training directory inside data/processed/')
    parser.add_argument(
        '--valid_dir',
        type=str,
        default='chuncks_random_c1mill_valid',
        help='Validation directory inside data/processed/')
    parser.add_argument(
        '--test_dir',
        type=str,
        default='chuncks_random_c1mill_test',
        help='Testing directory inside data/processed/')
    parser.add_argument(
        '--loss_tolerance',
        type=float,
        default=0.0001,
        help='stop criteria')     
    return parser.parse_known_args()

# %%
if __name__ == '__main__':    
    tf.app.run(main=main, argv=[sys.argv[0]])

# Very useful blog on imbalanced data:
# http://www.kdnuggets.com/2016/08/learning-from-imbalanced-classes.html

