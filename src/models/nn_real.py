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
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline
from tensorflow.python.framework import ops
from sklearn import metrics

# import mort_data


RANDOM_SEED = 123  # Set the seed to get reproducable results.
DT_FLOAT = tf.float32
NP_FLOAT = np.dtype('float32')

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


def variable_summaries(name, var):
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
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('calculate_std'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.histogram('histogram', var)
    return


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
    with tf.device('/cpu:0'): # this operation is assigned to this device, but this make a copy of data when is transferred on and off the device, which is expensive.
        var = tf.get_variable(
            name,
            shape,
            initializer=initializer,
            regularizer=regularizer,
            dtype=dtype)
    return var


def _create_variable(name,
                     shape,
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
    variable_summaries(name + '/summaries', var)
    return var


def create_weights(name, shape):
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

    regularizer = tf.contrib.layers.l1_regularizer(
        float(FLAGS.reg_rate), 'penalty')
    return _create_variable(name, shape, kernel_initializer, regularizer,
                            dtype)


def bias_variable(name, shape, layer_name):
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
        if (not FLAGS.weighted_sampling) and 'soft' in layer_name:
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


def dropout_layer(name, tensor_before):
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
        variable_summaries('input_dropped_out', dropped)
    return dropped


def batch_normalization(name, input_tensor):
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
    train_flag = tf.get_default_graph().get_tensor_by_name('train_flag:0')
    with tf.name_scope(name):
        normalized = tf.layers.batch_normalization(
            input_tensor,
            center=True,
            scale=True,
            training=train_flag,
            name=name)  # renorm=True, renorm_momentum=0.99)
        variable_summaries('normalized_batch', normalized)
    return normalized


def layer_normalization(name, input_tensor):
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
        variable_summaries('normalized_layer', normalized)
    return normalized


def normalize(name, input_tensor, norm_type):
    """Perform either type (batch/layer) of normalization."""
    if not FLAGS.batch_norm:
        return input_tensor
    if norm_type.lower() == 'batch':
        return batch_normalization(name, input_tensor)
    if norm_type.lower() == 'layer':
        return layer_normalization(name, input_tensor)
    raise ValueError('Invalid value for norm_type: ' + norm_type)


def nn_layer(input_tensor, output_dim, layer_name, act):
    """Create a simple neural net layer.

    It performs the affine transformation and uses the activation function to
    nonlinearize. It further sets up name scoping so that the resultant graph
    is easy to read, and adds a number of summary ops.
    """
    input_dim = input_tensor.shape[1].value    
    with tf.variable_scope(layer_name): # A context manager for defining ops that creates variables (layers).
        weights = create_weights('weights', [input_dim, output_dim])
        # This is outdated and no longer applies: Do not change the order of
        # batch normalization and drop out. batch # normalization has to stay
        # __before__ the drop out layer.
        variable_summaries('input', input_tensor)
        input_tensor = dropout_layer('dropout', input_tensor)
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
        preactivate = normalize('layer_normalization', mixed,
                                'layer')  # + biases
        # tf.summary.histogram('pre_activations', preactivate)
        # preactivate = dropout_layer('dropout', preactivate)
        with tf.name_scope('activation') as scope:
            activations = act(preactivate, name=scope)
            tf.summary.histogram('activations', activations)
    return activations


def calculate_loss(labels, logits, weights):
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
        with tf.name_scope('regularization'):
            penalty = tf.losses.get_regularization_loss(name='penalty')
            tf.summary.scalar('weight_norm', penalty / (1e-8 + FLAGS.reg_rate)) #for printing out
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
            tf.summary.scalar('weighted_cross_entropy', weighted_cross_entropy)

            # cross_entropy = tf.losses.softmax_cross_entropy(
            #     onehot_labels=labels,
            #     logits=logits,
            #     weights=1.0,
            #     loss_collection=None)
            # tf.summary.scalar('cross_entropy', cross_entropy)
        return tf.add(weighted_cross_entropy, penalty, name=scope) # Returns x + y element-wise.


def train(loss, learning_rate):
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

    def get_learning_rate(initial_learning_rate, global_step):
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
                global_step,
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
            # Create a variable to track the global step.
            global_step = tf.get_variable(
                'train/global_step',
                shape=[],
                initializer=tf.constant_initializer(0, dtype=tf.int32),
                trainable=False)
            final_learning_rate = get_learning_rate(learning_rate, global_step)

            # optimizer = tf.train.GradientDescentOptimizer(final_learning_rate)
            # optimizer = tf.train.MomentumOptimizer(
            #     final_learning_rate, FLAGS.momentum, use_nesterov=True)
            optimizer = tf.train.AdagradOptimizer(final_learning_rate)

            # Use the optimizer to apply the gradients that minimize the loss
            # (and increment the global step counter) as a single training step.
            return optimizer.minimize(
                loss, global_step=global_step, name=scope)


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
                auc_func(labels, scores, class_, str(ind))
                for ind, class_ in enumerate(classes) # pair (index ej. 0, value ej. '0')
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
                    temp_array.append(auc)
        return tf.stack(temp_array, axis=0, name=main_scope) # Stacks a list of rank-R tensors into one rank-(R+1) tensor.




def get_auc_pr_curve(labels, scores, name, num_thresholds):    
    with tf.name_scope(name) as scope:                             
        AUC_PR = []
        for i in range(7):  
            data, update_op = tf.contrib.metrics.precision_recall_at_equal_thresholds(
                            name='pr_data',
                            predictions=scores[:, i],
                            labels=tf.cast(labels[:, i], tf.bool),
                            num_thresholds=num_thresholds)
            ops.add_to_collections(ops.GraphKeys.UPDATE_OPS, update_op)
        
            summary_lib.pr_curve_raw_data_op(
                                name='curve',
                                true_positive_counts=data.tp,
                                false_positive_counts=data.fp,
                                true_negative_counts=data.tn,
                                false_negative_counts=data.fn,
                                precision=data.precision,
                                recall=data.recall,
                                num_thresholds=num_thresholds,
                                display_name='Precision-Recall Curve',
                                description='Predictions must be in the range [0-1]')
    
            summary_lib.scalar(
                                'f1_max',
                                tf.reduce_max(
                                    2.0 * data.precision * data.recall / tf.maximum(
                                        data.precision + data.recall, 1e-7)))
            
            # AUC_PR.append(metrics.auc(tf.stack(data.recall), tf.stack(data.precision)))   # we cant use sklearn with tensorflow definition!
            auc, update_op = tf.metrics.auc(scores[:, i], labels[:, i], num_thresholds=num_thresholds, curve='PR')
            ops.add_to_collections(ops.GraphKeys.UPDATE_OPS, update_op)
            AUC_PR.append(auc)
            
        return tf.stack( # Pack along first dim
            AUC_PR,
            axis=0,
            name=scope)
    
    

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
        # print ('labels_int: ', labels_int) # labels_int:  Tensor("metrics/intlabels:0", shape=(?,), dtype=int64)
        # print('predictions: ', predictions)
        probs = tf.nn.softmax(logits, name='probs') # Computes softmax activations. softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
        # print('probs: ', probs) # probs:  Tensor("metrics/probs:0", shape=(?, 7), dtype=float32)

    m_list = get_m_hand(labels, probs, 'metrics/m_measure')
    accuracy = get_accuracy(labels_int, logits, 'metrics/accuracy')    
    auc = get_auc(labels, probs, True, 'metrics/auc')    
    conf_mtx = get_confusion_matrix(labels_int, predictions,
                                    len(classes), 'metrics/confusion')
    # --- pr_auc = get_auc_pr_curve(labels, probs, 'metrics/pr_curve', 200)
    
    # this is for the definition of the graph:
    return accuracy, conf_mtx, auc, m_list # ---, labels_int, predictions, probs, pr_auc


def add_hidden_layers(features, architecture, act=tf.nn.relu):
    """Add hidden layers to the model using the architecture parameters."""
    hidden_out = features
    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope #JIT compiler compiles and runs parts of TF graphs via XLA, fusing multiple operators (kernel fusion) nto a small number of compiled kernels.
    with jit_scope(): #this operation will be compiled with XLA.
        for hid_i in range(1, FLAGS.n_hidden + 1):
            hidden_out = nn_layer(hidden_out,
                                  architecture['n_hidden_{:1d}'.format(hid_i)],
                                  '{:1d}_hidden'.format(hid_i), act)
    return hidden_out


def inference(features, architecture):
    """Build the forward model and return the logits and labels placeholder."""
    train_flag = tf.placeholder(tf.bool, None, name='train_flag')
    with tf.name_scope('input_normalization') as scope:
        # feature_norm = features
        feature_norm = tf.contrib.layers.layer_norm( # 
            features, center=True, scale=True, scope=scope)
        # feature_norm = tf.layers.batch_normalization(
        #     features,
        #     center=True,  # False,
        #     scale=True,  # False,
        #     training=train_flag,
        #     name='input_normalization/norm')
        variable_summaries('input_normalized', feature_norm)

    hidden_out = add_hidden_layers(feature_norm, architecture)
    # Linear output layer for the logits
    logits = (nn_layer(hidden_out, architecture['n_classes'],
                       '9_softmax_linear', tf.identity))
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
def build_graph(architecture, learning_rate):
    """Build the computation graph for the neural net."""
    print('Building the computation graph in TF....')
    tf.set_random_seed(RANDOM_SEED)
    with tf.Graph().as_default() as comp_graph:
        features = tf.placeholder(
            DT_FLOAT, [None, architecture['n_input']], name='features')
        labels = tf.placeholder(
            DT_FLOAT, [None, architecture['n_classes']], name='targets')
        example_weights = tf.placeholder(
            DT_FLOAT, [None], name='example_weights')
        logits = inference(features, architecture) #makes all processing from input (features) to output (nn_layer) but with tf.placeholders
        loss = calculate_loss(labels, logits, example_weights)
        # Accuracy is only for reporting purposes, won't be used to train.
        accuracy, conf_mtx, auc_list, m_list = calculate_metrics( # ---, labels_int, predictions, probs, pr_auc
            labels, logits)
        train(loss, learning_rate)
        with tf.name_scope('0_performance'):
            # Scalar summaries to track the loss and accuracy over time in TB.
            tf.summary.scalar('0accuracy', accuracy)
            tf.summary.scalar('1better_accuracy',
                              tf.reduce_mean(
                                  tf.diag_part(conf_mtx / tf.reduce_sum(
                                      conf_mtx, axis=1, keepdims=True))))
            tf.summary.scalar('2auc', tf.reduce_mean(auc_list))
            tf.summary.scalar('3m_measure', tf.reduce_mean(m_list))
            tf.summary.scalar('4loss', loss)
            # tf.summary.scalar('5pr-auc', tf.reduce_mean(pr_auc))
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
def run_model(comp_graph):
    """Run the model represented by the input computation graph."""
    config = tf.ConfigProto() # is a configuring class for the graph.
    # Turns on XLA JIT compilation if the XLA flag is on.
    jit_level = tf.OptimizerOptions.ON_1 if FLAGS.xla else 0  # pylint: disable=no-member  # tf.OptimizerOptions.ON_1: IT compilation is turned on at the session level
    config.graph_options.optimizer_options.global_jit_level = jit_level  # pylint: disable=no-member
    with tf.Session(graph=comp_graph, config=config) as sess:
        writers = {
            'batch': tf.summary.FileWriter(FLAGS.logdir + '/batch',
                                           sess.graph),
            'train': tf.summary.FileWriter(
                FLAGS.logdir + '/train', graph=None),
            'valid': tf.summary.FileWriter(
                FLAGS.logdir + '/valid', graph=None)
        }
        try:
            batch_training(sess, writers)
        finally:
            for mode in writers:
                writers[mode].close()
            if FLAGS.test_flag:
                print_stats(get_metrics(sess, 'test'), 'test')
    return


def batch_training(sess, writers):
    """Iterate over the dataset based on the number of epochs and train."""

    def train_one_epoch(epoch, batch_size):
        print("Complete one epoch of the training.")		
        batch_num = DATA.train.num_examples // batch_size        
        print('batch_num:', batch_num)
        avg_cost = 0
        # keep track of the number of times batch_cost is not calculated
        excluded = 0
        for batch_i in range(batch_num):
            step = epoch * batch_num + batch_i
            # print ('step: ', step) # it depends on the epoch, it is continuos values steps along the epochs.
            # print ('(step * batch_size) % (DATA.train.num_examples): ', (step * batch_size) % (DATA.train.num_examples))
            if step > 0 and (step * batch_size) % (DATA.train.num_examples) < batch_size:
                # Train and record execution stats
                print ('(step * batch_size) % (DATA.train.num_examples): ', (step * batch_size) % (DATA.train.num_examples))
                train_and_summarize(sess, writers, step)
                excluded += 1
                # print(('Added run metadata for epoch {:03d}, '
                #        'batch {:03d}').format(epoch, batch_i))
            else:
                # print('else step: ', step)
                batch_cost, _ = sess.run( # Runs operations and evaluates tensors in fetches.
                    ['loss:0', 'train'], feed_dict=create_feed_dict('batch', DATA, FLAGS))
                avg_cost += batch_cost
        # The division should stay outside the inner for loop.
        if (batch_num - excluded) > 0: avg_cost = avg_cost / (batch_num - excluded)            
        return avg_cost 

    # # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(var_list=None, max_to_keep=5)
    # # Initialize all the variables in the graph.
    sess.run('init')
    print('Initialized all the local and global variables in the graph....')
    for epoch in range(FLAGS.epoch_num):
        avg_cost = train_one_epoch(epoch, FLAGS.batch_size)
        print('Epoch {:03d} | Avg Cost: {:.5f}'.format(epoch, avg_cost))
        print_stats(get_metrics(sess, 'train'), 'train')
        print_stats(get_metrics(sess, 'valid'), 'valid')
        # Do __not__ delete the following 2 lines; they periodically save the
        # model.
        checkpoint_file = FLAGS.logdir + '/model.ckpt' # os.path.join(FLAGS.logdir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=epoch)
    return


def reshape_m_mtx(mtx):
    """Reshape the python list into a np array."""
    new_mtx = [0]
    for i in range(6):
        new_mtx.extend(mtx[i * 7:(i + 1) * 7])
        new_mtx.append(0)
    temp = np.array(new_mtx).reshape(7, 7)
    temp = .5 * (temp + temp.T)
    # # temp = np.triu(temp)
    return temp


def print_stats(stats, name):
    """Print the given stats."""
    conf_mtx, acc, auc_list, m_mtx_list = stats  #, labels_int, predictions, probs, pr_auc = stats
    
    conf_mtx = conf_mtx / conf_mtx.sum(axis=1, keepdims=True)
    m_mtx = reshape_m_mtx(m_mtx_list)
    print(
        '{:s}:'.format(name),
        '(Silly) ACC={:.4f}, Better ACC={:.4f},'.format(
            acc, conf_mtx.diagonal().mean()),
        'M-Measure={:.4f},'.format(m_mtx.sum() / (49 - 7)),
        'AUC={:.4f}'.format(auc_list.mean()), )
    print(('\t' * 6).join(['Confusion Matrix', 'M-Measure Matrix', 'AUC']))
    for conf_row, row, auc in zip(conf_mtx, m_mtx, auc_list):
        for conf_value in conf_row:
            print('{:.4f}'.format(conf_value), '\t', end='')
        print('|\t', end='')
        for value in row:
            print('{:.4f}'.format(value), '\t', end='')
        print('| \t{:.4f}'.format(auc), '\n')
    print('--------------------------------------------------------------'
          '------------')


def reset_and_update(sess, feed_dict):
    """Reset the local variables and update the necessary update ops."""
    sess.run('init/local/init')
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


def get_metrics(sess, mode):
    """Get the accuracy over the dataset corresponding to the given mode."""
    feed_dict = create_feed_dict(mode, DATA, FLAGS)
    reset_and_update(sess, feed_dict)
    
    metrics = sess.run(
        [
            'metrics/confusion/SparseTensorDenseAdd:0',
            'metrics/accuracy:0', 
            'metrics/auc:0', # 'metrics/auc/0/auc:0', # this is for showing the first position of array!!
            'metrics/m_measure:0',
            #'metrics/intlabels:0',
            #'metrics/predictions:0',
            #'metrics/probs:0',
            #'metrics/pr_curve:0',            
        ],    
        feed_dict=feed_dict)    
    #pmetrics = tf.Print(metrics, [metrics], message='Metrics: ')
    # print(pmetrics.eval(Session=sess))
    print('SparseTensorDenseAdd, accuracy, auc, m_measure: ', metrics)
    # output = sess.run('input_normalization/9_softmax_linear', feed_dict=feed_dict)
    return metrics


def train_and_summarize(sess, writers, step):
    """Train and record execution metadata for use in TB."""
    batch_writer = writers['batch']
    summary, _ = sess.run(
        ['Merge/MergeSummary:0', 'train'], feed_dict=create_feed_dict('batch', DATA, FLAGS))
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

    write_summaries(sess, writers, 'train', step)
    write_summaries(sess, writers, 'valid', step)
    return


def write_summaries(sess, writers, mode, step):
    """Get the summary protobuf and write it to the Summary.FileWriter."""
    writer = writers[mode]
    feed_dict = create_feed_dict(mode, DATA, FLAGS)
    reset_and_update(sess, feed_dict)
    summary = sess.run('Merge/MergeSummary:0', feed_dict=feed_dict)
    writer.add_summary(summary, step)
    writer.flush()
    return


def create_feed_dict(tag, DATA, FLAGS):
    """Create the feed dictionary for mapping data onto placeholders in the graph."""
    if tag == 'batch':
        features, targets, example_weights = DATA.train.next_batch( # example_weights = [1.0] it is not used in the inference.
            FLAGS.batch_size)
        # features = np.random.randn(*features.shape)
    elif tag == 'train':
        features = DATA.train.orig.features
        targets = DATA.train.orig.labels
        example_weights = np.ones_like(targets[:, 1])
    elif tag == 'valid':
        features = DATA.validation.features
        targets = DATA.validation.labels
        example_weights = np.ones_like(targets[:, 1])
    else:
        features = DATA.test.features
        targets = DATA.test.labels
        example_weights = np.ones_like(targets[:, 1])

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

def retrieve_tf_model():
    checkpoint_file = os.path.join(Path(FLAGS.logdir), 'my_test_model-1000.meta')
    print(checkpoint_file)
    # with tf.Session() as sess:    
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(checkpoint_file)
    new_saver.restore(sess, tf.train.latest_checkpoint(Path(FLAGS.logdir)))

def retrieve_FLAGS():
    return FLAGS
##########################
# ## SETTINGS
##########################
def main(_):
    print("Run the main program.")
    # # Number of Training Examples: 55_000
    # # Number of Validation Examples: 5_000
    # # Number of Test Examples: 10_000
    # # training batches per epoch (using a batch size of 64): 859

    # To determine an optimal set of hyperparameters, see Section 11.4.2 of the
    # deep learning book. Has (1) grid, (2) random, and (3) Bayesian
    # model-based search methods.Swersky et al. have a paper mentioned in that
    # section (published in 2014).

    # Hyperparameters
    #print("FLAGS.epoch_num", FLAGS.epoch_num)
    FLAGS.epoch_num = 20  # 14  # 17  # 35  # 15
    #print("FLAGS.epoch_num", FLAGS.epoch_num)
    FLAGS.batch_size = 256  # do NOT increase this to 1024 # 64  # 128  #
    FLAGS.dropout_keep = 0.9  # 0.9  # 0.95  # .75  # .6
    # ### parameters for training optimizer.
    FLAGS.learning_rate = .25  # .075  # .15  # .25
    FLAGS.momentum = .5  # used by the momentum SGD.

    # ### parameters for inverse_time_decay
    FLAGS.decay_rate = 1
    FLAGS.decay_step = 1 * 80000
    FLAGS.rate_min = .015
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

    # Architecture	
    architecture = {
        'n_input': DATA.train.features.shape[1],
        #'n_hidden_1': 100,  # 200,  # 2 * 256,  # 512, # 128,
        #'n_hidden_2': 140,  # 256,
        # 'n_hidden_3': 140,  # 128,
        # 'n_hidden_4': 140,
        # 'n_hidden_5': 140,
        'n_classes': DATA.train.num_classes
    }
    print('len(FLAGS.s_hidden)', len(FLAGS.s_hidden))
    if FLAGS.n_hidden < 0 : raise ValueError('The size of hidden layer must be at least 0')
    if (FLAGS.n_hidden > 0) and (FLAGS.n_hidden != len(FLAGS.s_hidden)) : raise ValueError('Sizes in hidden layers should match!')
    for hid_i in range(1, FLAGS.n_hidden+1):
        architecture['n_hidden_{:1d}'.format(hid_i)] = FLAGS.s_hidden[hid_i-1]
    '''
    for hid_i in range(1, FLAGS.n_hidden + 1):
            hidden_out = nn_layer(hidden_out,
                                  architecture['n_hidden_{:1d}'.format(hid_i)],
                                  '{:1d}_hidden'.format(hid_i), act)
    '''
    print('architecture', architecture)
    #print("FLAGS.n_hidden", FLAGS.n_hidden)
    #FLAGS.n_hidden = len(architecture) - 2
    #print("FLAGS.n_hidden", FLAGS.n_hidden)

    if tf.gfile.Exists(FLAGS.logdir):
       tf.gfile.DeleteRecursively(FLAGS.logdir)
    tf.gfile.MakeDirs(FLAGS.logdir)

    graph = build_graph(architecture, FLAGS.learning_rate)
    run_model(graph)
    return


def update_parser(parser):
    """Parse the arguments from the CLI and update the parser."""
    parser.add_argument(
        '--epoch_num',
        type=int,
        default=1000,
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
        default='/real_summaries',
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
        default=[100, 70, 70],
        help='Size of each hidden layer.')
    parser.add_argument(
        '--stratified_flag',
        type=bool,
        default=False,
        help='Execute Stratified Sampling of the DataSet by Delinquency Status')
    return parser.parse_known_args()


FLAGS, UNPARSED = update_parser(argparse.ArgumentParser())
print("FLAGS", FLAGS)
print("UNPARSED", UNPARSED)
FLAGS.weighted_sampling = False  # True  #

# %%
if __name__ == '__main__':    
    # random seed for the mnist iterator
    np.random.seed(RANDOM_SEED)  # pylint: disable=no-member
    DATA = md.get_data(220000, 20000, 20000, FLAGS.weighted_sampling, dataset_name='MORT', stratified_flag = FLAGS.stratified_flag, refNorm=True)  # 'MNIST')
    # print(DATA.validation.labels.sum(axis=0))
    # main(1)
    # print("before tf.app.run(...)")    
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)

    # # # temp_cov = np.cov(DATA.train.orig.features.T)
    # # # _, xx, v = np.linalg.svd(temp_cov)
    # # # print(xx)
    # # # print(xx / xx.sum())
    # from sklearn.decomposition import PCA
    # from sklearn.preprocessing import QuantileTransformer
    # from sklearn.preprocessing import RobustScaler
    # data = DATA.train.orig.features
    # normalizer1 = QuantileTransformer(output_distribution='uniform')
    # # normalizer1 = RobustScaler(quantile_range=(5.0, 95.0))
    # data1 = normalizer1.fit_transform(data)
    # pca = PCA(n_components=.99, svd_solver='full')
    # pca.fit(data1)
    # print(pca.explained_variance_)
    # print(pca.explained_variance_ratio_)
    # # print(pca.singular_values_)

# bash commmand:
# tensorboard --logdir=/tmp/tensorflow/real/logs/real_summaries/

# Very useful blog on imbalanced data:
# http://www.kdnuggets.com/2016/08/learning-from-imbalanced-classes.html

