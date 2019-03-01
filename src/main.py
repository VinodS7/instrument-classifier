
from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from models.model_basic_cnn import ModelBasicCNN
from models.singleLayerMM2 import SingleLayerMM2


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                         'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_data', '/import/c4dm-05/vs/train-no-reed.tfrecord', 'Directory to put the training data.')
flags.DEFINE_string('valid_data', '/import/c4dm-05/vs/valid-no-reed.tfrecord', 'Directory to put the training data.')
flags.DEFINE_boolean('is_training', True, 'If true, selected model trains')
flags.DEFINE_string('model_name','single_layer_mm2','Name of the model for training or inference')
flags.DEFINE_integer('nb_classes',11,'Number of classes for the classifier')
flags.DEFINE_integer('valid_iters',241,'Number of valid iterations')
flags.DEFINE_integer('train_iters',5307,'Number of train iterations')


print(FLAGS)


if(FLAGS.model_name=='model_basic_cnn'):
    if(FLAGS.is_training):
        model = ModelBasicCNN(FLAGS.learning_rate,FLAGS.is_training,FLAGS.nb_classes,FLAGS.num_epochs,FLAGS.train_iters,FLAGS.valid_iters,FLAGS.batch_size,FLAGS.train_data,FLAGS.valid_data)
        #model.test_load_from_tfrecords()
        model.train()

if(FLAGS.model_name=='single_layer_mm2'):
    if(FLAGS.is_training):
        model = SingleLayerMM2(FLAGS.learning_rate,FLAGS.is_training,FLAGS.nb_classes,FLAGS.num_epochs,FLAGS.train_iters,FLAGS.valid_iters,FLAGS.batch_size,FLAGS.train_data,FLAGS.valid_data)
        #model.test_load_from_tfrecords()
        model.train()