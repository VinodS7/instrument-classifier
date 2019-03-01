from __future__ import absolute_import, print_function, division

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from keras.layers import Input, Dropout, concatenate
from keras.layers.advanced_activations import ELU
from models.model_parent_class import ModelsParentClass



N_MEL_BANDS = 80
SEGMENT_DUR = 247
N_CLASSES = 11


MODEL_WEIGHT_BASEPATH = 'weights/'
MODEL_HISTORY_BASEPATH = 'history/'
MODEL_MEANS_BASEPATH = 'means/'
MAX_EPOCH_NUM = 400
EARLY_STOPPING_EPOCH = 20
SGD_LR_REDUCE = 5
BATCH_SIZE = 16
init_lr = 0.001



class SingleLayerMM2(ModelsParentClass):

    def __init__(self, learning_rate, training, nb_classes, num_epochs, train_iters, valid_iters, batch_size,
                 train_data, valid_data):
        # mode defines whether the model is training or infering
        self.training = training
        self.learning_rate = learning_rate
        self.nb_classes = nb_classes
        self.num_epochs = num_epochs
        self.train_iters = train_iters
        self.valid_iters = valid_iters
        self.batch_size = batch_size
        self.train_data = train_data
        self.valid_data = valid_data

        super()

    def forward_propogation(self, x):
        melgram_input = tf.reshape(x, [-1, 247, 80, 1])
        if K.image_dim_ordering() == 'th':
            input_shape = (1, N_MEL_BANDS, SEGMENT_DUR)
            channel_axis = 1
        else:
            input_shape = (N_MEL_BANDS, SEGMENT_DUR, 1)
            channel_axis = 3
        melgram_input = Input(shape=input_shape)

        m_sizes = [50, 70]
        n_sizes = [1, 3, 5]
        n_filters = [128, 64, 32]
        maxpool_const = 4
        maxpool_size = int(SEGMENT_DUR / maxpool_const)
        layers = list()

        for m_i in m_sizes:
            for i, n_i in enumerate(n_sizes):
                x = Convolution2D(m_i, n_i, n_filters[i],
                                  border_mode='same',
                                  init='he_normal',
                                  W_regularizer=l2(1e-5),
                                  name=str(n_i) + '_' + str(m_i) + '_' + 'conv')(melgram_input)
                x = BatchNormalization(axis=channel_axis, mode=0, name=str(n_i) + '_' + str(m_i) + '_' + 'bn')(x)
                x = ELU()(x)
                x = MaxPooling2D(pool_size=(N_MEL_BANDS, maxpool_size), name=str(n_i) + '_' + str(m_i) + '_' + 'pool')(
                    x)
                x = Flatten(name=str(n_i) + '_' + str(m_i) + '_' + 'flatten')(x)
                layers.append(x)
        x = concatenate(layers)
        x = Dropout(0.5)(x)
        x = Dense(N_CLASSES, init='he_normal', W_regularizer=l2(1e-5), activation='softmax', name='prediction')(x)
        model = Model(melgram_input, x)




        return model

    def train(self):
        #super().print_message()
        # Creating a graph
        graph = tf.Graph()

        with graph.as_default():


            # Create global step for savind models
            global_step = tf.train.get_or_create_global_step()

            # Create pipeline for loading training dataset
            filename = tf.placeholder(tf.string, shape=[None])
            dataset = tf.data.TFRecordDataset(filename)
            dataset = dataset.map(super().read_from_tfrecord)
            dataset = dataset.shuffle(1000)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            spec, label = iterator.get_next()

            # Compute logits
            logits = self.forward_propogation(spec)

            # Compute cross entropy loss and record variable
            loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=logits)

            tf.summary.scalar('loss', loss)

            # optimize weights
            decayed_lr = tf.train.exponential_decay(self.learning_rate, global_step, 10000, 0.95, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr).minimize(loss, global_step=global_step)

            # Calculate training_accuracy of model
            accuracy, confusion_matrix = super().compute_accuracy(logits, label, self.nb_classes)
            tf.summary.scalar('accuracy', accuracy)
            # Initialize variables and variable saver
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("logdir/binary/train")
            valid_writer = tf.summary.FileWriter("logdir/binary/valid")

        with tf.Session(graph=graph) as sess:
            K.set_session(sess)
            # initialize model parameters
            sess.run([init])

            for epoch in range(self.num_epochs):
                sess.run([iterator.initializer], feed_dict={filename: [self.train_data]})
                t_c = 0
                t_l = 0
                for train_iter in range(self.train_iters):
                    l, _, acc, summary = sess.run([loss, optimizer, accuracy, merged])
                    t_c += np.squeeze(acc)
                    t_l += np.squeeze(l)
                    train_writer.add_summary(summary, epoch * self.train_iters + train_iter)
                c = 0
                val_loss = 0
                count = 0
                cm = np.zeros([self.nb_classes, self.nb_classes])
                sess.run([iterator.initializer], feed_dict={filename: [self.valid_data]})
                for valid_iter in range(self.valid_iters):
                    acc, l, summary, confm = sess.run([accuracy, loss, merged, confusion_matrix])
                    c += np.squeeze(acc)
                    cm += np.squeeze(confm)
                    val_loss += np.squeeze(l)
                    valid_writer.add_summary(summary, epoch * self.valid_iters + valid_iter)
                print("Validation accuracy:", float(c / self.valid_iters), "Validation loss:",
                      float(val_loss / self.valid_iters), "Training accuracy:", float(t_c / self.train_iters),
                      "Training loss:", float(t_l / self.train_iters))
            print(cm)

            # Save model for this epoch
            # saver.save(sess, './src/saved_model/nsynth_binary_classifier', global_step=global_step)

    def test_load_from_tfrecords(self):
        graph = tf.Graph()
        print("Train data path:", self.train_data)
        print("Valid data path:", self.valid_data)
        print("Batch size:", self.batch_size)
        with graph.as_default():
            # Create global step for savind models
            global_step = tf.train.get_or_create_global_step()

            # Create pipeline for loading training dataset
            filename = tf.placeholder(tf.string, shape=[None])
            dataset = tf.data.TFRecordDataset(filename)
            dataset = dataset.map(super().read_from_tfrecord)
            dataset = dataset.shuffle(1000)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            spec, label = iterator.get_next()

        with tf.Session(graph=graph) as sess:
            sess.run([iterator.initializer], feed_dict={filename: [self.train_data]})
            s, l = sess.run([spec, label])
            print("Train labels:", l)
            sess.run([iterator.initializer], feed_dict={filename: [self.valid_data]})
            s, l = sess.run([spec, label])
            print("Valid labels:", l)
