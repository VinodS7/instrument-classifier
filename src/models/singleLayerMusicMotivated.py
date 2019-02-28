import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from keras.layers import Input, Dropout, Concatenate
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD

import argparse



from keras.regularizers import l2

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

#class SingleLayerMusicMotivated(ModelsParentClass):




def build_model():

    if K.image_data_format() == 'th':
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

    model_layers = list()

    for m_i in m_sizes:
        for i, n_i in enumerate(n_sizes):
            x = Conv2D(n_filters[i], (m_i, n_i),
                              padding='same',
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(1e-5),
                              name=str(n_i)+'_'+str(m_i)+'_'+'conv')(melgram_input)
            x = BatchNormalization(axis=channel_axis, name=str(n_i)+'_'+str(m_i)+'_'+'bn')(x)
            x = ELU()(x)
            x = MaxPooling2D(pool_size=(N_MEL_BANDS, SEGMENT_DUR/maxpool_const), name=str(n_i)+'_'+str(m_i)+'_'+'pool')(x)
            x = Flatten(name=str(n_i)+'_'+str(m_i)+'_'+'flatten')(x)
            model_layers.append(x)

    x = Concatenate(axis=channel_axis)(model_layers)
    x = Dropout(0.5)(x)
    x = Dense(N_CLASSES, kernel_initializer='he_normal', kernel_regularizer=l2(1e-5), activation='softmax', name='prediction')(x)
    model = Model(melgram_input, x)
    return model

def train(train_dir,val_dir):
    train_dataset = tf.data.TFRecordDataset(train_dir)
    val_dataset = tf.data.TFRecordDataset(val_dir)
    model = build_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_EPOCH)

    save_clb = ModelCheckpoint(
        "{weights_basepath}/{model_path}/".format(
            weights_basepath=MODEL_WEIGHT_BASEPATH,
            model_path="SingleLayerMM") +
        "epoch.{epoch:02d}-val_loss.{val_loss:.3f}-fbeta.{val_fbeta_score:.3f}" + "-{key}.hdf5".format(
            key="SLMM"),
        monitor='val_loss',
        save_best_only=True)

    lrs = LearningRateScheduler(lambda epoch_n: init_lr / (2**(epoch_n//SGD_LR_REDUCE)))
    model.summary()
    optimizer = SGD(lr=init_lr, momentum=0.9, nesterov=True);
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    history = model.fit(train_dataset,
                                  samples_per_epoch=289205,
                                  epochs=MAX_EPOCH_NUM,
                                  verbose=2,
                                  callbacks=[save_clb, early_stopping, lrs],
                                  validation_data=val_dataset,
                                  class_weight=None,
                                  nb_worker=1)

    print(history)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adds a spectogram field to the TFRecord")
    parser.add_argument("train", help="Train TFRecord file")
    parser.add_argument("val", help="Val TFRecord file")
    args = parser.parse_args()
    train(args.train, args.val)



