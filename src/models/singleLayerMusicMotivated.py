from keras import backend as K
from keras.models import Model
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from keras.layers import Input, Dropout, merge
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2

N_MEL_BANDS = 80
SEGMENT_DUR = 247

def build_model(n_classes):

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

    layers = list()

    for m_i in m_sizes:
        for i, n_i in enumerate(n_sizes):
            x = Convolution2D(n_filters[i], m_i, n_i,
                              border_mode='same',
                              init='he_normal',
                              W_regularizer=l2(1e-5),
                              name=str(n_i)+'_'+str(m_i)+'_'+'conv')(melgram_input)
            x = BatchNormalization(axis=channel_axis, mode=0, name=str(n_i)+'_'+str(m_i)+'_'+'bn')(x)
            x = ELU()(x)
            x = MaxPooling2D(pool_size=(N_MEL_BANDS, SEGMENT_DUR/maxpool_const), name=str(n_i)+'_'+str(m_i)+'_'+'pool')(x)
            x = Flatten(name=str(n_i)+'_'+str(m_i)+'_'+'flatten')(x)
            layers.append(x)

    x = merge(layers, mode='concat', concat_axis=channel_axis)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, init='he_normal', W_regularizer=l2(1e-5), activation='softmax', name='prediction')(x)
    model = Model(melgram_input, x)

    return model