from tensorflow.keras import regularizers, backend
from tensorflow.keras.layers import Dense, MaxPooling2D, Convolution2D, Flatten, ZeroPadding2D,BatchNormalization, \
    GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model


def create_model():
    def l2_norm2d(x, pool_size, strides,
                  padding='valid', data_format=None):
        if strides is None:
            strides = pool_size
        x = x ** 2
        x = backend.pool2d(x, pool_size, strides,
                           padding, data_format, pool_mode='avg')
        x = backend.sqrt(x)
        return x

    X = Input(shape=(224, 224, 3))
    zero3 = ZeroPadding2D(padding=(4, 4))(X)
    conv_3 = Convolution2D(64, (7, 7), strides=(2, 2), activation='selu')(zero3)
    max_1 = MaxPooling2D((3, 3), strides=(2, 2))(conv_3)
    batch_1 = BatchNormalization()(max_1)

    zero4 = ZeroPadding2D(padding=(2, 2))(batch_1)
    inca1_1 = Convolution2D(64, (1, 1), activation='selu')(zero4)
    inca1_1 = Convolution2D(192, (3, 3), activation='selu')(inca1_1)
    batch_2 = BatchNormalization()(inca1_1)
    max_2 = MaxPooling2D((3, 3), strides=(2, 2))(batch_2)

    inca2_1 = Convolution2D(64, (1, 1), activation='selu')(max_2)

    inca2_2 = Convolution2D(96, (1, 1), activation='selu')(max_2)
    zero5 = ZeroPadding2D(padding=(1, 1))(inca2_2)
    inca2_2 = Convolution2D(128, (3, 3), activation='selu')(zero5)

    inca2_3 = Convolution2D(16, (1, 1), activation='selu')(max_2)
    zero6 = ZeroPadding2D(padding=(2, 2))(inca2_3)
    inca2_3 = Convolution2D(32, (5, 5), activation='selu')(zero6)

    max_3 = MaxPooling2D((3, 3), strides=(2, 2))(max_2)
    max_3 = ZeroPadding2D(padding=((7, 8), (7, 8)))(max_3)
    max_3 = Convolution2D(32, (1, 1), activation='selu')(max_3)


    conc1 = backend.concatenate([inca2_1, inca2_2, inca2_3, max_3])


    inca2_4 = Convolution2D(64, (1, 1), activation='selu')(conc1)

    inca2_5 = Convolution2D(96, (1, 1), activation='selu')(conc1)
    zero8 = ZeroPadding2D(padding=(1, 1))(inca2_5)
    inca2_5 = Convolution2D(128, (3, 3), activation='selu')(zero8)

    inca2_6 = Convolution2D(32, (1, 1), activation='selu')(conc1)
    zero9 = ZeroPadding2D(padding=(2, 2))(inca2_6)
    inca2_6 = Convolution2D(64, (5, 5), activation='selu')(zero9)

    l2a_1 = l2_norm2d(conc1, (3, 3), (2, 2))
    l2a_1 = ZeroPadding2D(padding=((7, 8), (7, 8)))(l2a_1)
    l2a_1 = Convolution2D(64, (1, 1), activation='selu')(l2a_1)

    conc2 = backend.concatenate([inca2_4, inca2_5, inca2_6, l2a_1])


    zero11 = ZeroPadding2D(padding=(1, 1))(conc2)
    zero12 = ZeroPadding2D(padding=(2, 2))(conc2)
    zero13 = ZeroPadding2D(padding=(1, 1))(conc2)

    inca2_7 = Convolution2D(128, (1, 1), activation='selu')(zero11)
    inca2_7 = Convolution2D(256, (3, 3), strides=(2, 2), activation='selu')(inca2_7)
    inca2_8 = Convolution2D(32, (1, 1), activation='selu')(zero12)
    inca2_8 = Convolution2D(64, (5, 5), strides=(2, 2), activation='selu')(inca2_8)
    max_4 = MaxPooling2D((3, 3), strides=(2, 2))(zero13)

    conc3 = backend.concatenate([inca2_7, inca2_8, max_4])


    inca3_1 = Convolution2D(256, (1, 1), activation='selu')(conc3)

    inca3_2 = Convolution2D(96, (1, 1), activation='selu')(conc3)
    zero14 = ZeroPadding2D(padding=(1, 1))(inca3_2)
    inca3_2 = Convolution2D(192, (3, 3), activation='selu')(zero14)

    inca3_3 = Convolution2D(32, (1, 1), activation='selu')(conc3)
    zero15 = ZeroPadding2D(padding=(2, 2))(inca3_3)
    inca3_3 = Convolution2D(64, (5, 5), activation='selu')(zero15)

    zero16 = ZeroPadding2D(padding=((7, 8), (7, 8)))(conc3)
    l2a_2 = l2_norm2d(zero16, (3, 3), (2, 2))
    l2a_2 = Convolution2D(128, (1, 1), activation='selu')(l2a_2)

    conc4 = backend.concatenate([inca3_1, inca3_2, inca3_3, l2a_2])


    inca3_4 = Convolution2D(224, (1, 1), activation='selu')(conc4)

    inca3_5 = Convolution2D(112, (1, 1), activation='selu')(conc4)
    zero17 = ZeroPadding2D(padding=(1, 1))(inca3_5)
    inca3_5 = Convolution2D(224, (3, 3), activation='selu')(zero17)

    inca3_6 = Convolution2D(32, (1, 1), activation='selu')(conc4)
    zero18 = ZeroPadding2D(padding=(2, 2))(inca3_6)
    inca3_6 = Convolution2D(64, (5, 5), activation='selu')(zero18)

    l2a_3 = ZeroPadding2D(padding=((7, 8), (7, 8)))(conc4)
    l2a_3 = l2_norm2d(l2a_3, (3, 3), (2, 2))
    l2a_3 = Convolution2D(128, (1, 1), activation='selu')(l2a_3)

    conc5 = backend.concatenate([inca3_4, inca3_5, inca3_6, l2a_3])


    inca3_7 = Convolution2D(192, (1, 1), activation='selu')(conc5)

    inca3_8 = Convolution2D(128, (1, 1), activation='selu')(conc5)
    zero20 = ZeroPadding2D(padding=(1, 1))(inca3_8)
    inca3_8 = Convolution2D(256, (3, 3), activation='selu')(zero20)

    inca3_9 = Convolution2D(32, (1, 1), activation='selu')(conc5)
    zero21 = ZeroPadding2D(padding=(2, 2))(inca3_9)
    inca3_9 = Convolution2D(64, (5, 5), activation='selu')(zero21)

    l2a_4 = ZeroPadding2D(padding=((7, 8), (7, 8)))(conc5)
    l2a_4 = l2_norm2d(l2a_4, (3, 3), (2, 2))
    l2a_4 = Convolution2D(128, (1, 1), activation='selu')(l2a_4)

    conc6 = backend.concatenate([inca3_7, inca3_8, inca3_9, l2a_4])


    inca3_10 = Convolution2D(160, (1, 1), activation='selu')(conc6)

    inca3_11 = Convolution2D(144, (1, 1), activation='selu')(conc6)
    zero23 = ZeroPadding2D(padding=(1, 1))(inca3_11)
    inca3_11 = Convolution2D(288, (3, 3), activation='selu')(zero23)

    inca3_12 = Convolution2D(32, (1, 1), activation='selu')(conc6)
    zero24 = ZeroPadding2D(padding=(2, 2))(inca3_12)
    inca3_12 = Convolution2D(64, (5, 5), activation='selu')(zero24)

    l2a_5 = ZeroPadding2D(padding=((7, 8), (7, 8)))(conc6)
    l2a_5 = l2_norm2d(l2a_5, (3, 3), (2, 2))
    l2a_5 = Convolution2D(128, (1, 1), activation='selu')(l2a_5)

    conc7 = backend.concatenate([inca3_10, inca3_11, inca3_12, l2a_5])


    zero26 = ZeroPadding2D(padding=(1, 1))(conc7)
    zero27 = ZeroPadding2D(padding=(2, 2))(conc7)
    zero28 = ZeroPadding2D(padding=(1, 1))(conc7)
    inca3_13 = Convolution2D(160, (1, 1), activation='selu')(zero26)
    inca3_13 = Convolution2D(256, (3, 3), strides=(2, 2), activation='selu')(inca3_13)
    inca3_14 = Convolution2D(64, (1, 1), activation='selu')(zero27)
    inca3_14 = Convolution2D(128, (5, 5), strides=(2, 2), activation='selu')(inca3_14)
    max_5 = MaxPooling2D((3, 3), strides=(2, 2))(zero28)
    conc8 = backend.concatenate([inca3_13, inca3_14, max_5])


    inca4_1 = Convolution2D(384, (1, 1), activation='selu')(conc8)

    inca4_2 = Convolution2D(192, (1, 1), activation='selu')(conc8)
    zero29 = ZeroPadding2D(padding=(1, 1))(inca4_2)
    inca4_2 = Convolution2D(384, (3, 3), activation='selu')(zero29)

    inca4_3 = Convolution2D(48, (1, 1), activation='selu')(conc8)
    zero30 = ZeroPadding2D(padding=(2, 2))(inca4_3)
    inca4_3 = Convolution2D(128, (5, 5), activation='selu')(zero30)

    l2a_6 = ZeroPadding2D(padding=(4, 4))(conc8)
    l2a_6 = l2_norm2d(l2a_6, (3, 3), (2, 2))
    l2a_6 = Convolution2D(128, (1, 1), activation='selu')(l2a_6)

    conc9 = backend.concatenate([inca4_1, inca4_2, inca4_3, l2a_6])


    inca4_4 = Convolution2D(384, (1, 1), activation='selu')(conc9)

    inca4_5 = Convolution2D(192, (1, 1), activation='selu')(conc9)
    zero32 = ZeroPadding2D(padding=(1, 1))(inca4_5)
    inca4_5 = Convolution2D(384, (3, 3), activation='selu')(zero32)

    inca4_6 = Convolution2D(48, (1, 1), activation='selu')(conc9)
    zero33 = ZeroPadding2D(padding=(2, 2))(inca4_6)
    inca4_6 = Convolution2D(128, (5, 5), activation='selu')(zero33)

    max_6 = ZeroPadding2D(padding=((4, 4), (4, 4)))(conc9)
    max_6 = MaxPooling2D((3, 3), strides=(2, 2))(max_6)
    max_6 = Convolution2D(128, (1, 1), activation='selu')(max_6)

    conc10 = backend.concatenate([inca4_4, inca4_5, inca4_6, max_6])

    glob = GlobalAveragePooling2D()(conc10)
    dense1 = Dense(128, activation='selu')(glob)
    L2a = backend.l2_normalize(dense1)

    return Model(inputs=[X], outputs=L2a)