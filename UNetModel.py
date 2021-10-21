from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Concatenate, BatchNormalization, Dropout, UpSampling2D
from tensorflow.keras.models import Model


class UNetModel:
    """
    Class for creating a standard U-Net architecture
    """

    def build(self, width, height, n_channels=3, n_classes=3):
        """
        Created the U-Net network
        :param width: 
        :param height:
        :param n_channels:
        :param n_classes:
        :return:
        """

        inputs = Input(shape=(height, width, n_channels))

        # encoder
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(inputs)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(conv)
        conv1 = BatchNormalization()(conv) # skip connection #1
        max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)

        conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(max_pool)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(conv)
        conv2 = BatchNormalization()(conv) # skip connection #2
        max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)

        conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(max_pool)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(conv)
        conv3 = BatchNormalization()(conv) # skip connection #3
        max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv3)

        # bottleneck
        conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(max_pool)
        conv = BatchNormalization()(conv) #skip connection 4 
        conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(conv)
        conv4 = BatchNormalization()(conv)
        max_pool=MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv4)

        # decoder
        us = UpSampling2D((2, 2))(conv)
        skip_con = Concatenate()([us, conv3])
        conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(skip_con)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(conv)
        conv = BatchNormalization()(conv)

        us = UpSampling2D((2, 2))(conv)
        skip_con = Concatenate()([us, conv2])
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(skip_con)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(conv)
        conv = BatchNormalization()(conv)

        us = UpSampling2D((2, 2))(conv)
        skip_con = Concatenate()([us, conv1])
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(skip_con)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(conv)
        conv = BatchNormalization()(conv)  

        outputs = Conv2D(n_classes, (1, 1), padding='same', activation='sigmoid')(conv)
        model = Model(inputs, outputs)

        return model
