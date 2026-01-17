import tensorflow as tf 
from tensorflow.keras.layers import TimeDistributed,GlobalAveragePooling2D,\
                                    Dense,Conv2D,Input,ConvLSTM2D,LayerNormalization,\
                                    MaxPooling2D,Concatenate,UpSampling2D,Activation,Layer

from tensorflow.keras.models import Model

class ConvLSTMAttentionBlock(Layer):
    def __init__(self, channels, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction

        # Channel attention
        self.avg_pool = TimeDistributed(
            GlobalAveragePooling2D()
        )
        self.fc1 = Dense(channels // reduction, activation='relu')
        self.fc2 = Dense(channels, activation='sigmoid')

        # Spatial attention
        self.spatial_conv = TimeDistributed(
            Conv2D(
                filters=1,
                kernel_size=7,
                padding='same',
                activation='sigmoid'
            )
        )

    def call(self, x):
        # x: (B, T, H, W, C)

        #  Channel Attention 
        c = self.avg_pool(x)              # (B, T, C)
        c = self.fc1(c)                   # (B, T, C//r)
        c = self.fc2(c)                   # (B, T, C)
        c = tf.expand_dims(c, axis=2)     # (B, T, 1, C)
        c = tf.expand_dims(c, axis=2)     # (B, T, 1, 1, C)

        x = x * c

        #  Spatial Attention 
        avg_s = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_s = tf.reduce_max(x, axis=-1, keepdims=True)
        s = tf.concat([avg_s, max_s], axis=-1)

        s = self.spatial_conv(s)          # (B, T, H, W, 1)

        return x * s

def build_convlstm_bottleneck128(
    input_shape=(3, 256, 256, 28),
    dropout=0.3,
    reduction=4
):
    inputs = Input(shape=input_shape)

    #  ENCODER 
    e1 = ConvLSTM2D(
        24, 3, padding='same',
        return_sequences=True,
        dropout=dropout,
    )(inputs)
    e1 = LayerNormalization()(e1)

    p1 = TimeDistributed(MaxPooling2D(2))(e1)

    e2 = ConvLSTM2D(
        48, 3, padding='same',
        return_sequences=True,
        dropout=dropout
    )(p1)
    e2 = LayerNormalization()(e2)

    p2 = TimeDistributed(MaxPooling2D(2))(e2)

    e3 = ConvLSTM2D(
        96, 3, padding='same',
        return_sequences=True,
        dropout=dropout
    )(p2)
    e3 = LayerNormalization()(e3)

    #  BOTTLENECK 
    p3 = TimeDistributed(MaxPooling2D(2))(e3)

    b = ConvLSTM2D(
        128, 3, padding='same',
        return_sequences=True,
        dropout=dropout
    )(p3)
    b = LayerNormalization()(b)
    b = ConvLSTMAttentionBlock(128,reduction=reduction)(b)

    #  DECODER 
    u3 = TimeDistributed(UpSampling2D(2))(b)
    u3 = Concatenate(axis=-1)([u3, e3])
    u3 = ConvLSTM2D(96, 3, padding='same', return_sequences=True)(u3)

    u2 = TimeDistributed(UpSampling2D(2))(u3)
    u2 = Concatenate(axis=-1)([u2, e2])
    u2 = ConvLSTM2D(48, 3, padding='same', return_sequences=True)(u2)
    u2 = ConvLSTMAttentionBlock(48,reduction=reduction)(u2)

    u1 = TimeDistributed(UpSampling2D(2))(u2)
    u1 = Concatenate(axis=-1)([u1, e1])
    u1 = ConvLSTM2D(24, 3, padding='same', return_sequences=True)(u1)
    u1 = ConvLSTMAttentionBlock(24,reduction=reduction)(u1)

    # OUTPUT 
    x = ConvLSTM2D(
        16, 3, padding='same',
        return_sequences=False
    )(u1)

    out = Conv2D(1, 1, padding='same')(x)
    out = Activation('linear', dtype='float32', name='predictions')(out)

    return Model(inputs, out, name="ConvLSTM_UNet_Att128")
