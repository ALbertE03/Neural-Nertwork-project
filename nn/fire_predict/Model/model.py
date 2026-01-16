import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

class ConvLSTMAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction

        # Channel attention
        self.avg_pool = tf.keras.layers.TimeDistributed(
            tf.keras.layers.GlobalAveragePooling2D()
        )
        self.fc1 = tf.keras.layers.Dense(channels // reduction, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

        # Spatial attention
        self.spatial_conv = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(
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


        return x * s
def build_convlstm_bottleneck128(
    input_shape=(3, 256, 256, 28),
    dropout=0.3
):
    inputs = layers.Input(shape=input_shape)

    #  ENCODER 
    e1 = layers.ConvLSTM2D(
        24, 3, padding='same',
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=0.2
    )(inputs)
    e1 = layers.LayerNormalization()(e1)

    p1 = layers.TimeDistributed(layers.MaxPooling2D(2))(e1)

    e2 = layers.ConvLSTM2D(
        48, 3, padding='same',
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=0.2
    )(p1)
    e2 = layers.LayerNormalization()(e2)

    p2 = layers.TimeDistributed(layers.MaxPooling2D(2))(e2)

    e3 = layers.ConvLSTM2D(
        96, 3, padding='same',
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=0.2
    )(p2)
    e3 = layers.LayerNormalization()(e3)

    #  BOTTLENECK 
    p3 = layers.TimeDistributed(layers.MaxPooling2D(2))(e3)

    b = layers.ConvLSTM2D(
        128, 3, padding='same',
        return_sequences=True,
        dropout=dropout
    )(p3)
    b = layers.LayerNormalization()(b)
    b = ConvLSTMAttentionBlock(128)(b)

    #  DECODER 
    u3 = layers.TimeDistributed(layers.UpSampling2D(2))(b)
    u3 = layers.Concatenate(axis=-1)([u3, e3])
    u3 = layers.ConvLSTM2D(96, 3, padding='same', return_sequences=True)(u3)

    u2 = layers.TimeDistributed(layers.UpSampling2D(2))(u3)
    u2 = layers.Concatenate(axis=-1)([u2, e2])
    u2 = layers.ConvLSTM2D(48, 3, padding='same', return_sequences=True)(u2)
    u2 = ConvLSTMAttentionBlock(48)(u2)

    u1 = layers.TimeDistributed(layers.UpSampling2D(2))(u2)
    u1 = layers.Concatenate(axis=-1)([u1, e1])
    u1 = layers.ConvLSTM2D(24, 3, padding='same', return_sequences=True)(u1)
    u1 = ConvLSTMAttentionBlock(24)(u1)

    #  OUTPUT 
    x = layers.ConvLSTM2D(
        16, 3, padding='same',
        return_sequences=False
    )(u1)

    out = layers.Conv2D(1, 1, padding='same')(x)
    out = layers.Activation('linear', dtype='float32', name='predictions')(out)

    return models.Model(inputs, out, name="ConvLSTM_UNet_Att128")
