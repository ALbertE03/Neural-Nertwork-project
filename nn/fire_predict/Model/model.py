import tensorflow as tf
from tensorflow.keras import layers, models

class AttentionBlock3D(layers.Layer):
    """Attention Gate para 3D UNet en TensorFlow."""
    def __init__(self, F_int, **kwargs):
        super().__init__(**kwargs)
        self.W_g = models.Sequential([
            layers.Conv3D(F_int, kernel_size=1, strides=1, padding='same', use_bias=True),
            layers.BatchNormalization()
        ])
        self.W_x = models.Sequential([
            layers.Conv3D(F_int, kernel_size=1, strides=1, padding='same', use_bias=True),
            layers.BatchNormalization()
        ])
        self.psi = models.Sequential([
            layers.Conv3D(1, kernel_size=1, strides=1, padding='same', use_bias=True),
            layers.BatchNormalization(),
            layers.Activation('sigmoid')
        ])

    def call(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = layers.Activation('relu')(g1 + x1)
        psi = self.psi(psi)
        return x * psi

def conv_block(x, out_c, dropout_rate):
    """Bloque de doble convolución 3D."""
    x = layers.Conv3D(out_c, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv3D(out_c, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

def build_unet3d(input_shape=(3, 256, 256, 28), out_channels=1, dropout=0.3):
    # Dimensiones: (Tiempo, Alto, Ancho, Canales)
    inputs = layers.Input(shape=input_shape)

    # Encoder
    # Nivel 1
    e1 = conv_block(inputs, 32, dropout)
    # MaxPool3D: Solo reducimos H y W (1, 2, 2) igual que en tu PyTorch
    p1 = layers.MaxPooling3D(pool_size=(1, 2, 2))(e1)
    
    # Nivel 2
    e2 = conv_block(p1, 64, dropout)
    p2 = layers.MaxPooling3D(pool_size=(1, 2, 2))(e2)

    # Bottleneck
    b = conv_block(p2, 256, dropout)

    # Decoder
    # Up 2
    u2 = layers.UpSampling3D(size=(1, 2, 2))(b)
    # Atención 2: g es la señal del decoder, x es la skip connection
    att2 = AttentionBlock3D(F_int=64)(g=u2, x=e2)
    d2 = layers.Concatenate()([u2, att2])
    d2 = conv_block(d2, 64, dropout)

    # Up 1
    u1 = layers.UpSampling3D(size=(1, 2, 2))(d2)
    # Atención 1
    att1 = AttentionBlock3D(F_int=32)(g=u1, x=e1)
    d1 = layers.Concatenate()([u1, att1])
    d1 = conv_block(d1, 32, dropout)

    # Salida Final
    out = layers.Conv3D(out_channels, kernel_size=1)(d1)
    
    # Temporal Compressor: Reduce la dimensión T de 3 a 1
    # Usamos padding='valid' para que el kernel de 3 reduzca (3, H, W) -> (1, H, W)
    out = layers.Conv3D(out_channels, kernel_size=(3, 1, 1), padding='valid')(out)
    
    # Squeeze: En TF usamos Reshape o Lambda para quitar la dimensión temporal
    # De (Batch, 1, 256, 256, 1) -> (Batch, 256, 256, 1)
    out = layers.Reshape((input_shape[1], input_shape[2], out_channels))(out)

    return models.Model(inputs, out, name="UNet3D_Attention")
