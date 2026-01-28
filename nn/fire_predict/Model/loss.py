import tensorflow as tf
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1.0, name="dice_loss"):
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_pred_prob = tf.nn.sigmoid(y_pred)

        # Aplanar los tensores
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred_prob, [-1])

        # Cálculo de componentes
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        
        # Dice: (2 * Intersección) / (Suma de elementos de cada set)
        dice = (2. * intersection + self.smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth
        )

        return 1 - dice