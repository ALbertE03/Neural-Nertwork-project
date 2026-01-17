import tensorflow as tf
class FocalDiceLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, smooth=1.0, name="focal_dice_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_tol = tf.nn.max_pool2d(y_true, ksize=5, strides=1, padding='SAME')

        y_pred_prob = tf.nn.sigmoid(y_pred)
        eps = tf.keras.backend.epsilon()
        y_pred_prob = tf.clip_by_value(y_pred_prob, eps, 1.0 - eps)

   
        focal_pos = -self.alpha * tf.pow(1 - y_pred_prob, self.gamma) * tf.math.log(y_pred_prob)
        focal_neg = -(1 - self.alpha) * tf.pow(y_pred_prob, self.gamma) * tf.math.log(1 - y_pred_prob)

        focal = y_true * focal_pos + (1 - y_true_tol) * focal_neg
        focal_loss = tf.reduce_mean(focal)

        y_true_f = tf.reshape(y_true, [-1])
        y_true_tol_f = tf.reshape(y_true_tol, [-1])
        y_pred_f = tf.reshape(y_pred_prob, [-1])

        tp = tf.reduce_sum(y_true_f * y_pred_f)
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
        fp = tf.reduce_sum((1 - y_true_tol_f) * y_pred_f)

        dice = (2 * tp + self.smooth) / (2 * tp + fn + fp + self.smooth)
        dice_loss = 1 - dice

        return 0.5 * focal_loss + 1.5 * dice_loss
