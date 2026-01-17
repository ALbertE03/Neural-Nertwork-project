import tenaoereflow as tf

class TolerantRecall(tf.keras.metrics.Metric):
    def __init__(self, tol_ksize=5, threshold=0.5, name="tolerant_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tol_ksize = tol_ksize
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.nn.sigmoid(y_pred)
        y_pred_bin = tf.cast(y_pred > self.threshold, tf.float32)

        tp = tf.reduce_sum(y_pred_bin * y_true)
        y_pred_tol = tf.nn.max_pool2d(y_pred_bin, ksize=self.tol_ksize, strides=1, padding='SAME')
        fn = tf.reduce_sum(y_true * (1.0 - y_pred_tol))

        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        return self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)



class TolerantFalseNegatives(tf.keras.metrics.Metric):
    def __init__(self, tol_ksize=5, threshold=0.5, name="tolerant_fn", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tol_ksize = tol_ksize
        self.threshold = threshold

        self.fn = self.add_weight(name="fn", initializer="zeros", dtype=tf.float32)
        self.pos = self.add_weight(name="pos", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)


        y_pred_bin = tf.cast(tf.nn.sigmoid(y_pred) > self.threshold, tf.float32)

        y_pred_tol = tf.nn.max_pool2d(
            y_pred_bin,
            ksize=self.tol_ksize,
            strides=1,
            padding='SAME'
        )

        fn = tf.reduce_sum(y_true * (1.0 - y_pred_tol))
        pos = tf.reduce_sum(y_true)

        self.fn.assign_add(fn)
        self.pos.assign_add(pos)

    def result(self):
        return self.fn / (self.pos + tf.keras.backend.epsilon())

    def reset_state(self):
        self.fn.assign(0.0)
        self.pos.assign(0.0)

            
class TolerantF1Score(tf.keras.metrics.Metric):
    def __init__(self, tol_ksize=5, threshold=0.5, name="tolerant_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tol_ksize = tol_ksize
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(tf.nn.sigmoid(y_pred) > self.threshold, tf.float32)
        
        y_true_tol = tf.nn.max_pool2d(y_true, ksize=self.tol_ksize, strides=1, padding='SAME')
   
        y_pred_tol = tf.nn.max_pool2d(y_pred_bin, ksize=self.tol_ksize, strides=1, padding='SAME')

        tp = tf.reduce_sum(y_pred_bin * y_true)
        fp = tf.reduce_sum(y_pred_bin * (1.0 - y_true_tol))
        fn = tf.reduce_sum(y_true * (1.0 - y_pred_tol))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        
        f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
        return f1

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
      