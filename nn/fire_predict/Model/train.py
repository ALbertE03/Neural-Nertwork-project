import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print('Compute dtype: ',policy.compute_dtype)
print('Variable dtype: ',policy.variable_dtype)

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

        tp = tf.reduce_sum(y_true_f * y_pred_tol_f)        
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_tol_f))    
        fp = tf.reduce_sum((1 - y_true_tol_f) * y_pred_f) 

        dice = (2 * tp + self.smooth) / (2 * tp + fn + fp + self.smooth)
        dice_loss = 1 - dice

        return 0.5 * focal_loss + 1.5 * dice_loss

class SaveHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.history_dict = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(self.model.optimizer.learning_rate.numpy())

        current_logs = {
            "epoch": epoch + 1,
            "lr": lr,
        **{k: float(v) for k, v in logs.items()}}
        self.history_dict.append(current_logs)
        
        with open(self.filepath, 'w') as f:
            json.dump(self.history_dict, f, indent=4)


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


class VisualVerifyCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, save_dir='visual_logs'):
        super().__init__()
        self.dataset = dataset
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
     
        for inputs, targets in self.dataset.take(1):
            preds = self.model.predict(inputs, verbose=0)
            preds = tf.nn.sigmoid(preds).numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            

            axes[0].imshow(inputs[0, -1, :, :, 0], cmap='viridis')
            axes[0].set_title("Input (Last Frame)")
            
            # Máscara Real
            axes[1].imshow(targets[0, :, :, 0], cmap='inferno')
            axes[1].set_title("Ground Truth")
            
            # Predicción
            axes[2].imshow(preds[0, :, :, 0], cmap='inferno')
            axes[2].set_title(f"Prediction (Epoch {epoch+1})")
            
            for ax in axes: ax.axis('off')
            
            plt.savefig(f"{self.save_dir}/epoch_{epoch+1}.png")
            plt.close()
            
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
        

def main():
    batch_size = 8 
    epochs = 200
    os.makedirs('saved/checkpoints', exist_ok=True)

    last_checkpoint_path = 'saved/checkpoints/last_model_convlstm7.weights.h5'
    best_checkpoint_path = 'saved/checkpoints/best_fire_model7_convlstm.weights.h5'
    history_path = 'training_history7.json'
    

    metrics = [
        TolerantRecall(name="tol_recall"),
        TolerantFalseNegatives(name="tol_fn"),
        TolerantF1Score(name="f1_score"), 
        tf.keras.metrics.BinaryIoU(target_class_ids=[1], name="iou"), # IoU para clase fuego
        tf.keras.metrics.Precision(name="precision")
    ]


    model = build_convlstm_bottleneck128(input_shape=(3, 256, 256, 28))

    initial_epoch = 0
    best_val_f1 = -1.0 
    history_cb = SaveHistoryCallback(history_path)

    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                saved_history = json.load(f)
                history_cb.history_dict = saved_history
                if saved_history:
                    initial_epoch = saved_history[-1]['epoch']
        
                    f1_values = [h.get('val_f1_score', -1.0) for h in saved_history]
                    best_val_f1 = max(f1_values)
                    print(f"[*] Historial cargado. Reanudando en época {initial_epoch + 1}")
                    print(f"[*] Mejor val_f1 histórico: {best_val_f1:.4f}")
        except Exception as e:
            print(f"[!] Error cargando historial: {e}")

    if os.path.exists(last_checkpoint_path):
        print(f"[*] Cargando pesos recientes: {last_checkpoint_path}")
        model.load_weights(last_checkpoint_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=FocalDiceLoss(),
        metrics=metrics
    )


    best_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint_path,
        monitor='val_f1_score', 
        mode='max',            
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    if best_val_f1 != -1.0:
        best_ckpt_callback.best = best_val_f1

    visual_cb = VisualVerifyCallback(val_tf_ds)

    callbacks = [
        best_ckpt_callback,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=last_checkpoint_path,
            save_best_only=False,
            save_weights_only=True,
            verbose=0 
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        history_cb,
        visual_cb
    ]

    print(f"\nIniciando entrenamiento ConvLSTM")

    steps_per_epoch = len(train_dataset) // batch_size
    validation_steps = len(val_dataset) // batch_size
    
    model.fit(
        train_tf_ds,
        steps_per_epoch=steps_per_epoch,     
        validation_data=val_tf_ds,
        validation_steps=validation_steps,   
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )