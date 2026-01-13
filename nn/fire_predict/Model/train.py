import tensorflow as tf
import os
import json
from nn.fire_predict.Model.unet3d import build_unet3d
from tensorflow.keras import mixed_precision
class TverskyBCELoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.8, beta=0.2, smooth=1.0, pos_weight=2.0, name="tversky_bce_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):

        y_pred_prob = tf.nn.sigmoid(y_pred)
        
        
        y_true_tol = tf.nn.max_pool2d(y_true, ksize=5, strides=1, padding='SAME')
        

        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_true_tol_f = tf.cast(tf.reshape(y_true_tol, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred_prob, [-1]), tf.float32)

        tp = tf.reduce_sum(y_true_f * y_pred_f)
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f)) # Castigo si no ve el fuego real
        fp = tf.reduce_sum((1 - y_true_tol_f) * y_pred_f) # Castigo solo si está FUERA de la zona de 2px

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        tversky_loss = 1.0 - tversky_index

        bce = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight)
        bce_loss = tf.reduce_mean(bce)

        return (0.5 * bce_loss) + (1.5 * tversky_loss)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print('Compute dtype: ',policy.compute_dtype)
print('Variable dtype: ',policy.variable_dtype)

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


def main():
    batch_size = 32
    epochs = 50

    last_checkpoint_path = 'saved/checkpoints/last_model3.weights.h5'
    best_checkpoint_path = 'saved/checkpoints/best_fire_model3.weights.h5'
    history_path = 'training_history3.json'

    steps_per_epoch = len(train_dataset) // batch_size
    validation_steps = len(val_dataset) // batch_size

    model = build_unet3d(input_shape=(3, 256, 256, 28))

    initial_epoch = 0
    best_val_iou = -1.0
    history_cb = SaveHistoryCallback(history_path)

    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            saved_history = json.load(f)
            history_cb.history_dict = saved_history
            if saved_history:
                initial_epoch = saved_history[-1]['epoch']
                ious = [h.get('val_iou', -1.0) for h in saved_history]
                best_val_iou = max(ious)
                print(f"[*] Historial cargado. Mejor val_iou histórico: {best_val_iou:.4f}")

    if os.path.exists(last_checkpoint_path):
        print(f"[*] Reanudando desde pesos: {last_checkpoint_path}")
        model.load_weights(last_checkpoint_path)
    elif os.path.exists(best_checkpoint_path):
        print("[*] Usando mejor modelo previo.")
        model.load_weights(best_checkpoint_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4,
            clipnorm=1.0
        ),
        loss=TverskyBCELoss(),
        metrics=[tf.keras.metrics.BinaryIoU(
            target_class_ids=[1],
            name="iou"
        ),tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")]
    )

    best_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint_path,
        monitor='val_iou',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    if best_val_iou != -1.0:
        best_ckpt_callback.best = best_val_iou

    callbacks = [
        best_ckpt_callback,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=last_checkpoint_path,
            save_best_only=False,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            'training_log3.csv',
            append=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        ),
        history_cb
    ]

    print(f"\nIniciando entrenamiento: Época {initial_epoch + 1}")

    model.fit(
        train_tf_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_tf_ds,
        validation_steps=validation_steps,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )