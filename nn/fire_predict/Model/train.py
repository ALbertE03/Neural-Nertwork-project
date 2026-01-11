import tensorflow as tf
import os
import json

from tensorflow.keras import mixed_precision
class BCEDiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, name="bce_dice_loss"):
        super().__init__(name=name)
        self.smooth = smooth
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, y_true, y_pred):
        # BCE Loss
        bce_loss = self.bce(y_true, y_pred)
        
        # Dice Loss
        y_pred_prob = tf.nn.sigmoid(y_pred)
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred_prob, [-1])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth
        )
        
        return bce_loss + dice_loss
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

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
            **{k: float(v) for k, v in logs.items()} 
        }
        self.history_dict.append(current_logs)
        
        with open(self.filepath, 'w') as f:
            json.dump(self.history_dict, f, indent=4)


def main():
    batch_size = 8  
    epochs = 50
    
    last_checkpoint_path = 'saved/checkpoints/last_model1.weights.h5' 
    best_checkpoint_path = 'saved/checkpoints/best_fire_model1.weights.h5' 
    history_path = 'training_history.json'
    
    steps_per_epoch = len(train_dataset) // batch_size
    validation_steps = len(val_dataset) // batch_size

    
    model = build_unet3d(input_shape=(3, 256, 256, 28))
    
    
    initial_epoch = 0
    history_cb = SaveHistoryCallback(history_path)

    if os.path.exists(last_checkpoint_path):
        print(f"\n[*] Reanudando desde el último estado: {last_checkpoint_path}")
        model.load_weights(last_checkpoint_path)
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                saved_history = json.load(f)
                history_cb.history_dict = saved_history
                if saved_history:
                    initial_epoch = saved_history[-1]['epoch']
                    print(f"[*] Continuando desde la época {initial_epoch}")
    elif os.path.exists(best_checkpoint_path):
        print(f"\n[*] No hay 'last_model', pero cargando 'best_fire_model' para no empezar de cero.")
        model.load_weights(best_checkpoint_path)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, clipnorm=1.0),
        loss=BCEDiceLoss(),
        metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], name="iou")]
    )


    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor='val_iou',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=last_checkpoint_path,
            save_best_only=False, 
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('training_log1.csv', append=True), 
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        history_cb
    ]

    print(f"\nIniciando entrenamiento: Época {initial_epoch + 1} de {epochs}")
    
    model.fit(
        train_tf_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_tf_ds,
        validation_steps=validation_steps,
        epochs=epochs,
        initial_epoch=initial_epoch, 
        callbacks=callbacks
    )



if __name__ == "__main__":
    main() 