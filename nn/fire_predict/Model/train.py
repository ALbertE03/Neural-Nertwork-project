import tensorflow as tf
import json
import os
from tensorflow.keras import mixed_precision
from callbacks import SaveHistoryCallback,VisualVerifyCallback
from loss import FocalDiceLoss
from constants import BATCH_SIZE,INPUT_SHAPE,DROPOUT,REDUCTION,GAMMA,ALPHA,SMOOTH,last_checkpoint_path,history_path,best_checkpoint_path
from metrics import TolerantRecall,TolerantFalseNegatives,TolerantF1Score
from model import build_convlstm_bottleneck128

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print('Compute dtype: ',policy.compute_dtype)
print('Variable dtype: ',policy.variable_dtype)

def main():
    os.makedirs('saved/checkpoints', exist_ok=True)

    # Metricas
    metrics = [
        TolerantRecall(name="tol_recall"),
        TolerantFalseNegatives(name="tol_fn"),
        TolerantF1Score(name="f1_score"), 
        tf.keras.metrics.BinaryIoU(target_class_ids=[1], name="iou"), 
        tf.keras.metrics.Precision(name="precision")
    ]


    model = build_convlstm_bottleneck128(input_shape=INPUT_SHAPE,dropout=DROPOUT,reduction=REDUCTION)

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
        loss=FocalDiceLoss(gamma=GAMMA,alpha=ALPHA,smooth=SMOOTH),
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

    callbacks = [
        best_ckpt_callback,
        history_cb,
        VisualVerifyCallback(val_tf_ds),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=last_checkpoint_path,
            save_best_only=False,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print(f"\nIniciando entrenamiento")

    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    validation_steps = len(val_dataset) // BATCH_SIZE
    
    model.fit(
        train_tf_ds,
        steps_per_epoch=steps_per_epoch,     
        validation_data=val_tf_ds,
        validation_steps=validation_steps,   
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )