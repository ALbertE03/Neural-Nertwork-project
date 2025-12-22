import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import json

from pgn_model import PointerGeneratorNetwork
from optimizer import build_optimizer
from beam_search import BeamSearch
from dataset import PGNDataset, pgn_collate_fn
from vocabulary import Vocabulary
from config import Config
from constant import *

class Trainer:
    """
    Clase para entrenar el modelo Pointer-Generator Network.
    """
    
    def __init__(self, config, vocab):
        """
        Args:
            config: Config object
            vocab: Vocabulary object
        """
        self.config = config
        self.vocab = vocab
        self.device = config['device']
        
        # Modelo
        self.model = PointerGeneratorNetwork(config, vocab).to(self.device)
        
        # Optimizer
        self.optimizer = build_optimizer(self.model, config)
        
        # Beam search para validación
        self.beam_search = BeamSearch(
            self.model,
            vocab,
            beam_size=config['beam_size'],
            max_len=config['tgt_len']
        )
        
        # Tracking
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # History
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'vocab_loss': [],
            'coverage_loss': [],
            'val_loss': []
        }
        
        # Paths
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self, train_loader, val_loader, num_epochs):
        """
        Entrena el modelo.
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            num_epochs: Número de épocas
        """
        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Épocas: {num_epochs}")
        print(f"Batch size: {self.config['train_batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Pointer-Generator: {self.config['is_pgen']}")
        print(f"Coverage: {self.config['is_coverage']}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Actualizar learning rate
            self.optimizer.update_learning_rate(epoch)
            current_lr = self.optimizer.get_learning_rate()
            
            print(f"\n--- Epoch {epoch+1}/{num_epochs} (LR: {current_lr:.6f}) ---")
            
            # Training
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Guardar history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['vocab_loss'].append(train_metrics['vocab_loss'])
            self.train_history['coverage_loss'].append(train_metrics['coverage_loss'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            
            # Guardar checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            save_model = self.config['save_model_epoch'] if self.config['save_model_epoch'] is not None else True
            if save_model:
                self._save_checkpoint(epoch, is_best)
            
            # Tiempo
            epoch_time = time.time() - epoch_start_time
            print(f"Tiempo de época: {epoch_time/60:.2f} min")
            
        # Guardar history final
        save_hist = self.config['save_history'] if self.config['save_history'] is not None else True
        if save_hist:
            self._save_history()
        
        print(f"\n{'='*60}")
        print(f"Entrenamiento completado!")
        print(f"Mejor Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
    
    def _train_epoch(self, train_loader, epoch):
        """Entrena una época."""
        self.model.train()
        
        total_loss = 0.0
        total_vocab_loss = 0.0
        total_coverage_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Training", total=len(train_loader))
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch, is_training=True)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Optimizer step (con gradient clipping)
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            total_vocab_loss += outputs['vocab_loss'].item()
            total_coverage_loss += outputs['coverage_loss'].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'vocab': f"{outputs['vocab_loss'].item():.4f}",
                'cov': f"{outputs['coverage_loss'].item():.4f}"
            })
            
            # Limitar iteraciones por época si está configurado
            iters_per_epoch = self.config['iters_per_epoch']
            if iters_per_epoch and batch_idx >= iters_per_epoch:
                break
        
        pbar.close()
        
        # Promedios
        avg_loss = total_loss / num_batches
        avg_vocab_loss = total_vocab_loss / num_batches
        avg_coverage_loss = total_coverage_loss / num_batches
        
        print(f"Train Loss: {avg_loss:.4f} | "
              f"Vocab: {avg_vocab_loss:.4f} | "
              f"Coverage: {avg_coverage_loss:.4f}")
        
        return {
            'loss': avg_loss,
            'vocab_loss': avg_vocab_loss,
            'coverage_loss': avg_coverage_loss
        }
    
    def _validate_epoch(self, val_loader, epoch):
        """Valida el modelo."""
        self.model.eval()
        
        total_loss = 0.0
        total_vocab_loss = 0.0
        total_coverage_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation", total=len(val_loader))
            
            for batch in pbar:
                if batch is None:
                    continue
                outputs = self.model(batch, is_training=True)
                
                total_loss += outputs['loss'].item()
                total_vocab_loss += outputs['vocab_loss'].item()
                total_coverage_loss += outputs['coverage_loss'].item()
                num_batches += 1
                
                pbar.set_postfix({
                    'val_loss': f"{outputs['loss'].item():.4f}"
                })
            
            pbar.close()
        
        # Promedios
        avg_loss = total_loss / num_batches
        avg_vocab_loss = total_vocab_loss / num_batches
        avg_coverage_loss = total_coverage_loss / num_batches
        
        print(f"Val Loss: {avg_loss:.4f} | "
              f"Vocab: {avg_vocab_loss:.4f} | "
              f"Coverage: {avg_coverage_loss:.4f}")
        
        return {
            'loss': avg_loss,
            'vocab_loss': avg_vocab_loss,
            'coverage_loss': avg_coverage_loss
        }
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Guarda un checkpoint del modelo."""
        checkpoint = {
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.get_config_dict(),
            'train_history': self.train_history
        }
        
        # Guardar checkpoint de época (SIEMPRE)
        epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, epoch_path)
        print(f"✓ Guardado checkpoint época {epoch+1}")
        
        # Guardar último checkpoint (sobreescribe)
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_last.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar mejor checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Guardado mejor modelo (Val Loss: {self.best_val_loss:.4f})")
    
    def _save_history(self):
        """Guarda el historial de entrenamiento."""
        history_path = os.path.join(self.checkpoint_dir, 'train_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"✓ Historial guardado en {history_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Carga un checkpoint.
        
        Args:
            checkpoint_path: Ruta al checkpoint
        """
        if not os.path.exists(checkpoint_path):
            print(f"⚠ Checkpoint no encontrado: {checkpoint_path}")
            return
        
        print(f"Cargando checkpoint desde {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint.get('train_history', self.train_history)
        
        print(f"✓ Checkpoint cargado (Epoch {self.start_epoch}, Step {self.global_step})")
    
    def find_latest_checkpoint(self):
        """Busca el checkpoint más reciente."""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        # Buscar checkpoint_last.pt primero
        last_checkpoint = os.path.join(self.checkpoint_dir, 'checkpoint_last.pt')
        if os.path.exists(last_checkpoint):
            return last_checkpoint
        
        # Si no existe, buscar el checkpoint de época más reciente
        epoch_checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt'):
                try:
                    epoch_num = int(filename.replace('checkpoint_epoch_', '').replace('.pt', ''))
                    epoch_checkpoints.append((epoch_num, os.path.join(self.checkpoint_dir, filename)))
                except ValueError:
                    continue
        
        if epoch_checkpoints:
            # Retornar el de mayor número de época
            epoch_checkpoints.sort(reverse=True)
            return epoch_checkpoints[0][1]
        
        return None


def main():
    """
    Función principal para entrenar el modelo.
    """
    # 1. Configurar reproducibilidad
    if REPRODUCIBILITY:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 2. Crear vocabulario
    print("Construyendo vocabulario...")
    vocab = Vocabulary(
        CREATE_VOCABULARY=CREATE_VOCABULARY,
        PAD_TOKEN=PAD_TOKEN,
        UNK_TOKEN=UNK_TOKEN,
        START_DECODING=START_DECODING,
        END_DECODING=END_DECODING,
        MAX_VOCAB_SIZE=MAX_VOCAB_SIZE,
        CHECKPOINT_VOCABULARY_DIR=CHECKPOINT_VOCABULARY_DIR,
        DATA_DIR=DATA_DIR,
        VOCAB_NAME=VOCAB_NAME
    )
    vocab.build_vocabulary()
    print(f"✓ Vocabulario construido: {vocab.total_size()} palabras")
    
    # 3. Configurar modelo
    config = Config(
        max_vocab_size=vocab.total_size(),
        src_len=MAX_LEN_SRC,
        tgt_len=MAX_LEN_TGT,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_enc_layers=NUM_ENC_LAYERS,
        num_dec_layers=NUM_DEC_LAYERS,
        use_gpu=USE_GPU,
        is_pgen=IS_PGEN,
        is_coverage=IS_COVERAGE,
        coverage_lambda=COV_LOSS_LAMBDA,
        grad_clip=GRAD_CLIP,
        epochs=EPOCHS,
        data_path=DATA_DIR,
        generated_text_dir=GENERATED_TEXT_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        reproducibility=REPRODUCIBILITY,
        plot=PLOT,
        dropout_ratio=DROPOUT_RATIO,
        bidirectional=BIDIRECTIONAL,
        save_history=SAVE_HISTORY,
        save_model_epoch=SAVE_MODEL_EPOCH,
        seed=SEED,
        device=DEVICE,
        decoding_strategy=DECODING_STRATEGY,
        beam_size=BEAM_SIZE,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        learner=LEARNER,
        learning_rate=LEARNING_RATE,
        iters_per_epoch=ITERS_PER_EPOCH,
        gpu_id=GPU_ID,
        warmup_epochs=WARMUP_EPOCHS
    )
    
    print(config)
    
    # 4. Crear datasets
    print("\nCargando datasets...")
    train_dataset = PGNDataset(
        vocab=vocab,
        MAX_LEN_SRC=config['src_len'],
        MAX_LEN_TGT=config['tgt_len'],
        data_dir=config['data_path'],
        split='train'
    )
    
    val_dataset = PGNDataset(
        vocab=vocab,
        MAX_LEN_SRC=config['src_len'],
        MAX_LEN_TGT=config['tgt_len'],
        data_dir=config['data_path'],
        split='val'
    )
    
    print(f"✓ Train: {len(train_dataset)} ejemplos")
    print(f"✓ Val: {len(val_dataset)} ejemplos")
    
    # 5. Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        collate_fn=pgn_collate_fn,
        num_workers=0,  # Ajustar según tu sistema
        pin_memory=True if USE_GPU else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        collate_fn=pgn_collate_fn,
        num_workers=0,
        pin_memory=True if USE_GPU else False
    )
    
    # 6. Crear trainer
    trainer = Trainer(config, vocab)
    
    # 7. Buscar y cargar checkpoint automáticamente
    latest_checkpoint = trainer.find_latest_checkpoint()
    if latest_checkpoint:
        print(f"\n🔄 Checkpoint encontrado: {latest_checkpoint}")
        trainer.load_checkpoint(latest_checkpoint)
        print(f"Continuando desde época {trainer.start_epoch}\n")
    else:
        print("\n🆕 No se encontró checkpoint previo. Iniciando entrenamiento desde cero.\n")
    
    # 8. Entrenar
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['epochs']
    )


if __name__ == "__main__":
    main()
