import torch
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm

from pgn_model import PointerGeneratorNetwork
from dataset import PGNDataset, pgn_collate_fn
from vocabulary import Vocabulary
from config import Config
from optimizer import build_optimizer
from constant import *


class FineTuner:
    """
    Clase para hacer fine-tuning del modelo pre-entrenado.
    """
    
    def __init__(self, config, vocab, pretrained_path, finetune_config):
        """
        Args:
            config: Config object
            vocab: Vocabulary object
            pretrained_path: Ruta al checkpoint pre-entrenado
            finetune_config: Dict con configuración de fine-tuning
        """
        self.config = config
        self.vocab = vocab
        self.device = config['device']
        self.finetune_config = finetune_config
        
        # Crear modelo
        self.model = PointerGeneratorNetwork(config, vocab).to(self.device)
        
        # Cargar pesos pre-entrenados
        print(f"Cargando modelo pre-entrenado desde {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Modelo cargado (Epoch {checkpoint['epoch']})")
        print(f"✓ Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        # Configurar optimizador para fine-tuning
        self.optimizer = build_optimizer(
            self.model,
            learner=finetune_config['optimizer'],
            lr=finetune_config['learning_rate'],
            warmup_epochs=finetune_config.get('warmup_epochs', 0)
        )
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = []
        
        # Directorio para guardar checkpoints de fine-tuning
        self.checkpoint_dir = finetune_config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"CONFIGURACIÓN DE FINE-TUNING")
        print(f"{'='*60}")
        print(f"Learning Rate: {finetune_config['learning_rate']}")
        print(f"Epochs: {finetune_config['epochs']}")
        print(f"Batch Size: {finetune_config['batch_size']}")
        print(f"Grad Clip: {finetune_config['grad_clip']}")
        print(f"Warmup Epochs: {finetune_config.get('warmup_epochs', 0)}")
        
        if finetune_config.get('freeze_encoder', False):
            print(f"⚠ Encoder congelado (sin entrenamiento)")
        if finetune_config.get('freeze_embeddings', False):
            print(f"⚠ Embeddings congelados (sin entrenamiento)")
        
        print(f"{'='*60}\n")
        
        # Congelar capas si se especifica
        self._freeze_layers()
    
    def _freeze_layers(self):
        """Congela capas según la configuración."""
        if self.finetune_config.get('freeze_embeddings', False):
            print("Congelando embeddings...")
            for param in self.model.encoder.embedding.parameters():
                param.requires_grad = False
        
        if self.finetune_config.get('freeze_encoder', False):
            print("Congelando encoder...")
            for param in self.model.encoder.parameters():
                param.requires_grad = False
    
    def _train_epoch(self, train_loader):
        """Entrena una época."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            # Forward pass
            outputs = self.model(batch, is_training=True)
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.finetune_config['grad_clip']
            )
            
            # Update
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        pbar.close()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _validate_epoch(self, val_loader):
        """Valida una época."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            
            for batch_idx, batch in enumerate(pbar):
                if batch is None:
                    continue
                
                outputs = self.model(batch, is_training=True)
                loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            pbar.close()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _save_checkpoint(self, val_loss, is_best=False, is_last=False):
        """Guarda checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'finetune_config': self.finetune_config
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'finetune_best.pt')
            torch.save(checkpoint, path)
            print(f"✓ Mejor modelo guardado: {path}")
        
        if is_last:
            path = os.path.join(self.checkpoint_dir, 'finetune_last.pt')
            torch.save(checkpoint, path)
        
        # Guardar por época si se especifica
        if self.finetune_config.get('save_every_epoch', False):
            path = os.path.join(self.checkpoint_dir, f'finetune_epoch_{self.current_epoch + 1}.pt')
            torch.save(checkpoint, path)
    
    def finetune(self, train_loader, val_loader):
        """
        Ejecuta el fine-tuning.
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
        """
        print(f"\n{'='*60}")
        print(f"INICIANDO FINE-TUNING")
        print(f"{'='*60}\n")
        
        for epoch in range(self.finetune_config['epochs']):
            self.current_epoch = epoch
            
            # Update learning rate (warmup)
            self.optimizer.update_learning_rate(epoch)
            current_lr = self.optimizer.get_current_lr()
            
            print(f"\nEpoch {epoch + 1}/{self.finetune_config['epochs']}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Train
            train_loss = self._train_epoch(train_loader)
            
            # Validate
            val_loss = self._validate_epoch(val_loader)
            
            # Log
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            
            # Save history
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            })
            
            # Save best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ✓ Nuevo mejor modelo (Val Loss: {val_loss:.4f})")
            
            self._save_checkpoint(val_loss, is_best=is_best, is_last=True)
        
        # Save final history
        history_path = os.path.join(self.checkpoint_dir, 'finetune_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✓ Historial guardado en {history_path}")
        
        print(f"\n{'='*60}")
        print(f"FINE-TUNING COMPLETADO")
        print(f"{'='*60}")
        print(f"Mejor Val Loss: {self.best_val_loss:.4f}")
        print(f"Checkpoint: {os.path.join(self.checkpoint_dir, 'finetune_best.pt')}")
        print(f"{'='*60}\n")


def main():
    """
    Función principal para fine-tuning.
    """
    # 1. Cargar vocabulario
    print("Cargando vocabulario...")
    vocab = Vocabulary(
        CREATE_VOCABULARY=False,
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
    print(f"✓ Vocabulario cargado: {vocab.total_size()} palabras")
    
    # 2. Configuración del modelo (misma que pre-entrenamiento)
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
        dropout_ratio=DROPOUT_RATIO,
        bidirectional=BIDIRECTIONAL,
        device=DEVICE,
        decoding_strategy=DECODING_STRATEGY,
        beam_size=BEAM_SIZE,
        gpu_id=GPU_ID
    )
    
    # 3. Configuración de fine-tuning
    finetune_config = {
        'optimizer': 'adam',
        'learning_rate': 0.00005,  # LR más bajo que pre-entrenamiento
        'epochs': 10,  # Menos épocas para fine-tuning
        'batch_size': 16,  # Batch size más pequeño
        'grad_clip': 2.0,
        'warmup_epochs': 1,  # Poco warmup
        'freeze_encoder': False,  # True para solo entrenar decoder
        'freeze_embeddings': False,  # True para congelar embeddings
        'save_every_epoch': False,  # True para guardar cada época
        'checkpoint_dir': os.path.join(BASE_DIR, 'saved', 'finetune')
    }
    
    # 4. Cargar datasets
    print("\nCargando datasets...")
    
    train_dataset = PGNDataset(
        vocab=vocab,
        MAX_LEN_SRC=config['src_len'],
        MAX_LEN_TGT=config['tgt_len'],
        data_dir=DATA_DIR,
        split='train'
    )
    
    val_dataset = PGNDataset(
        vocab=vocab,
        MAX_LEN_SRC=config['src_len'],
        MAX_LEN_TGT=config['tgt_len'],
        data_dir=DATA_DIR,
        split='val'
    )
    
    print(f"✓ Train: {len(train_dataset)} ejemplos")
    print(f"✓ Val: {len(val_dataset)} ejemplos")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=finetune_config['batch_size'],
        shuffle=True,
        collate_fn=pgn_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=finetune_config['batch_size'],
        shuffle=False,
        collate_fn=pgn_collate_fn,
        num_workers=0
    )
    
    # 5. Ruta del modelo pre-entrenado
    pretrained_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_best.pt')
    
    if not os.path.exists(pretrained_path):
        pretrained_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_last.pt')
    
    if not os.path.exists(pretrained_path):
        print(f"⚠ No se encontró ningún checkpoint pre-entrenado en {CHECKPOINT_DIR}")
        print("Debes entrenar el modelo primero con train.py")
        return
    
    # 6. Crear fine-tuner y ejecutar
    finetuner = FineTuner(config, vocab, pretrained_path, finetune_config)
    finetuner.finetune(train_loader, val_loader)


if __name__ == "__main__":
    main()
