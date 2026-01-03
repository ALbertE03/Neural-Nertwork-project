import torch
import torch.optim as optim
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


class FinetuneDataset(PGNDataset):
    """
    Dataset para Fine-tuning que carga datos desde directorios de archivos múltiples.
    """
    def __init__(self, vocab, MAX_LEN_SRC, MAX_LEN_TGT, src_dir, tgt_dir):
        self.vocab = vocab
        self.MAX_LEN_SRC = MAX_LEN_SRC
        self.MAX_LEN_TGT = MAX_LEN_TGT
        self.is_tokenized = False 
        
        self.PAD_ID = self.vocab.word2id(self.vocab.pad_token)
        self.SOS_ID = self.vocab.word2id(self.vocab.start_decoding)
        self.EOS_ID = self.vocab.word2id(self.vocab.end_decoding)
        self.UNK_ID = self.vocab.word2id(self.vocab.unk_token)

        self.src_lines = []
        self.tgt_lines = []

        # Listar archivos
        if not os.path.exists(src_dir):
             raise FileNotFoundError(f"Directory not found: {src_dir}")

        src_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.src.txt')])
        
        print(f"Escaneando {len(src_files)} archivos en {src_dir}...")
        
        for src_file in src_files:
            # Extraer ID: data_001.src.txt -> 001
            try:
                # Asumiendo formato data_XXX.src.txt
                parts = src_file.split('_')
                if len(parts) < 2: continue
                
                file_id = parts[1].split('.')[0]
                tgt_file = f"target_{file_id}.tgt.txt"
                
                src_path = os.path.join(src_dir, src_file)
                tgt_path = os.path.join(tgt_dir, tgt_file)
                
                if os.path.exists(tgt_path):
                    with open(src_path, 'r', encoding='utf-8') as f:
                        src_content = f.readlines()
                    with open(tgt_path, 'r', encoding='utf-8') as f:
                        tgt_content = f.readlines()
                    
                    # Filtrar líneas vacías y asegurar correspondencia
                    # Asumimos correspondencia línea a línea estricta
                    if len(src_content) != len(tgt_content):
                        # Intentar sincronizar eliminando vacíos
                        src_clean = [l.strip() for l in src_content if l.strip()]
                        tgt_clean = [l.strip() for l in tgt_content if l.strip()]
                        
                        if len(src_clean) == len(tgt_clean):
                            self.src_lines.extend(src_clean)
                            self.tgt_lines.extend(tgt_clean)
                        else:
                            print(f"⚠ Saltando {src_file}: Desajuste de líneas ({len(src_content)} vs {len(tgt_content)})")
                    else:
                        self.src_lines.extend([l.strip() for l in src_content])
                        self.tgt_lines.extend([l.strip() for l in tgt_content])
                else:
                    # print(f"Warning: No target file found for {src_file}")
                    pass
            except Exception as e:
                print(f"Error processing {src_file}: {e}")

        assert len(self.src_lines) == len(self.tgt_lines)
        print(f"✓ Dataset de Fine-tuning cargado: {len(self.src_lines)} pares de oraciones")


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
        
        # Configurar Scheduler (ReduceLROnPlateau)
        # Reduce el LR si la loss de validación no mejora
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=1, 
            verbose=True,
            min_lr=1e-7
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
        """Entrena una época con Gradient Accumulation."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        accumulation_steps = self.finetune_config.get('gradient_accumulation_steps', 1)
        self.optimizer.zero_grad()  # Reset inicial
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            # Forward pass
            outputs = self.model(batch, is_training=True)
            loss = outputs['loss']
            
            # Escalar loss para acumulación
            loss = loss / accumulation_steps
            
            # Backward pass (acumula gradientes)
            loss.backward()
            
            # Paso de optimización cada N batches
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.finetune_config['grad_clip']
                )
                
                # Update weights
                self.optimizer.step()
                
                # Reset gradients
                self.optimizer.zero_grad()
            
            # Tracking (deshacer escalado para mostrar loss real)
            current_loss = loss.item() * accumulation_steps
            total_loss += current_loss
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        # Asegurar que se apliquen gradientes remanentes si el último batch no completó el ciclo
        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.finetune_config['grad_clip']
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
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
            path = os.path.join(self.checkpoint_dir, 'finetune_best-v2.pt')
            torch.save(checkpoint, path)
            print(f"✓ Mejor modelo guardado: {path}")
        
        if is_last:
            path = os.path.join(self.checkpoint_dir, 'finetune_last-v2.pt')
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
            current_lr = self.optimizer.get_learning_rate()
            
            print(f"\nEpoch {epoch + 1}/{self.finetune_config['epochs']}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Train
            train_loss = self._train_epoch(train_loader)
            
            # Validate
            val_loss = self._validate_epoch(val_loader)
            
            # Step Scheduler (solo si ya pasó el warmup)
            if epoch >= self.finetune_config.get('warmup_epochs', 0):
                self.scheduler.step(val_loss)
            
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
        'learning_rate': 0.00010,  # LR más bajo que pre-entrenamiento
        'epochs': 15,  # Menos épocas para fine-tuning
        'batch_size': 16,  # Batch size reducido para ahorrar memoria
        'gradient_accumulation_steps': 1,  # Acumular gradientes (Batch efectivo = 16)
        'grad_clip': 2.0,
        'warmup_epochs': 3,  # Poco warmup
        'freeze_encoder': True,  # True para solo entrenar decoder
        'freeze_embeddings': True,  # True para congelar embeddings
        'save_every_epoch': False,  # True para guardar cada época
        'checkpoint_dir': os.path.join(BASE_DIR, 'saved', 'finetune')
    }
    
    # 4. Cargar datasets
    print("\nCargando datasets de fine-tuning...")
    
    src_dir = os.path.join(DATA_DIR, 'outputs_src')
    tgt_dir = os.path.join(DATA_DIR, 'outputs_tgt')
    
    full_dataset = FinetuneDataset(
        vocab=vocab,
        MAX_LEN_SRC=config['src_len'],
        MAX_LEN_TGT=config['tgt_len'],
        src_dir=src_dir,
        tgt_dir=tgt_dir
    )
    
    # Split train/val (90/10)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # Usar generador determinista
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"✓ Total: {len(full_dataset)} ejemplos")
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
    model_path = os.path.join('/Users/alberto/Desktop/Neural-Nertwork-project/nn/PNL/saved/working/checkpoint_best2.pt')
        

   
    # 6. Crear fine-tuner y ejecutar
    finetuner = FineTuner(config, vocab, model_path, finetune_config)
    finetuner.finetune(train_loader, val_loader)


if __name__ == "__main__":
    main()
