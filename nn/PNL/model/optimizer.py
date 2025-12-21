import torch
import torch.optim as optim
import math

class ScheduledOptimizer:
    """
    Optimizer con learning rate scheduling para Pointer-Generator Network.
    Implementa:
    - Warm-up lineal
    - Decaimiento (opcional)
    - Gradient clipping
    """
    
    def __init__(self, optimizer, config):
        """
        Args:
            optimizer: PyTorch optimizer (Adam, SGD, etc.)
            config: Config object con hiperparámetros
        """
        self.optimizer = optimizer
        self.config = config
        
        self.initial_lr = config['learning_rate']
        self.current_lr = config['learning_rate']
        self.warmup_epochs = config['warmup_epochs'] if config['warmup_epochs'] else 0
        self.grad_clip = config['grad_clip']
        
        self.current_epoch = 0
        self.current_step = 0
    
    def step(self):
        """Realiza un paso de optimización con gradient clipping."""
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self._get_parameters(),
                self.grad_clip
            )
        
        self.optimizer.step()
        self.current_step += 1
    
    def zero_grad(self):
        """Resetea los gradientes."""
        self.optimizer.zero_grad()
    
    def update_learning_rate(self, epoch):
        """
        Actualiza el learning rate basado en el epoch actual.
        
        Args:
            epoch: Epoch actual (0-indexed)
        """
        self.current_epoch = epoch
        
        # Warm-up: incremento lineal del learning rate
        if epoch < self.warmup_epochs:
            # LR crece linealmente de 0 a initial_lr
            self.current_lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Después del warm-up, mantener o decrementar
            # Opción 1: Mantener constante
            self.current_lr = self.initial_lr
            
            # Opción 2: Decaimiento exponencial (comentado por defecto)
            # decay_rate = 0.5
            # decay_epochs = 5
            # epochs_after_warmup = epoch - self.warmup_epochs
            # self.current_lr = self.initial_lr * (decay_rate ** (epochs_after_warmup / decay_epochs))
        
        # Aplicar nuevo learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
    
    def get_learning_rate(self):
        """Retorna el learning rate actual."""
        return self.current_lr
    
    def _get_parameters(self):
        """Obtiene todos los parámetros del optimizer."""
        params = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group['params'])
        return params
    
    def state_dict(self):
        """Guarda el estado del optimizer."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'current_lr': self.current_lr
        }
    
    def load_state_dict(self, state_dict):
        """Carga el estado del optimizer."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.current_epoch = state_dict['current_epoch']
        self.current_step = state_dict['current_step']
        self.current_lr = state_dict['current_lr']


def build_optimizer(model, config):
    """
    Construye el optimizer basado en la configuración.
    
    Args:
        model: Modelo PyTorch
        config: Config object
        
    Returns:
        ScheduledOptimizer
    """
    learner_type = (config['learner'] if config['learner'] else 'adam').lower()
    learning_rate = config['learning_rate']
    
    if learner_type == 'adam':
        base_optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif learner_type == 'sgd':
        base_optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )
    elif learner_type == 'adagrad':
        base_optimizer = optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            initial_accumulator_value=0.1
        )
    else:
        raise ValueError(f"Optimizer desconocido: {learner_type}")
    
    # Envolver en ScheduledOptimizer
    scheduled_optimizer = ScheduledOptimizer(base_optimizer, config)
    
    return scheduled_optimizer
