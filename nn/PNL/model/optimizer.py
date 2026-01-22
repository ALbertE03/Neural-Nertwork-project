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
        Solo maneja el Warm-up. Después del warm-up, deja que un Scheduler externo (ej: ReduceLROnPlateau)
        maneje el LR.
        
        Args:
            epoch: Epoch actual (0-indexed)
        """
        self.current_epoch = epoch
        
        # Warm-up: incremento lineal del learning rate
        if epoch < self.warmup_epochs:
            # LR crece linealmente de 0 a initial_lr
            self.current_lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
            
            # Aplicar nuevo learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
        
        # Si no es warmup, NO tocamos el LR aquí para no interferir con ReduceLROnPlateau
    
    def get_learning_rate(self):
        """Retorna el learning rate actual del optimizador."""
        # Devolver el LR real del optimizador (por si lo cambió un scheduler externo)
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
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


def build_optimizer(model, config=None, learner=None, lr=None, warmup_epochs=None, grad_clip=None):
    """
    Construye el optimizer basado en la configuración.
    
    Args:
        model: Modelo PyTorch
        config: Config object (opcional)
        learner: Tipo de optimizador (opcional, override config)
        lr: Learning rate (opcional, override config)
        warmup_epochs: Epochs de warmup (opcional, override config)
        grad_clip: Gradient clipping (opcional, override config)
        
    Returns:
        ScheduledOptimizer
    """
    # Resolver parámetros (prioridad: argumentos explícitos > config > defaults)
    if config is not None:
        _learner = learner if learner is not None else config.get('learner', 'adam')
        _lr = lr if lr is not None else config.get('learning_rate', 0.001)
        _warmup = warmup_epochs if warmup_epochs is not None else config.get('warmup_epochs', 0)
        _grad_clip = grad_clip if grad_clip is not None else config.get('grad_clip', 0.0)
    else:
        _learner = learner if learner is not None else 'adam'
        _lr = lr if lr is not None else 0.001
        _warmup = warmup_epochs if warmup_epochs is not None else 0
        _grad_clip = grad_clip if grad_clip is not None else 0.0

    learner_type = _learner.lower()
    
    if learner_type =='adagrad':
        base_optimizer = optim.Adagrad(
            model.parameters(),
            lr=_lr,
            initial_accumulator_value=0.1
        )
    elif learner_type == 'adam':
        base_optimizer = optim.Adam(
            model.parameters(),
            lr=_lr,
            betas=(0.9, 0.999),
            eps=1e-6
        )
    elif learner_type == 'adamw':
        base_optimizer = optim.AdamW(
            model.parameters(),
            lr=_lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.01
        )
    elif learner_type == 'sgd':
        base_optimizer = optim.SGD(
            model.parameters(),
            lr=_lr,
            momentum=0.9
        )
    else:
        raise ValueError(f"Optimizer desconocido: {learner_type}")
    
    # Config sintética para ScheduledOptimizer
    opt_config = {
        'learning_rate': _lr,
        'warmup_epochs': _warmup,
        'grad_clip': _grad_clip
    }
    
    # Envolver en ScheduledOptimizer
    scheduled_optimizer = ScheduledOptimizer(base_optimizer, opt_config)
    
    return scheduled_optimizer
