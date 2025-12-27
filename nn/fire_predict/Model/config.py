from . import constants

class Config:
    def __init__(self, **kwargs):
        # Data Config
        self.data_x_path = kwargs.get('data_x_path', constants.DEFAULT_DATA_X_PATH)
        self.data_y_path = kwargs.get('data_y_path', constants.DEFAULT_DATA_Y_PATH)
        self.img_height = kwargs.get('img_height', constants.IMG_HEIGHT)
        self.img_width = kwargs.get('img_width', constants.IMG_WIDTH)
        self.input_channels = kwargs.get('input_channels', constants.INPUT_CHANNELS)
        
        # Model Config
        self.hidden_channels = kwargs.get('hidden_channels', constants.HIDDEN_CHANNELS)
        self.kernel_size = kwargs.get('kernel_size', constants.KERNEL_SIZE)
        
        # Training Config
        self.batch_size = kwargs.get('batch_size', constants.BATCH_SIZE)
        self.learning_rate = kwargs.get('learning_rate', constants.LEARNING_RATE)
        self.epochs = kwargs.get('epochs', constants.EPOCHS)
        self.input_seq_len = kwargs.get('input_seq_len', constants.INPUT_SEQ_LEN)
        self.pred_seq_len = kwargs.get('pred_seq_len', constants.PRED_SEQ_LEN)
        self.checkpoint_dir = kwargs.get('checkpoint_dir', constants.CHECKPOINT_DIR)

    def __repr__(self):
        return str(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
