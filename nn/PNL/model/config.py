import torch
class Config:
    def __init__(
        self,
        **kwargs
    ):
        self.config_dict = {
            **kwargs
        }

        self._init_device()
        
    def _init_device(self):
        if self.config_dict["use_gpu"] and torch.cuda.is_available():
            self.config_dict["device"] = torch.device(
                f"cuda:{self.config_dict['gpu_id']}"
            )
        else:
            self.config_dict["device"] = torch.device("cpu")

    def __getitem__(self, item):
        return self.config_dict.get(item, None)
    
    def get_config_dict(self):
        return self.config_dict
        
    def __str__(self):
        args_info = "\nHyper Parameters:\n"
        for key, value in self.config_dict.items():
            args_info += f"{key}={value}\n"
        return args_info

    def __repr__(self):
        return self.__str__()
