from mobilevideogpt.model.videomamba import build_videomamba
from mobilevideogpt.model.videomamba.config import config_dict
from mobilevideogpt.model.videomamba.utils import ConfigWrapper, setup_videomamba

def build_videomamba(model_path):
    config = ConfigWrapper(config_dict)
    # Skip pretrained weight loading during model construction.
    # from_pretrained's safetensors already contains the correct weights
    # and will overwrite them anyway. Skipping saves ~1.2GB peak memory
    # which is critical on memory-constrained devices like Jetson (8GB).
    config.vision_encoder.pretrained = None
    videomamba_model = setup_videomamba(config)
    return videomamba_model

