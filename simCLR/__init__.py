from .simclr_sate import SimCLR_sate
from .simclr_sate_rgb import SimCLRSateRgb
from .simclr_sate_all import SimCLRSateAll

simclr_model = {
    'simclr_sate': SimCLR_sate,
    'simclr_sate_rgb': SimCLRSateRgb,
    'simclr_sate_all':SimCLRSateAll
}