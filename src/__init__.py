from .hillshpaeDataModule import HillshapeDataModule
from .satelliteDataModule import SatelliteDataModule

datamodule_model = {
    'satellite_rgb': SatelliteDataModule,
    'satellite_hill': HillshapeDataModule
}