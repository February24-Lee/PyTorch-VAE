from .hillshpaeDataModule import HillshapeDataModule
from .satelliteDataModule import SatelliteDataModule
from .allDataModule import AllDataModule

datamodule_model = {
    'satellite_rgb': SatelliteDataModule,
    'satellite_hill': HillshapeDataModule,
    'satellite_all': AllDataModule
}