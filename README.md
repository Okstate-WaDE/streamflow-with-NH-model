# Streamflow Predictions with NeuralHydrology
This project aims to combine Kratzert et al.'s research into predicting streamflow with LSTMs with online open-source data.
### Download
Unfortunately, there is no way to download just one of CAMELS, CAMGEN, or GENERIC (yet). To use this project, clone or download the entire repository.

## CAMELS
### Data Setup
CAMELS requires no data setup.
### Configuration
Certain fields in `nh_model/config.yml` require changes. They all have comments, I'll also list them here later.
### Running the Model
Use the command `nh-run evaluate --run-dir /PATH/TO/CAMELS/nh_model` to run on CAMELS.

## CAMGEN
### Data Setup
CAMGEN requires no data setup. However, the `import_camels.py` file, which was used to create CAMGEN, is available. If you want to re-run it, make sure CAMGEN and CAMELS are in the same directory.
### Configuration
Certain fields in `nh_model/config.yml` require changes. They all have comments, I'll also list them here later.
### Running the Model
Use the command `nh-run evaluate --run-dir /PATH/TO/CAMGEN/nh_model` to run on CAMGEN.

## GENERIC
### Data Setup
GENERIC comes with the data for two basins: Piscataquis River near Dover-Foxcroft, ME (USGS-01031500) and Washita River near Pauls Valley, OK (USGS-07328500). To load the data for other basins, list their USGS gauge IDs (line-by-line, should be 8 digits, leave out the USGS- prefix) in `basin_list.txt` and then run `setup_gridmet.py`.
### Configuration
Certain fields in `nh_model/config.yml` require changes. They all have comments, I'll also list them here later.
### Running the Model
Use the command `nh-run evaluate --run-dir /PATH/TO/GENERIC/nh_model` to run on GENERIC.