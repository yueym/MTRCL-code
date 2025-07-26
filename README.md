# A New Method for Enhancing Short-Term Wind Speed Forecast Capability of Numerical Weather Prediction

This repository is the official PyTorch implementation of the paper "A New Method for Enhancing Short-Term Wind Speed Forecast Capability of Numerical Weather Prediction".

### Data Download
1. Download the ECMWF-TIGGE data from 2021 to 2024 from the website https://apps.ecmwf.int/datasets/data/tigge/levtype=sfc/type=cf/.
2. Download the DEM data for the corresponding study area from https://doi.org/10.5067/ASTER/ASTGTM.003.
3. Download the ERA5 wind speed data from 2021 to 2024 from https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview.

### Data Preprocessing and Feature Selection
1. After data download, use the `data_alignment.py` file in the `Data_preprocessing` folder for initial data preprocessing.
2. Then, use the `data_relevance_visualization_CatBoost.py` file for feature selection.

### Model Training, Validation, and Testing
1. After feature selection, use the `train.py` file in the `Model_training` folder for model training.
2. Next, use the `validate.py` file in the `Model_validating` folder for model validation.
3. Finally, use the `test.py` file in the `Model_testing` folder to test the wind speed prediction results.

### Baselines
The `Baselines` folder contains the training, validation, and testing codes for some baselines used in this study.

### Results
The `Results` folder contains some final result graphs of this study for reference. 
