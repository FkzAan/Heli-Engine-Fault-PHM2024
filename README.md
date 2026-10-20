# Helicopter Engine Fault Detection (PHM 2024)

FYP: Machine Learning-Based Fault Detection for Helicopter Turboshaft Engines Using Sensor Data.

## Folders
- dataset/    put X_train.csv and y_train.csv here (from the PHM 2024 Data Challenge)
- notebooks/  01_phm2024_analysis.ipynb
- figures/    charts saved by the notebook
- model/      trained models (later)
- prototype/  Streamlit app (later)

## Dataset
PHM North America 2024 Conference Data Challenge (PHM Society).
Only the labelled training files are used: X_train.csv (7 sensors) and y_train.csv (column `faulty`: 0 = normal, 1 = faulty).
Do not use `trq_margin` as an input: the label is calculated from it.
