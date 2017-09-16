# ML-weld-break-prediction-
Machine learning for weld break prediction.


In this project machine learning is used to predict whether the welded coil will break or not from the data gathered during the welding process.

The data consists of fixed and time-series data. For the time-series data, different features are extracted based on the context of the data being represented.

Traditionally, weld joints are analysed using the formulae for weld strength, strength of connection, load capacity of the weld based on the properties of the metal being welded and that of the weld material.

However, using machine learning makes the process of predicting sucess of weld more robust to various factors that are not considered in empirical formulae and also makes it possible to have a real-time prediction to assist the weld process.

The file 'build_features.py' extracts meaningful features from the raw data both from fixed and time-series observations for each sample.
'main.py' uses a simple SVM learning model to identify whether the weld will break or not.
