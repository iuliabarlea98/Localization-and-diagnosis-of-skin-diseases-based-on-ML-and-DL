# Localization-and-diagnosis-of-skin-diseases-based-on-ML-and-DL
University-Thesis project
-get HAM10000 dataset, available at https://challenge.isic-archive.com/data
-This project has two purposes: 1.Pixel classification achieved through a segmentation algorithm built using various filters for feature extraction, YCbCr color space conversion and Random Forest Classifier (Machine Learning), 2.Dataset classification into corresponding diseases classes using a Convolutional Neural Networks model
-The classification is integrated into a GUI (Tkinter) that simulates a computer aided examination
-The interface is user-friedly, has a few buttons
-The first button allows you to choose a picture from a directory. After pressing it, the model is predicting the result.
-Another button has the role to pop up a top-level window for recommendations about that specific disease
-A button to another top-level window used for registering, the data is stored into a JSON file
-Exit button
