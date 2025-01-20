Music Genre Identification AI Model 🎵

This project is a deep learning-based Music Genre Identification model that classifies audio files into one of the following genres:

Classical,
Country,
Disco,
Hip-Hop,
Jazz,
Metal,
Pop,
Reggae,
Rock.

The model utilizes a 2D Convolutional Neural Network (CNN) implemented using TensorFlow/Keras, trained on spectrogram images generated from audio files.



🚀 Features

Preprocessing: Converts audio files into spectrogram images.

2D CNN Architecture: Includes convolutional, pooling, dropout, and dense layers in a sequential model.

Genre Classification: Predicts the music genre from spectrograms.



Dataset Setup:

Place your audio files in the dataset/ folder under the respective genre subfolders.

Generate Spectrograms:

The create_spectrogram function uses librosa architecture to read the auido files, and converts them into spectrograms to extract features.

Save Spectrograms :

This will save spectrograms into respective folders, organized by genre.



📊 Model Architecture

The model is built using TensorFlow/Keras and has the following layers:


Input Layer: Takes the spectrogram images as input.

Convolutional Layers: Extract spatial features from spectrograms.

MaxPooling Layers: Reduces the spatial dimensions of the feature maps.

Dropout Layers: Prevents overfitting by randomly dropping neurons during training.

Fully Connected (Dense) Layers: Classifies the extracted features into genres.

Model Summary

Layer	Output Shape	Parameters

Conv2D + ReLU	(128, 128, 32)	896

MaxPooling2D	(64, 64, 32)	0

Conv2D + ReLU	(64, 64, 64)	18,496

MaxPooling2D	(32, 32, 64)	0

Dropout (0.25)	(32, 32, 64)	0

Flatten	(65536)	0

Dense + ReLU	(128)	8,388,864

Dropout (0.5)	(128)	0

Dense (Output)	(9)	1,161



🎓 Train the Model

Train the CNN model using the prepared spectrogram dataset:



🔍 Predict Genre

Use the trained model to predict the genre of a new audio file:



📈 Results

The model achieved the following accuracy on the train set-

Training Accuracy: 100%



📝 Requirements

Python 3.8+

TensorFlow 2.0+

Librosa

Matplotlib

NumPy

Pandas



🙌 Acknowledgments

Libraries: TensorFlow, Librosa, and Matplotlib.



📧 Contact

For questions or suggestions, please contact mulkraj.77@gmail.com.
