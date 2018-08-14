# Freesound General-Purpose Audio Tagging Challenage

## About
This project is for the Freesound General-Purpose audio Tagging Challenage on Kaggle.

## Team member
Zongming Liu - B00784897 - zongming.liu@dal.ca

## Dependencies
numpy,
librosa,
pandas,
tensorflow,
keras

## Instructions
inputs.py - Python script to convert all audio samples in the audio_train directory to a single numpy array of log-mel spectrograms. To run it, type ```python inputs.py <the path to your audio_train directory> <the path to your train.csv file>```. After it finishes, two numpy files called 'samples.npy' and 'labels.npy' will be crated in your current directory. The first file contains the training data, the second file contains the corresponding labels for the training data.

model.py - Python script to create a Keras model, train it, and save it in local file system. To run it, type ```python model.py <cnn_1d | cnn_1d_dropout | cnn_1d_dense_dropout> <the name of the file you want to save the trained model> ```

predict.py - Python script to generate predictions/Kaggle submissions. To run it , type ```python predict.py <path to your saved model file>, <path to the audio_test directory> <path to the sample_submission.csv file> <path to the train.csv file> <the file name you want to save the generated submissions>```. In this repository, I have included some pre-trained models in the pre_trained_models directory. Feel free to use them to generate predictions/Kaggle submissions file.

In the submissions directory, I have included all my submission so far.


## Milestones
Milestone 1 - get familiar with audio data, playing around with the baseline model provided, etc. July 3 - July 6

Milestone 2 - read research papers realted to audio tagging systems. July 7 - July 11

Milestone 3 - design the model, implementing the model, and testing. July 12 - July 20

Milestone 4 - compare the test performance with the models on leaderboard of Kaggle, try to improve the model, and iterate. July 21 - July 31

Milestone 5 - write the report and prepare for presentation. August 1

## Work log
I think it really surprises me that 1-D convolution works better and requires less time to train than the proposed framed input form with 2-D convolution in the baseline system. I would say that I actually spent 3 days to trying to figure out how to save the framed examples they used in the baseline system into disk. The resulting framed examples of log-mel spectrograms are just too large to be fit into the memory in my laptop. Then I decided to give 1-D convolution a try, and it works pretty well, at least from my expections and resources.

However, I think if I could do it again or have longer time, I would definitely try to use VGG network first, since they seem to be not that complext, though fairly deep. Since right now I feel that with my shallow CNN model, it might not be powerful as one of the Deep CNNs to quickly learn from the audio data.
