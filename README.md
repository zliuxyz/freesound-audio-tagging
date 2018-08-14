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
July 3 - July 6
Played with the baseline model, tring to understand how do they transform the audio so that the input to their cnn is of the same size.

July 15 - July 20
Trying to rebuild the baseline model by myself using Keras. And I found out that when I was trying to do preprocessing on the training data, there will be memeory error since there are just too many frames after extracted them from the spectrograms.

July 20 - July 25
I decided to use only the manually verifired dataset, and transfrom them into log-mel spectrograms, and then divide them into equal length of frames with shape (25, 64). I saved the numpy array as a file on my disk. Trying to train the baseline model I built using keras. However, the training time is very long, it took about 7 hours to finish 30 epochs. The training accuracy is very low as well, which only gives me about 40%.

July 27 - Aug 1
I decided to use log-mel spectrograms direclty as the input, since first, it reduces the number of inputs we have, second, the change in each audio sample over time is preversed, maybe our model can capture that. Also, I notice that audio data is naturally sequential. So, it makes more sense to be able to learn from the data considering it's change over time. Thus, we could just use a 1-D CNN to slide through the input from left to right.

Aug 4 - Aug 7
Train the 1-D CNN model to classify the training set. The training accuracy went to about 84% after 30 epochs, which was done in 3 hours. The test accuracy in Kaggle was around 75%. It seems that there is some overfitting.

Aug 8 - Aug 13
Added a dropout layer before the dense layer to overcome the overfitting problem. The training accuracy went to about 81% after 30 epochs, which was done in 3 hours. The test accuracy in Kaggle was around 80.73%.

I think it really surprises me that 1-D convolution works better and requires less time to train than the proposed framed input form with 2-D convolution in the baseline system. I would say that I actually spent 3 days to trying to figure out how to save the framed examples they used in the baseline system into disk. The resulting framed examples of log-mel spectrograms are just too large to be fit into the memory in my laptop. Then I decided to give 1-D convolution a try, and it works pretty well, at least from my expections and resources.

However, I think if I could do it again or have longer time, I would definitely try to use VGG network first, since they seem to be not that complext, though fairly deep. Since right now I feel that with my shallow CNN model, it might not be powerful as one of the Deep CNNs to quickly learn from the audio data.

Also, I feel like it would be better for me to have a teammate. Since I was actually taking 4 graduate-level CS classes this summer. Each of this class has a project to work on. So, sometimes, it's kind of hard for me to balance my time to make great contribution to every part of every project. In the meantime, I was doing co-op job search, so I guess the workload for me was not that light. If I had a teammate, I feel I could achieve more from this project.
