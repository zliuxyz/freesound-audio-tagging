import inputs
import sys
import pandas as pd
import numpy as np
from keras.models import load_model


def predict(model=None, clip_dir='audio_test', test_csv_file='sample_submission.csv', train_csv_file='train.csv', submission_csv_file=None):
    model = load_model(model)
    test = pd.read_csv(test_csv_file)
    train = pd.read_csv(train_csv_file)
    unique_labels = train.label.unique().tolist()
    fname = test.fname
    columns = ['fname', 'label']
    df = pd.DataFrame(columns=columns)
    for i in range(len(fname)):
        print(i, fname[i])
        try:
            feature = inputs.wav_to_log_mel_spectrogram(clip_dir=clip_dir, clip=fname[i])
            feature = feature.T
            print(feature.shape)
            temp = np.zeros((3001, 64))
            temp[0:feature.shape[0], 0:feature.shape[1]] = feature
            temp = np.expand_dims(temp, axis=0)
            prediction = model.predict(temp)
            prediction = prediction.flatten()
            print(prediction.shape)
            first = np.argmax(prediction)
            prediction[first] = 0
            second = np.argmax(prediction)
            prediction[second] = 0
            third = np.argmax(prediction)
            prediction[third] = 0
            label = unique_labels[first] + ' ' + unique_labels[second] + ' ' + unique_labels[third]
        except ValueError:
            label = test.label[i]
        print(label)
        df.loc[i] = [fname[i], label]
    df.to_csv(submission_csv_file, encoding='utf-8', index=False)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Wrong number of arguments...')
        sys.exit(-1)
    model = sys.argv[1]
    clip_dir = sys.argv[2]
    test_csv_file = sys.argv[3]
    train_csv_file = sys.argv[4]
    submission_csv_file = sys.argv[5]
    predict(model=model, clip_dir=clip_dir, test_csv_file=test_csv_file, train_csv_file=train_csv_file, submission_csv_file=submission_csv_file)
