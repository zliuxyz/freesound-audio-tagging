import os
import sys
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

SAMPLE_RATE = 44100


def wav_to_log_mel_spectrogram(clip_dir=None, clip=None):
    filename = os.path.join(clip_dir, clip)

    # wav file to waveform
    y, sr = librosa.load(path=filename, sr=SAMPLE_RATE)
    duration = librosa.get_duration(y=y, sr=sr)

    # waveform to mel spectrogram
    hop_length = int(round(SAMPLE_RATE * 0.010))
    win_length = int(round(SAMPLE_RATE * 0.025))
    n_fft = 2 ** int(np.ceil(np.log(win_length) / np.log(2.0)))
    n_mels = 64
    fmin = 125
    fmax = 7500
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, fmin=fmin, fmax=fmax)

    # mel spectrogram to log mel spectrogram
    mel_log_offset = 0.001
    log_mel_spectrogram = np.log(mel_spectrogram + mel_log_offset)
    # return shape [T, n_mels]
    return log_mel_spectrogram.T


def wav_to_data(clip_dir='audio_train', csv_file='train.csv'):
    train = pd.read_csv(csv_file)
    unique_labels = train.label.unique().tolist()
    fname = train.fname
    label = train.label
    max_length = 0
    spectrograms = []
    labels = np.zeros((len(fname), 41))
    for i in range(len(fname)):
        label_index = unique_labels.index(label[i])
        labels[i, label_index] = 1
        s = wav_to_log_mel_spectrogram(clip_dir=clip_dir, clip=fname[i])
        if s.shape[0] > max_length:
            max_length = s.shape[0]
        spectrograms.append(s)
    np.save('labels.npy', labels)
    features = np.zeros((len(fname), max_length, 64))
    for i in range(len(spectrograms)):
        feature = spectrograms[i]
        features[i, 0:feature.shape[0], 0:feature.shape[1]] = feature
    np.save('samples.npy', samples)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Wrong number of arguments...")
        sys.exit(1)
    clip_dir = sys.argv[1]
    csv_file = sys.argv[2]
    wav_to_data(clip_dir=clip_dir, csv_file=csv_file)
