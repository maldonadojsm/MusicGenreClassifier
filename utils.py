# !/usr/bin/env python
# title           :utils.py
# description     :Houses several scripts used during ML training and model serving.
# author          :Sebastian Maldonado
# date            :7/22/2020
# version         :0.0
# usage           :SEE README.md
# notes           :Enter Notes Here
# python_version  :3.7,7
# conda_version   :4.8.2
# tf_version      :1.14
# =================================================================================================================
from __future__ import unicode_literals
import tensorflow as tf
import librosa
import youtube_dl
import math
import numpy as np


def extract_audio(url):
    ydl_opts = {'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'
                                    }],
                'outtmpl': '/audio_files/tf_audio.%(ext)s'}

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def prepare_file(filepath):

    TRACK_DURATION = 30
    SAMPLE_RATE = 22050
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    NUM_MFCC = 13
    N_FFT = 2048
    HOP_LENGTH = 512
    NUM_SEGMENTS_PER_TRACK = 10
    SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS_PER_TRACK)
    NUM_MFCC_VECTORS_PER_SEGMENT = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)

    signal, sample_rate = librosa.load(filepath, sr=SAMPLE_RATE)

    # Cut sample to 30 seconds

    first_thirty_seconds = librosa.time_to_samples(30, sr=SAMPLE_RATE)
    audio_clip = signal[:first_thirty_seconds]

    data = {
        "mfcc": []
    }

    for sample in range(NUM_SEGMENTS_PER_TRACK):

        start = SAMPLES_PER_SEGMENT * sample
        finish = start + SAMPLES_PER_SEGMENT

        mfcc = librosa.feature.mfcc(signal[start:finish], SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=N_FFT,
                                    hop_length=HOP_LENGTH)

        transposed_mfcc = mfcc.T

        if len(transposed_mfcc) == NUM_MFCC_VECTORS_PER_SEGMENT:
            data['mfcc'].append(transposed_mfcc.tolist())

    return data


def perform_inference(audio_data) -> list:
    """
    Performs inference, using pre-trained ML model, for submitted news YouTube music video.
    :param audio_data: Mel-frequency cepstral coefficients extracted audio file
    :return: Returns top three prediction classes for audio (float)
    """

    # Load LSTM Trained Model
    input = np.array(audio_data['mfcc'])

    x = input
    x = x[..., np.newaxis]
    #x = x[np.newaxis, ...]
    model = tf.keras.models.load_model('../models/music_genre_cnn.hdf5')

    prediction = model.predict(x)
    result = np.argmax(prediction, axis=1)

    list_result = result.tolist()

    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    return genres[max(set(list_result), key=list_result.count)]
