####### use these functions to make transforms easier with less code. 


import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt
import sklearn

# Global Variables 
 
sr = 22050    # sample rate. based on Nyquist frequency, we only care about frequencies up to 10kHz therefor the sample rate will only perserve those frequencies 
n_fft = 2048
hop_length = 512
duration = 30        # length of song to be used (in seconds) 
n_mels=128           # number of Mel Filter Bins used
n_mfcc = 20          # number of Coefficients 


# all functions are meant to take in the file path of an audio file and returns a matrix of a feature extraction 

###### returns Mel Spectrogram
def get_mels(filename, sr=sr):
    y, sr = librosa.load(
        path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
        sr = sr,          # by convention the default sample rate is 22050, lower if not enough processing power 
        mono = True,      # stereo isn't important. 
        offset = 0.0,     # start reading audio after this time (in seconds)
        duration = duration, 
        res_type = 'kaiser_best'
        )
    M = librosa.feature.melspectrogram(y, n_fft=n_fft, hop_length=hop_length)
    M_db = librosa.power_to_db(M, ref=np.max)

    return M_db


###### returns Mel-Frequency Cepstral Coefficient
def get_mfcc(filename, sample_rate=sr):
    y, sr = librosa.load(
        path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
        sr = sample_rate, # by convention the default sample rate is 22050, lower if not enough processing power 
        mono = True,      # stereo isn't important. 
        offset = 0.0,     # start reading audio after this time (in seconds)
        duration = duration, 
        res_type = 'kaiser_best'
        )

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mels=n_mels)
    return mfcc

    




