####### use these functions to make transforms easier with less code. 


import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt
import sklearn

# Global Variables 

sr = 22050 
sample_rate = sr    # sample rate. based on Nyquist frequency, we only care about frequencies up to 10kHz therefor the sample rate will only perserve those frequencies 
n_fft = 2048
hop_length = 512
duration = 29 # length of song to be used (in seconds) 
n_mels=128

# Music Classification is usually done with 5 features:
# Mel-Frequency Cepstral Coefficients, Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, and Spectral Roll-off 


###### returns Mel Spectrogram
def get_mels(filename, sample_rate=sr):
    y, sr = librosa.load(
        path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
        sr = sample_rate, # by convention the default sample rate is 22050, lower if not enough processing power 
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

def scaled_mfcc(filename):

    y, sr = librosa.load(
        path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
        sr = sample_rate, # by convention the default sample rate is 22050, lower if not enough processing power 
        mono = True,      # stereo isn't important. 
        offset = 0.0,     # start reading audio after this time (in seconds)
        duration = duration, 
        res_type = 'kaiser_best'
        )
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mels=n_mels)
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    return mfccs
    

## Spectral Centroid
## indicates "center of mass" which is calculated as the weighted mean of frequencies 

def spectral_centroids(filename):

    y, sr = librosa.load(
        path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
        sr = sample_rate, # by convention the default sample rate is 22050, lower if not enough processing power 
        mono = True,      # stereo isn't important. 
        offset = 0.0,     # start reading audio after this time (in seconds)
        duration = duration, 
        res_type = 'kaiser_best'
        )
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)
    return spectral_centroids


### zero-crossing rate is the rate at which the signal changes from positive to negative or back 
def zero_crossing(filename):
    y, sr = librosa.load(
        path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
        sr = sample_rate, # by convention the default sample rate is 22050, lower if not enough processing power 
        mono = True,      # stereo isn't important. 
        offset = 0.0,     # start reading audio after this time (in seconds)
        duration = duration, 
        res_type = 'kaiser_best'
        )
    zero_crossings = librosa.zero_crossings(y=y, pad=False)
    return zer_crossings

###Chroma features or a representation of the 12 pitch classes
def chroma_features(filename):
    y, sr = librosa.load(
        path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
        sr = sample_rate, # by convention the default sample rate is 22050, lower if not enough processing power 
        mono = True,      # stereo isn't important. 
        offset = 0.0,     # start reading audio after this time (in seconds)
        duration = duration, 
        res_type = 'kaiser_best'
        )
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    return chromagram

#### Spectral Rolloff is a messure of the shape of the signal for each frame in a signal/
def spectral_rolloff(filename):
    y, sr = librosa.load(
        path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
        sr = sample_rate, # by convention the default sample rate is 22050, lower if not enough processing power 
        mono = True,      # stereo isn't important. 
        offset = 0.0,     # start reading audio after this time (in seconds)
        duration = duration, 
        res_type = 'kaiser_best'
        )
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return spectral_rolloff
    















def get_cqt(filename, sample_rate=sr):
    y, sr = librosa.load(
        path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
        sr = sample_rate, # by convention the default sample rate is 22050, lower if not enough processing power 
        mono = True,      # stereo isn't important. 
        offset = 60.0,     # start reading audio after this time (in seconds)
        duration = duration, 
        res_type = 'kaiser_best'
        )    

    C = librosa.cqt(y, sr=sr)
    logC = librosa.amplitude_to_db(abs(C))
    return logC


# features extraction 

#def features(y, sr)

    #M = librosa.feature.melspectrogram(y=y, sr=sr)
    #MFCC = librosa.feature.mfcc(y=y, sr=sr)
    #chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    #tonnetz = librosa.feature.tonnetz(y=y, sr=sr) # Computes the tonal centroid features 
    #delta = librosa.feature.delta()
