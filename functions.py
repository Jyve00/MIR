####### use these functions to make transforms easier with less code. 


import numpy as np
import librosa 
import matplotlib.pyplot as plt


def transform(filename, sample_rate=22050):
    y, sr = librosa.load(
        path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
        sr = sample_rate, # by convention the default sample rate is 22050, lower if not enough processing power 
        mono = True,      # stereo isn't important. 
        offset = 0.0,     # start reading audio after this time (in seconds)
        duration = None, 
        res_type = 'kaiser_best'
        )


    spectrogram = np.abs(librosa.stft(y))
    melspec = librosa.feature.melspectrogram(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)



    # spectral_centroid
    # spectral_bandwidth
    # spectral_rolloff 
    # spectral_contrast 







# MIR tools suggested in the Librosa Paper by its creator Brian Mcfee 
# important features: beat detection, Tempo, chroma , 


# separate harmonic and percussive waveforms. take percussive wave and run thru on_set_detection

    


