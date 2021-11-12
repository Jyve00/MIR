####### use these functions to make transforms easier with less code. 


import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt

# Global Variables 

sr = 22050
n_fft = 2048
hop_length = 512


# the 4 functions are based off the 4 modules mentioned in Librosa Paper


# function to perform all required mathmatical transforms 
def transform(filename, sample_rate=22050):
    y, sr = librosa.load(
        path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
        sr = sample_rate, # by convention the default sample rate is 22050, lower if not enough processing power 
        mono = True,      # stereo isn't important. 
        offset = 0.0,     # start reading audio after this time (in seconds)
        duration = None, 
        res_type = 'kaiser_best'
        )


    D = librosa.stft(y)
    S = np.abs(librosa.stft(y))
    C = librosa.cqt(y, sr)

    return y, sr, D, S, C 



# features extraction 

def features(y, sr)

    M = librosa.feature.melspectrogram(y=y, sr=sr)
    MFCC = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr) # Computes the tonal centroid features 
    #delta = librosa.feature.delta()


    return M, MFCC, chroma, tonnetz, 


def efx(y, sr):
    y_harmonic, y_percussive = librosa.effects.hpss(y)  


def beats(y , sr):
    onset_envelope = librosa.onset.onset_strength(y, sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope)

    return onsets,  


# visuals 
def visuals():
    # display spectrogram
    log_power = librosa.logamplitude(C**2, ref_power = np.max, top_db=40)
    log_specshow = librosa.display.specshow(log_power, x_axis='time', y_axis='log')
    plt.colorbar()


    cqt_plot = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
    return log_specshow, cqt_plot




    plt.subplot(2,1 ,1)
    plt.plot(onset_envelope, label = 'Onset strength')
    plt.vlines(onset, 0, onset_envelope.max(), color='r', alpha=0.25)
    plt.xticks([]), plt.yticks([])
    plt.legend(frameon=True)
    plt.axis('tight')



    # spectral_centroid
    # spectral_bandwidth
    # spectral_rolloff 
    # spectral_contrast 







# MIR tools suggested in the Librosa Paper by its creator Brian Mcfee 
# important features: beat detection, Tempo, chroma , 


# separate harmonic and percussive waveforms. take percussive wave and run thru on_set_detection

    


#### make function to find musical key 