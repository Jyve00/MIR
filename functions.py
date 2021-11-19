# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score
import librosa


# functions

###### returns Mel Spectrogram
# def get_mels(filename, sr=sr):
#     y, sr = librosa.load(
#         path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
#         sr = sr,          # by convention the default sample rate is 22050, lower if not enough processing power 
#         mono = True,      # stereo isn't important. 
#         offset = 0.0,     # start reading audio after this time (in seconds)
#         duration = duration, 
#         res_type = 'kaiser_best'
#         )
#     M = librosa.feature.melspectrogram(y, n_fft=n_fft, hop_length=hop_length)
#     M_db = librosa.power_to_db(M, ref=np.max)

#     return M_db


# ###### returns Mel-Frequency Cepstral Coefficient
# def get_mfcc(filename, sample_rate=sr):
#     y, sr = librosa.load(
#         path = filename,  # load in audio file. MP3 not supported refer to Librosa documentation 
#         sr = sample_rate, # by convention the default sample rate is 22050, lower if not enough processing power 
#         mono = True,      # stereo isn't important. 
#         offset = 0.0,     # start reading audio after this time (in seconds)
#         duration = duration, 
#         res_type = 'kaiser_best'
#         )

#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mels=n_mels)
#     return mfcc

    
def get_mels(X, sample_duration=29):
    """
    Generate mel spectrograms for a collection of audio signals.
    
    Inputs:
        X: array-like
            The collection of relative paths for the audio signal files.
        sample_duration: int, default=29
            The duration in seconds of audio to be loaded.
        
    Output:
        X_mels: numpy ndarray (4D tensor)
            A 4D tensor of shape (number of audio signals, number of mels, number of frames, number of channels).
    """
    
    features = []
    for song in X:
        try:
            y, sr = librosa.load(song, duration=sample_duration)
            S_dB = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.min)
            
            # append to the list
            # create one new axis for concatenation later and another for the single amplitude channel
            features.append(S_dB[np.newaxis,..., np.newaxis])
        except:
            continue

    # concatenate along the first axis; result should be a 4D tensor of shape (#signals, #mels, #frames, #channels)
    X_mels = np.concatenate(features,axis=0)
    
    return X_mels


def create_min_max_scaler(X_tr):
    """
    Create a custom min max scaler to normalize data.
    
    Input:
        X_tr: numpy ndarray
            Training data for basis of normalization.
            
    Output:
        custom_scaler: function
            Function to transform data.
    """
    X_tr_max = X_tr.max()
    X_tr_min = X_tr.min()
    def custom_scaler(X):
        return (X - X_tr_min) / (X_tr_max - X_tr_min)
    return custom_scaler


def evaluate(model, history, X, y, labels):
    """
    Modified from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    and Lindsey Berlin at Flatiron School
    
    Plot accuracy and loss over training epochs. Make predictions and plot precision by multiclass target labels.
    
    Input:
        model: keras fit model object
            The trained model.
        history: keras history object
            The output from the trained model.
        X: numpy ndarray
            Input data.
        y: numpy ndarray
            Target data.
        labels: array-like
            Target class labels.
            
    Output:
        preds: numpy ndarray
            Model predictions from X.
        precision_df: pandas dataframe
            Precision by multiclass target labels.
    """
    
    # training history results
    sns.set_theme(context='notebook')
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8,8), sharex=True)
    fig.suptitle('Model Results')
    
    # text for textbox
    results = model.evaluate(X, y, verbose=0)
    acc_text = f"Test accuracy: {results[1]:.3f}"
    loss_text = f"Test loss: {results[0]:.3f}"

    # visualize accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_ylabel('Accuracy')
    ax1.legend(['train', 'test'], loc='upper left')
    ax1.text(x=0.82, y=0.1, s=acc_text, fontsize=14,
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax1.transAxes)
    ax1.grid(False)
    
    # visualize loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['train', 'test'], loc='upper left')
    ax2.text(x=0.85, y=0.15, s=loss_text, fontsize=14,
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax2.transAxes)
    ax2.grid(False)
    
    
    # get predictions on test set
    preds = model.predict(X)
    
    # precision scores by class
    precision = precision_score(np.argmax(y, 1), np.argmax(preds, 1), average=None)

    # data for plot
    precision_df = pd.DataFrame(precision, index=labels, columns=['Precision'])
    precision_df = precision_df.sort_values('Precision', ascending=False)

    # precision bar plot
    fig, ax = plt.subplots(figsize=(9, 6))

    prec = sns.barplot(data=precision_df,
                       x=precision_df.index,
                       y='Precision',
                       ax=ax,
                       color='royalblue')

    # Customize asthetic
    ax.grid(False)
    prec.set_title("Precision Scores by Genre")
    prec.set_xlabel('Genre')
    prec.set_ylabel('Precision')
    prec.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])

    # set yticklabels to be a %
    yticklabels = [f'{tick *100:.0f}%' for tick in prec.get_yticks()]
    prec.set_yticklabels(yticklabels)
    
    return preds, precision_df