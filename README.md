# Music Genre Classification

![alt text](https://www.incimages.com/uploaded_files/image/1920x1080/getty_626660256_2000108620009280158_388846.jpg)

**Authors**: 

- [Nicholas Indorf](https://github.com/Nindorph)
- [Andrew Whitman](https://github.com/andrewwhitman)
- [Stephen William](https://github.com/Jyve00)


## Overview

We have been tasked with building a genre classifier for use in a playlist generator system. Our data is a 1000 song samples with a single label from a list of 10 
genres. We processed these samples into a series of mel spectrograms, which were fed into a convolutional neural network. The output was which genre it believed the 
sample was most likely to belong to. We achieved 58% accuracy, and focused on precision scores for each genre. The model performed better for some genres than 
others. We recommend it inclusion within the playlist generator system. Even if it mis-classifies a song as a certain genre, it picks up on the major/important 
features that underly it. This would make the song a good fit for a playlist containing songs that are technically a different genre.

## Data
The original dataset can be found at http://marsyas.info/downloads/datasets.html

The Dataset consists of 1000 songs evenly divided up onto 10 music genres. The audio files are each 30 seconds long with a sample rate of 22050 Hz and bit dept of 16 bits. All the songs are in mono and in the .wav format. There was only one song that gave an encoding error and could not be import. The song in question was a Jazz song and it was removed from our dataset. The dataset also included 2 CSV's that provide some important information. One CSV file contains metadata on all 30 seconds of every song and the other is metadata from all songs but split up into 3 second segments. We mostly used the CSV's to connect the audio .wav filepaths with their correct labels. 

## Modeling

The data was split into train (75%), test (15%), and holdout (10%) sets.
The data was fed into a convolutional neural network.

## Evaluation

Final model achieved 58% accuracy overall and the below accuracy scores according to genre.
![precision scores for genres](https://github.com/Jyve00/MIR/blob/main/Images/Precision.png)

## Information

Check out our [notebook](https://github.com/Jyve00/MIR/blob/main/MusicGenreClassification.ipynb) for a more thorough discussion of our project, as well as our [presentation](https://github.com/Jyve00/MIR/blob/main/presentation.pdf).

## Repository Structure

```

├── Data                                <- folder containing csv data and nested subfolder of audio data
│   └── ...
├── images                              <- folder containing images for README and presentation
│   └── ...
├── notebooks                           <- folder containing additional notebooks for data exploration and modeling
│   └── ...
├── .gitattributes                      <- file specifying files for git lfs to track
├── .gitignore                          <- file specifying files/directories to ignore
├── MusicGenreClassification.ipynb      <- notebook detailing the data science process containing code and narrative
├── README.md                           <- Top-level README
├── presentation.pdf                    <- presentation slides for a business audience
└── functions.py                        <- Contains helper function for model evaluation

``` 
