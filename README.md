## Scriabin

# Introduction

I've come across many people that have difficulties determining the music genre of a song when compared to determining song names. I built Scriabin to alleviate people of this stress by applying deep learning techinques. More specifically, I constructed a Convolutional Neural Network that ingests the spectograms (sonograms) of audio snippets of songs which then returns a music genre prediction. The prediction is based on 10 possible genres (Blue, Classical, Country, Disco, Hip Hop, Jazz, Metal, Reggae and Rock). More importantly, my model exceeds <a href="https://arxiv.org/abs/1802.09697"> human accuracy</a> by 12%. 

# The Dataset

The model has been trained using the <a href="http://marsyas.info/downloads/datasets.htmland"> Marsyas dataset </a> which comprises of 1000 audio tracks each 30 seconds long, containing 10 genres (those mentioned previously), each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

# Why Spectrograms?

Spectrograms are visual representations of the spectrum of frequencies of signal as it varies throught time. When applied to audio, they are usually known as sonographs, voiceprints, or voicegrams:

For example, here is the spectrogram of the spoken words "nineteenth century"




