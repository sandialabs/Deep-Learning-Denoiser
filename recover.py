"""
###############################################################################
Copyright [2022] National Technology & Engineering Solutions of
Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
there is a non-exclusive license for use of this work by or on behalf of
the U.S. Government. Export of this data may require a license from
the United States Government.
###############################################################################
"""

import numpy as np
from scipy import signal
from keras.models import load_model

"""
This script uses the trained deep learning denoiser model 'cnn_denoiser_3comp.h5' 
to denoise the waveforms contained in the NumPy file 'raw_data.npy'. Inputs are 
the associated real and  imagineray parts of the STFTs ('stft_raw_data_real_norm.npy'
and 'stft_raw_data_img_norm.npy'), generated using the script 'ConvNet_Input_Creator_Norm.py'. 

R. Tibi (rtibi@sandia.gov), Oct 04, 2022
"""

## Model file
mfile = 'cnn_denoiser_3comp.h5'
print(mfile)

#### Open precomputed denoiser model file
data = load_model(mfile)

### Inputs ####
NOISY_DATA = np.load('raw_data.npy')
REAL_STFT_NOISY = np.load('stft_raw_data_real_norm.npy')
IMG_STFT_NOISY = np.load('stft_raw_data_img_norm.npy')

### Outputs ###
DENOISED_DATA = 'denoised_data.npy'
NOISE_DATA = 'noise_data.npy'

wfms = NOISY_DATA.shape[0] # Total number of waveforms

SEGL = 1                    # Segment length in seconds for Spectrogram STFT
OVERL = 0.5                   # Segment overlap in seconds for STFT
SAMPLERATE = 100.0

nseg = int(SEGL * SAMPLERATE)
nover = int(OVERL* SAMPLERATE)

for row in np.arange(0, wfms):
    print("Working on wfm #:", row)
    # Extract noisy waveform
    wfm_id = NOISY_DATA[row][0]
    noisy_signal = NOISY_DATA[row][1:]
    
     # Compute STFT of the noisy signal ####
    freq, time, Z = signal.stft(noisy_signal, SAMPLERATE, window='hann',
                nperseg=nseg, noverlap=nover, nfft=None, detrend=False, 
                return_onesided=False, boundary='zeros', padded=True, axis=-1)
    
    ### Extract real and imaginary parts of stft of the noisy waveform ###
    real_stft = (REAL_STFT_NOISY.reshape(wfms, 100, 121))[row, :, :]
    real_stft = np.reshape(real_stft, [-1, 100, 121, 1])
    
    img_stft = (IMG_STFT_NOISY.reshape(wfms, 100, 121))[row, :, :]
    img_stft = np.reshape(img_stft, [-1, 100, 121, 1])
    
    stft_realdata = np.concatenate((real_stft, img_stft), -1)
    
    #### predict signal and noise masks of the noisy waveform ###
    denoised = data.predict(stft_realdata)
    mask_signal = np.reshape(denoised[:, :, :, 0], [100, 121])
    mask_noise = np.reshape(denoised[:, :, :, 1], [100, 121])
    
    # Extract denoised signal ####
    Zest_signal = mask_signal * Z
    t1, xsignal = signal.istft(Zest_signal, fs=SAMPLERATE, window='hann', nperseg=nseg,
    noverlap=nover, nfft=None, input_onesided=False, boundary=True, time_axis=-1, freq_axis=-2)
    signal_amp = np.array(xsignal.real)
    
    # Extract noise ####
    Zest_noise = mask_noise * Z
    t2, xnoise = signal.istft(Zest_noise, fs=SAMPLERATE, window='hann', nperseg=nseg,
    noverlap=nover, nfft=None, input_onesided=False, boundary=True, time_axis=-1, freq_axis=-2)
    noise_amp = np.array(xnoise.real)
    
    # Inserting waveform id ###
    signal_amp = np.insert(signal_amp, 0, int(wfm_id))
    noise_amp = np.insert(noise_amp, 0, int(wfm_id))
    
    if row == 0:
        wf1 = np.array(signal_amp)
        wf2 = np.array(noise_amp)
    else:
        wf1 = np.vstack((wf1, np.array(signal_amp)))
        wf2 = np.vstack((wf2, np.array(noise_amp)))
        
### Saving waweform data as numpy file ###
np.save(DENOISED_DATA, wf1)
np.save(NOISE_DATA, wf2)

### Check data size ###
A = np.load(DENOISED_DATA)
print('Array size for signal data:', A.shape)

B = np.load(NOISE_DATA)
print('Array size for noise data:', B.shape)