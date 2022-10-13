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

"""
This script generates input data for the deep learning denoiser using as inputs
the NumPy file ('raw_data.npy'), containing raw waveforms; and the NumPy file ('metadata.npy'),
containing the metadata associated with the raw waveforms. Outputs are min-max-normalized
real and imaginary parts of the short-time Fourier transform (STFT) of each of the input waveforms.

R. Tibi (rtibi@sandia.gov), Oct 04, 2022
"""

### File containing raw waveform data ###
INPUT_DATA = np.load('raw_data.npy')

### File coontaining metadata  ###
METADATA = np.load('metadata.npy')

### Outputs ###
STFT_REAL_NORM = 'stft_raw_data_real_norm.npy'
STFT_IMAG_NORM = 'stft_raw_data_img_norm.npy'

nwf = INPUT_DATA.shape[0]

SEGL = 1                    # Segment length in seconds for STFT
OVERL = 0.5                 # Segment ovelap in seconds for STFT

# loop over waveforms
for wf in np.arange(0, nwf):
    wf_id = int(INPUT_DATA[wf][0])
    print("Processing Wfm #", wf_id)
          
    # Retrieve waveform
    amp_noisy_signal = INPUT_DATA[wf][1:]
    samplerate = float(METADATA[wf][3])
    
    nseg = int(SEGL * samplerate)
    nover = int(OVERL* samplerate)
    
    #### STFT of the input waveform
    f1, t1, Zns = signal.stft(amp_noisy_signal, fs=samplerate, window='hann',
            nperseg=nseg, noverlap=None, nfft=None, detrend=False, 
            return_onesided=False, boundary='zeros', padded=True, axis=-1)
    
    #### Normalize STFT of the input waveform
    #real part
    zreal_min = np.min(Zns.real); zreal_max = np.max(Zns.real)
    Zreal_norm = (Zns.real - zreal_min) / (zreal_max - zreal_min)
    
    #img part
    zimg_min = np.min(Zns.imag); zimg_max = np.max(Zns.imag)
    Zimg_norm = (Zns.imag - zimg_min) / (zimg_max - zimg_min)
    
    #### Generate numpy arrays of the STFTs of the input waveform
    if wf == 0:
        noisy_wfm_real = np.array(Zreal_norm)
        noisy_wfm_imag = np.array(Zimg_norm)
        
    else:
        noisy_wfm_real = np.vstack((noisy_wfm_real, np.array(Zreal_norm)))
        noisy_wfm_imag = np.vstack((noisy_wfm_imag, np.array(Zimg_norm)))
        
# Save STFTs of the input waveforms in the dataset
np.save(STFT_REAL_NORM, noisy_wfm_real)
np.save(STFT_IMAG_NORM, noisy_wfm_imag)

### Check array size ###
A = np.load(STFT_REAL_NORM)
print('Array size for real part of noisy signal STFT:', A.shape)

B = np.load(STFT_IMAG_NORM)
print('Array size for imagineray part of noisy signal STFT:', B.shape)