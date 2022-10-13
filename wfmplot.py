"""
###############################################################################
Copyright [2022] National Technology & Engineering Solutions of
Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
there is a non-exclusive license for use of this work by or on behalf of
the U.S. Government. Export of this data may require a license from
the United States Government.
###############################################################################
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import os.path

"""
This script displays the raw and denoised waveforms for each specified waveform ID. Inputs are files
containing the raw seismograms ('raw_data.npy'), denoised waveforms ('denoised_data.npy'), and
the associated metadata ('metadata.npy'). The red and green lines in the displayed waveforms
delimit the noise and signal windows, respectively, used to estimate the SNRs.

R. Tibi (rtibi@sandia.gov), Oct 04, 2022
"""

### For inline plotting ###
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')

if not os.path.exists('plots'):
        os.makedirs('plots')
        
RAW = np.load('raw_data.npy')
DENOISED = np.load('denoised_data.npy')
METADATA = np.load('metadata.npy')

wfms = DENOISED.shape[0]
ans = input('Input waveform ID (from 1 to %d): ' %(wfms))

plt.figure(figsize=(6, 4))

WLENGTH = 10          # length in sec for noise and signal window

ID = int(ans)
row = ID - 1

wf0 = RAW[row][1:];
wf1 = DENOISED[row][1:]
raw_id = int(RAW[row][0]); denoised_id = int(DENOISED[row][0])
amp0 = wf0/(max(abs(wf0))); amp1 = wf1/(max(abs(wf1)))

sta = str(METADATA[row][1]); comp = (str(METADATA[row][2]))[-1]
sample_rate = float(METADATA[row][3]); phase = str(METADATA[row][4])
start_time = float(METADATA[row][5]); arr_time = phase = float(METADATA[row][6])
npts = int(METADATA[row][7]) - 1

print("Waveform ID: %3d; SAMPLE RATE: %4d sps; NPTS: %5d" %(ID, sample_rate, npts))

# Indexes for noise and signal window
dtime1 = arr_time - start_time
end_time = start_time + (npts / sample_rate)
dtime2 = end_time - arr_time
if dtime1 <= WLENGTH + 1:
    WLENGTH = dtime1 # Reduced signal and noise window length for shorter pre-arrival window
if dtime2 <= WLENGTH:
    WLENGTH = dtime2 #Reduced signal and noise window length for shorter post-arrival window
    
signal_ind1 = int(dtime1 * sample_rate)
signal_ind2 = signal_ind1 + (int(WLENGTH * sample_rate))
noise_ind2 = signal_ind1 - int((1.0 * sample_rate))
noise_ind1 = noise_ind2 - (int(WLENGTH * sample_rate))

### Raw waveform
amp_noise = RAW[row][noise_ind1:noise_ind2]
rms_noise = math.sqrt(np.mean(np.square(amp_noise)))
amp_signal = RAW[row][signal_ind1:signal_ind2]
rms_signal = math.sqrt(np.mean(np.square(amp_signal)))
snr = 10 * (math.log10(rms_signal / rms_noise))
snr = ("%4.1f" % snr)


tarray = np.arange(0, float(npts)/float(sample_rate), 1.0/float(sample_rate))
plt.subplot(2,1,1)
plt.plot(tarray, amp0,'k', alpha=0.7, linewidth=0.7, color='b')

# Signal window
t1signal = float(signal_ind1) / sample_rate; t2signal = t1signal + WLENGTH
plt.plot(t1signal, 0, marker='|', color='w', markeredgewidth=1.5,
                 linestyle='', markersize=53)
marker1, = plt.plot(t1signal, 0, marker='|', color='green', markeredgewidth=1.5,
                 linestyle='', markersize=50)
plt.plot(t2signal, 0, marker='|', color='w', markeredgewidth=1.5,
                 linestyle='', markersize=53)
marker2, = plt.plot(t2signal, 0, marker='|', color='green', markeredgewidth=1.5,
                 linestyle='', markersize=50)

# Noise window
t1noise = float(noise_ind1) / sample_rate; t2noise = t1noise + WLENGTH
plt.plot(t1noise, 0, marker='|', color='w', markeredgewidth=1.5,
                 linestyle='', markersize=53)
marker3, = plt.plot(t1noise, 0, marker='|', color='red', markeredgewidth=1.5,
                 linestyle='', markersize=50)
plt.plot(t2noise, 0, marker='|', color='w', markeredgewidth=1.5,
                 linestyle='', markersize=53)
marker4, = plt.plot(t2noise, 0, marker='|', color='red', markeredgewidth=1.5,
                 linestyle='', markersize=50)

frame =plt.gca()
ax = plt.axis()
plt.minorticks_on()
plt.xlim(0, 60)
plt.yticks([-1,0, 1])
plt.ylim(-1, 1)
bbox_props = dict(boxstyle="round", fc="w", ec="0", alpha=1.) 
plt.text(59, 0.70, 'Raw', size=10, backgroundcolor='white', bbox=bbox_props,
         horizontalalignment='right')
plt.text(1, 0.75, 'ID=' + str(raw_id), size=9, horizontalalignment='left')
plt.text(1, 0.1, sta, size=10, horizontalalignment='left')
plt.text(1, -0.3, comp, size=10, horizontalalignment='left')
plt.text(59.5, -0.80, 'SNR=' + str(snr) + ' dB', size=10,
         horizontalalignment='right')


#Denoised waveform
amp_noise = DENOISED[row][noise_ind1:noise_ind2]
rms_noise = math.sqrt(np.mean(np.square(amp_noise)))
amp_signal = DENOISED[row][signal_ind1:signal_ind2]
rms_signal = math.sqrt(np.mean(np.square(amp_signal)))
snr = 10 * (math.log10(rms_signal / rms_noise))
snr = ("%4.1f" % snr)

plt.subplot(2,1,2)
plt.plot(tarray, amp1,'k', alpha=0.7, linewidth=0.7, color='b')
plt.plot(t1signal, 0, marker='|', color='w', markeredgewidth=1.5,
                 linestyle='', markersize=53)
marker1, = plt.plot(t1signal, 0, marker='|', color='green', markeredgewidth=1.5,
                 linestyle='', markersize=50)
plt.plot(t2signal, 0, marker='|', color='w', markeredgewidth=1.5,
                 linestyle='', markersize=53)
marker2, = plt.plot(t2signal, 0, marker='|', color='green', markeredgewidth=1.5,
                 linestyle='', markersize=50)

# Noise window
t1noise = float(noise_ind1) / sample_rate; t2noise = t1noise + WLENGTH
plt.plot(t1noise, 0, marker='|', color='w', markeredgewidth=1.5,
                 linestyle='', markersize=53)
marker3, = plt.plot(t1noise, 0, marker='|', color='red', markeredgewidth=1.5,
                 linestyle='', markersize=50)
plt.plot(t2noise, 0, marker='|', color='w', markeredgewidth=1.5,
                 linestyle='', markersize=53)
marker4, = plt.plot(t2noise, 0, marker='|', color='red', markeredgewidth=1.5,
                 linestyle='', markersize=50)

plt.minorticks_on()
plt.xlim(0, 60)
plt.xlabel('Time (seconds)')
plt.yticks([-1,0, 1])
plt.ylim(-1, 1)
plt.text(59, 0.70, 'Denoised', size=10, backgroundcolor='white', bbox=bbox_props,
         horizontalalignment='right')
plt.text(1, 0.75, 'ID=' + str(denoised_id), size=9, horizontalalignment='left')
plt.text(1, 0.1, sta, size=10, horizontalalignment='left')
plt.text(1, -0.3, comp, size=10, horizontalalignment='left')
plt.text(59.5, -0.80, 'SNR=' + str(snr) + ' dB', size=10,
         horizontalalignment='right')

plt.tight_layout()
plt.savefig('plots/wfm_ID' + str(ID) + '.png', dpi=300)
plt.show()