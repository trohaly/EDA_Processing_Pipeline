# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 07:28:10 2023
Developed by DCS Corp for DEVCOM Army Research Laboratory under contract 
number W911QX21D0004
Approved for public release, distribution is unlimited
Contributors: Thomas Rohaly, Mike Nonte, David Chhan
"""

import numpy as np
from scipy.signal import cheby1, filtfilt
from scipy import interpolate

#Correction script is based on the BioSemi Forum post, link below:
#https://www.biosemi.nl/forum/viewtopic.php?t=808

def getEDA(aux1, aux2, fs):
    
    # Step 1: Correct the scaling factors applied to data in the LSL app
    aux1 = (aux1/0.03125) * 256;
    aux2 = (aux2/0.03125) * 256;

    # Step 2 - Compute difference of Aux2 - Aux1
    eda = aux2 - aux1;
    
    # Step 3 - 4th order Lowpass filter 1dB passband ripple, 16Hz cutoff frequency, 
    wn = 16/(fs/2)
    b, a = cheby1(4, 1, wn)
    eda = filtfilt(b,a,eda);
    
    # Step 4 - apply scaler
    eda = 13.3*eda;    
    
    #Steps 5 and 6 -Decimate signal by a factor of 16 and Divide data into 128
    #data-point windows and calculate (max - min) for each each window.
    eda2 = []
    for i in range(0, len(eda) - 128, 128):
        eda2.append(np.max(eda[i:i + 128]) - np.min(eda[i:i + 128]))

    # Step 7 - Byte shift
    eda3 = np.array(eda2)/8192;
    
    # Step 8 - Convert ohms to Siemens
    eda3 = 1./eda3;
    
    # Step 9 - Convert S to uS
    eda3 = 1e6*eda3;

    #Step 10 - Upsample back to EEG.srate                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    x=np.arange(0,len(eda3))/(16)
    y = eda3
    
    f = interpolate.interp1d(x, y)
    newx = np.arange(0,x[-1],1/fs)
    eda3=f(newx)
    
    #Return the corrected signal
    return eda3
    
   
