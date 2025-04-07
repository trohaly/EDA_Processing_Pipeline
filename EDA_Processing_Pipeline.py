# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:37:42 2024
Developed by DCS Corp for DEVCOM Army Research Laboratory under contract  
number W911QX21D0004
Approved for public release, distribution is unlimited
Contributors: Thomas Rohaly, Mike Nonte, David Chhan
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest as IsForest
import EDA_Correction

def find_stream_by_name(streams, stream_name):
    
    for stream in streams:
        if stream['info']['name'][0] == stream_name:
            return stream
    
    raise Exception(f"stream {stream_name} not found!")
    
def window(a, w = 4, o = 2, copy = False):
    """ 
    a = data
    w = window length
    o = overlap
    This function takes data and windows it based on parameters 
    that you provide
    """
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view
    
def haarFWT(signal, level):

    s = .5;                  # scaling -- try 1 or ( .5 ** .5 )

    h = [ 1,  1 ];           # lowpass filter
    g = [ 1, -1 ];           # highpass filter        
    f = len ( h );           # length of the filter

    t = signal.tolist();              # 'workspace' array
    l = len ( t );           # length of the current signal
    y = [0] * l;             # initialise output

    t = t + [ 0, 0 ];        # padding for the workspace

    for i in range ( level ):

        y [ 0:l ] = [0] * l; # initialise the next level 
        l2 = l // 2;         # half approximation, half detail

        for j in range ( l2 ):            
            for k in range ( f ):                
                y [j]    += t [ 2*j + k ] * h [ k ] * s;
                y [j+l2] += t [ 2*j + k ] * g [ k ] * s;

        l = l2;              # continue with the approximation
        t [ 0:l ] = y [ 0:l ] ;

    return y

    
"""
Run functions in this order for the EDA processing pipeline:
    1. Get_EDA
    2. IsolationForest
    3. IQR_Threshold
"""

def Get_EDA(xdf, new_fs=16):
    
    """
    This function takes your XDF file and desired sampling frequency, applies the EDA correction script (Step 1) to
    the signal as well as initial outlier removal using a +/- 3 standard deviation threshold (Step 2). 
    
    Inputs
    -XDF file containing the BioSemi Stream
    -The correct sampling frequency of the EDA signal (default = 16)
    
    Outputs
    -EDA signal that has undergone:
        1. Correction via our EDA_Correction script
        2. Decimation to return a signal at the correct sampling frequency
        3. Outlier removal using any value +/- 3 standard deviations as the threshold
        4. Interpolating values for the missing values (linear interpolation & backward/forward filling as needed)
    -The new sampling frequency
    """
    
    eeg_stream = find_stream_by_name(xdf, "BioSemi") #Finding the Biosemi stream in the XDF
    
    #Grabbing the eda streams from the EEG data
    aux1 = eeg_stream['time_series'][:,73] #index 73 is channel 1 of eda signal
    aux2 = eeg_stream['time_series'][:,74] #index 74 is channel 2 of eda signal
    timestamps = eeg_stream['time_stamps']
    fs = round(eeg_stream['info']['effective_srate']) #sampling rate
    factor= fs/new_fs
    newfs = int(fs/factor)

    #Correcting the raw EDA signal 
    eda = EDA_Correction.getEDA(aux1, aux2, fs)
    eda = eda[~np.isnan(eda)] #make sure to use eda data that does not contain NaN
    
    #Decimate signal back down to the correct sampling frequency
    timestamps_re = timestamps[0: :int(factor)]
    eda_re = eda[0: :int(factor)]
    
    #If the signals have mismatched lengths trim the the longest one to match the shorter one.
    if len(timestamps_re) > len(eda_re):
        timestamps_re = timestamps_re[0:len(eda_re)]
    elif len(eda_re) > len(timestamps_re):
        eda_re = eda_re[0:len(timestamps_re)]
    else:
        pass    
        
    #Compiling the decimated signals and timestamps into a data frame
    eda_cleaned = pd.DataFrame({'eda': eda_re, 'Timestamps': timestamps_re})
    
    #generating the thresholds in order to remove outliers 
    upper_limit = np.nanmean(eda_cleaned['eda']) + (3*np.nanstd(eda_cleaned['eda']))
    lower_limit = np.nanmean(eda_cleaned['eda']) - (3*np.nanstd(eda_cleaned['eda']))
    
    #Looping through each datapoint to remove outliers and label data as good/bad
    eda_thresh = []
    extreme = []
    for i in np.arange(0, len(eda_cleaned['eda'])):
        if eda_cleaned['eda'][i] <= lower_limit:
            x = np.nan
            y = -1
            eda_thresh.append(x)
            extreme.append(y)
        elif eda_cleaned['eda'][i] >= upper_limit: 
            x = np.nan
            y = -1
            eda_thresh.append(x)
            extreme.append(y)
        else:
            x = eda_cleaned['eda'][i]
            y = 1
            eda_thresh.append(x)
            extreme.append(y)
            
    eda_cleaned['eda_thresh'] = pd.Series(eda_thresh).astype(float) #Signal with outliers removed
    eda_cleaned['Extreme'] = pd.Series(extreme) #Binary indicating whether value i was outside of the threshold (-1) or not (1)
    eda_cleaned['eda_int'] =  eda_cleaned['eda_thresh'].interpolate(method='linear') #Linear interpolation of the signal
    eda_cleaned['eda_int'] = eda_cleaned['eda_int'].fillna(method='bfill') #back-filling the data for any missing values that interpolation didn't work on
    eda_cleaned['eda_int'] = eda_cleaned['eda_int'].fillna(method='ffill') #forward-filling the data for any missing values that interpolation didn't work on

    return eda_cleaned, newfs

def IsolationForest(eda, newfs):
    """
    This function takes the EDA signal from the Get_EDA function, windows the data, generatues features and 
    feeds them into a Isolation Forest model (Step 3). The model's output is then used to remove data classified as 
    outliers, interpolates the data, and applies a 16-point rolling median filter.
    
    Inputs
    -The EDA signal from the Get_EDA function
    -The new sampling frequency
    
    Outputs
    -EDA signal that has undergone:
        1. The EDA signal after steps 3 and 4 of the processing pipeline
    """
    
    t=0.5
    w=int(t*newfs) 
    o=int(t*newfs) #o=w no overlaps
    y=np.array(eda['eda_int'])
    eda_windows = window(y, w=w, o=o)

    #Features selected based on Subramanian, et. al., 2021 (See end of file for full citation)
    eda_range = []
    eda_std = []
    eda_firstder_mean = []
    eda_firstder_med = []
    eda_firstder_std = []
    eda_firstder_min = []
    eda_firstder_max = []
    haar_m = []
    haar_md = []
    haar_sd = []
    haar_mn = []
    haar_mx = []
    
    #Generates features for each window and appends to a list
    for i in np.arange(0, len(eda_windows)):       
        haar_wave = haarFWT(eda_windows[i], 4)

        ranges = np.nanmax(eda_windows[i]) - np.nanmin(eda_windows[i])
        eda_range.append(ranges)
        
        stdev = np.std(eda_windows[i])
        eda_std.append(stdev)
        
        eda_1st_mean = np.nanmean(np.diff(eda_windows[i]))
        eda_firstder_mean.append(eda_1st_mean)
        
        eda_1st_med = np.nanmedian(np.diff(eda_windows[i]))
        eda_firstder_med.append(eda_1st_med)
        
        eda_1st_std = np.std(np.diff(eda_windows[i]))
        eda_firstder_std.append(eda_1st_std)
        
        eda_1st_min = np.diff(eda_windows[i], 1)
        x = np.nanmin(eda_1st_min)
        eda_firstder_min.append(x)
        
        eda_1st_max = np.nanmax(np.diff(eda_windows[i], 1))
        y = np.nanmax(eda_1st_max)
        eda_firstder_max.append(y)
        
        haar_mean = np.nanmean(haar_wave)
        haar_m.append(haar_mean)
        haar_median = np.nanmedian(haar_wave)
        haar_md.append(haar_median)
        haar_std = np.std(haar_wave)
        haar_sd.append(haar_std)
        haar_min = np.nanmin(haar_wave)
        haar_mn.append(haar_min)
        haar_max = np.nanmax(haar_wave)
        haar_mx.append(haar_max)
    
    #Combines the lists of features and compiles them into a single data frame
    feature_eng = pd.DataFrame({
                                'eda_range': eda_range,
                                'gst_std': eda_std,
                                'eda_1st_mean': eda_firstder_mean,
                                'eda_1st_median': eda_firstder_med,
                                'eda_1st_std': eda_firstder_std,
                                'eda_1st_min': eda_firstder_min,
                                'eda_1st_max': eda_firstder_max, 
                                'haar_mean': haar_m,
                                'haar_median': haar_md,
                                'haar_std': haar_sd,
                                'haar_min': haar_mn,
                                'haar_max': haar_mx})
    
    #Running an Isolation Forest to find artifacts
    model = IsForest(random_state = 1, max_features=0.75, bootstrap=True)

    clf = model.fit_predict(feature_eng)
    feature_eng['scores'] = model.decision_function(feature_eng) #scores = scores generated by the model
    feature_eng['preds'] = clf #preds = the classification of the model
    feature_eng['preds'].value_counts()

    #Applying model predictions to the continuous signal from the windowed feature predictions
    IF_class = feature_eng['preds']
    IF_class2 = IF_class.repeat(8).reset_index(drop=True)
    eda['preds'] = IF_class2
    
    preds_clean = []
    for i in np.arange(0, len(eda['preds'])):
        if eda['preds'][i] == -1:
            x = np.nan
            preds_clean.append(x)
        else:
            x = eda['eda_int'][i]
            preds_clean.append(x)
           
    eda['eda_Clean'] = preds_clean
    
    #Interpolate nans
    eda['eda_Interpolated'] = eda['eda_Clean'].interpolate(methods='linear')
    eda['eda_Interpolated'] = eda['eda_Interpolated'].fillna(method='bfill')
    eda['eda_Interpolated'] = eda['eda_Interpolated'].fillna(method='ffill')
 
    #Applying a 16-point rolling median filter
    eda['eda_final'] = eda['eda_Interpolated'].rolling(16).median()
    
    #Interpolating one more time as needed to handle nans generated from the rolling median filter
    eda['eda_final'] = eda['eda_final'].interpolate(methods='linear')
    eda['eda_final'] = eda['eda_final'].fillna(method='bfill')
    eda['eda_final'] = eda['eda_final'].fillna(method='ffill')
    
    #If you want the EDA signal at any point in the process you can add them here
    filtered_data = eda[['Timestamps', 'preds', 'Extreme', 'eda_final']] 

    return filtered_data

def IQR_Threshold(eda):
    """
    This function takes your signal from the IsolationForest function and computes one last pass on outliers.
    IQR thresholds (Step 5) are generated and then used to remove any last outliers and interpolate. All classification
    variables (from the inital filtering, the random forest model, and the IQR threshold) are gathered and combined
    into a single binary variable indicating which values have been interpolated. 
    
    Inputs
    -EDA signal from the IsolationForest function
    
    Outputs
    -A DataFrame with:
        1. The processed EDA signal
        2. Timestamps for the signal
        3. A binary variable that indicates whether or not any given data point is interpolated or not
    """
    
    #Generating the Interquartile range threshold
    Q1 = np.percentile(eda['eda_final'], 25, interpolation = 'midpoint')
    Q3 = np.percentile(eda['eda_final'], 75, interpolation = 'midpoint')
    IQR = Q3 - Q1
    Lower_threshold = Q1 - (1.5*IQR)
    Upper_threshold = Q3 + (1.5*IQR)
    
    #Removing outliers based on the threshold
    outliers = []
    for i in np.arange(0, len(eda['eda_final'])):
        if eda['eda_final'].iloc[i] <= Lower_threshold:
            x = -1
            outliers.append(x)
        elif eda['eda_final'].iloc[i] >= Upper_threshold:
            x = -1
            outliers.append(x)
        else:
            x = 1
            outliers.append(x)
            
    cleaned_eda = []
    for i in np.arange(0, len(eda['eda_final'])):
        if outliers[i] == -1:
            x = np.nan
            cleaned_eda.append(x)
        else:
            x = eda['eda_final'].iloc[i]
            cleaned_eda.append(x)
    
    #Using linear interpolation and backward/forward filling to deal with any NaNs
    eda['IQR_Outliers'] = pd.Series(outliers).astype(float)
    eda['eda_final'] = pd.Series(cleaned_eda)
    eda['eda_final'] = eda['eda_final'].interpolate(method = 'slinear')
    eda['eda_final'] = eda['eda_final'].fillna(method='bfill')
    eda['eda_final'] = eda['eda_final'].fillna(method='ffill')
    
    #Generating the variable Rejected_Values to combine the IQR_Outliers, preds, and Extreme binary variables.
    #
    rejected = []
    for i in np.arange(0, len(eda['IQR_Outliers'])):
        if eda['IQR_Outliers'].iloc[i] == -1:
            x = -1
            rejected.append(x)
        elif eda['preds'].iloc[i] == -1:
            x = -1
            rejected.append(x)
        elif eda['Extreme'].iloc[i] == -1:
            x = -1
            rejected.append(x)
        else:
            x = 1
            rejected.append(x)
    
    eda['Rejected_Values'] = pd.Series(rejected)
    
    eda = eda[['Timestamps', 'Rejected_Values', 'eda_final']]
    
    return(eda)

def Process_EDA_Data(xdf, newfs=16):
    """
    This functions runs all three functions of the processesing pipeline in order and returns
    the finale data frome from the IQR_Threshold function
    
    Inputs:
        1. XDF file
        2. The desired sampling frequency of the final signal
    """
    eda1, newfs = Get_EDA(xdf)
    eda2 = IsolationForest(eda1, newfs)
    eda3 = IQR_Threshold(eda2)
    
    return(eda3)


"""
Subramanian, S., Tseng, B., Barbieri, R., & Brown, E. N. (2021).
 Unsupervised Machine Learning Methods for Artifact Removal in Electrodermal Activity. 
 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), 399–402.
 https://doi.org/10.1109/EMBC46164.2021.9630535
"""