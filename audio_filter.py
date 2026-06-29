import numpy as np
from scipy import signal

def apply_filter(audio, sample_rate, frequency, filter_type, gain):
    nyquist = sample_rate / 2
    normalized_freq = frequency / nyquist
    
    if filter_type == "lowpass":
        b, a = signal.butter(4, normalized_freq, btype="low")
    elif filter_type == "highpass":
        b, a = signal.butter(4, normalized_freq, btype="high")
    elif filter_type == "bandpass":
        low = (frequency * 0.75) / nyquist
        high = (frequency * 1.25) / nyquist
        b, a = signal.butter(4, [low, high], btype="band")
    
    filtered_audio = signal.lfilter(b, a, audio)
    gain_linear = 10 ** (gain / 20)
    filtered_audio = filtered_audio * gain_linear
    
    return filtered_audio