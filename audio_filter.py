import numpy as np
from scipy import signal

def apply_filter(audio, sample_rate, result):
    nyquist = sample_rate / 2
    filtered_audio = audio.copy()
    
    for band in result["bands"]:
        frequency = band["frequency"]
        filter_type = band["filter_type"]
        gain = band["gain"]
        
        normalized_freq = frequency / nyquist
        normalized_freq = max(0.001, min(0.999, normalized_freq))
        
        if filter_type == "lowpass":
            b, a = signal.butter(4, normalized_freq, btype="low")
        elif filter_type == "highpass":
            b, a = signal.butter(4, normalized_freq, btype="high")
        elif filter_type == "bandpass":
            low = max(0.001, (frequency * 0.75) / nyquist)
            high = min(0.999, (frequency * 1.25) / nyquist)
            b, a = signal.butter(4, [low, high], btype="band")
        
        filtered_audio = signal.lfilter(b, a, filtered_audio)
        gain_linear = 10 ** (gain / 20)
        filtered_audio = filtered_audio * gain_linear
        
        # safety check after each band
        if not np.all(np.isfinite(filtered_audio)):
            print(f"Warning: filter at {frequency}Hz produced invalid values, skipping")
            filtered_audio = audio.copy()
            continue
    
    # final normalization to prevent clipping
    max_val = np.max(np.abs(filtered_audio))
    if max_val > 0:
        filtered_audio = filtered_audio / max_val * 0.9
    
    return filtered_audio