import librosa 
import numpy as np 
import matplotlib.pyplot as plt
import soundfile as sf 

from eq_llm import get_eq_suggestion
from audio_filter import apply_filter

## Audio Loading ##

def load_audio(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=audio, sr = sample_rate)

    print(f"Audio loaded successfully!")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Samples: {len(audio)}")

    return audio, sample_rate

##plot my time domain waveform ##
def plot_waveform(audio, sample_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.title("Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

##plot the spectrogram of the waveform ##
def plot_spectrogram(audio, sample_rate):
    print("Computing STFT...")
    stft = librosa.stft(audio)
    print("Converting to dB...")
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    print("Plotting...")
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.show()

## plotting the frequency spectrum ##

def plot_frequency_spectrum(audio, sample_rate):
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(magnitude), d=1/sample_rate)
    
    # only plot positive frequencies
    positive_freq_idx = frequency > 0
    frequency = frequency[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]
    
    plt.figure(figsize=(12, 4))
    plt.plot(frequency, magnitude)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.show()

##function to plot all at once##

def plot_all(audio, sample_rate):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Waveform
    plt.sca(axes[0])
    librosa.display.waveshow(audio, sr=sample_rate)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    
    # Spectrogram
    stft = librosa.stft(audio)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    plt.sca(axes[1])
    librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set_title("Spectrogram")
    
    # Frequency Spectrum
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(magnitude), d=1/sample_rate)
    positive_freq_idx = frequency > 0
    axes[2].plot(frequency[positive_freq_idx], magnitude[positive_freq_idx])
    axes[2].set_title("Frequency Spectrum")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Magnitude")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = input("Enter the path to your audio file: ")
    audio, sample_rate = load_audio(file_path)
    
    audio_stats = {
        "sample_rate": sample_rate,
        "duration": librosa.get_duration(y=audio, sr=sample_rate)
    }
    
    plot_all(audio, sample_rate)
    
    print("\nEQ Assistant ready! Type your EQ request or 'quit' to exit.")
    while True:
        user_input = input("\nEQ Request: ")
        if user_input.lower() == "quit":
            break
    
        result = get_eq_suggestion(user_input, audio_stats)
        print(f"\nEQ Suggestion:")
        print(f"Frequency: {result['frequency']} Hz")
        print(f"Filter type: {result['filter_type']}")
        print(f"Gain: {result['gain']} dB")
        print(f"Explanation: {result['explanation']}")
    
        filtered_audio = apply_filter(
            audio, 
            sample_rate, 
            result['frequency'], 
            result['filter_type'], 
            result['gain']
        )
    
        print("\nUpdating visualizations...")
        plot_all(filtered_audio, sample_rate)

## testing waveform file path ---> C:\Users\MaxKarolyi\OneDrive - NEWITY\Desktop\ocean.wav