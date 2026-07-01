import librosa 
import numpy as np 
import matplotlib.pyplot as plt
import soundfile as sf 
import pygame 
import os

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
    
     # normalize audio for plotting
    audio_normalized = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Waveform
    plt.sca(axes[0])
    librosa.display.waveshow(audio_normalized, sr=sample_rate)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    
    # Spectrogram
    stft = librosa.stft(audio_normalized)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    plt.sca(axes[1])
    librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set_title("Spectrogram")
    
    # Frequency Spectrum
    fft = np.fft.fft(audio_normalized)
    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(magnitude), d=1/sample_rate)
    positive_freq_idx = frequency > 0
    axes[2].plot(frequency[positive_freq_idx], magnitude[positive_freq_idx])
    axes[2].set_title("Frequency Spectrum")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Magnitude")
    
    plt.tight_layout()
    plt.show()

def play_audio(audio, sample_rate):
    pygame.mixer.init(frequency=sample_rate)
    
    tmp_path = "temp_playback.wav"
    audio_out = np.array(audio, dtype=np.float32)
    sf.write(tmp_path, audio_out, sample_rate, subtype='PCM_16')
    
    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()
    
    print("Playing audio... press Enter to stop.")
    input()
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(tmp_path)

if __name__ == "__main__":
    file_path = input("Enter the path to your audio file: ")
    audio, sample_rate = load_audio(file_path)
    
    audio_original = np.array(audio, dtype=np.float32)
    audio_filtered = [np.array(audio, dtype=np.float32)]
    
    audio_stats = {
        "sample_rate": sample_rate,
        "duration": librosa.get_duration(y=audio, sr=sample_rate)
    }
    
    plot_all(audio_original, sample_rate)
    
    print("""
╔════════════════════════════════════════╗
║         AUDIO EQ ASSISTANT            ║
╠════════════════════════════════════════╣
║  Commands:                            ║
║   • Type anything  → EQ request       ║
║   • play           → hear filtered    ║
║   • play original  → hear original    ║
║   • reset          → restore original ║
║   • quit           → exit             ║
╚════════════════════════════════════════╝
""")
    while True:
        user_input = input("EQ Request (or command): ")
        
        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "reset":
            audio_filtered[0] = np.array(audio_original, dtype=np.float32)
            print("Audio reset to original!")
            plot_all(audio_filtered[0], sample_rate)
            continue
        elif user_input.lower() == "play":
            play_audio(audio_filtered[0], sample_rate)
            continue
        elif user_input.lower() == "play original":
            play_audio(audio_original, sample_rate)
            continue
        
        result = get_eq_suggestion(user_input, audio_stats)
        print(f"\nEffect: {result['effect_name']}")
        print(f"Explanation: {result['explanation']}")
        
        audio_filtered[0] = apply_filter(audio_filtered[0], sample_rate, result)
        plot_all(audio_filtered[0], sample_rate)

        print(f"\n✓ Effect applied: {result['effect_name']}")
        print(f"  {result['explanation']}")
        print(f"\n  Type 'play' to hear it or make another request.")


## testing waveform file path ---> C:\Users\MaxKarolyi\OneDrive - NEWITY\Desktop\ocean.wav
## ---- > C:\Users\MaxKarolyi\OneDrive - NEWITY\Desktop\454573__jmanwierd__ambient-rain-and-thunder.m4a
## ----- > C:\Users\MaxKarolyi\OneDrive - NEWITY\Desktop\587907__timothyd4y__forest-at-night-ambience.wav
## ------ > C:\Users\MaxKarolyi\OneDrive - NEWITY\Desktop\353416__squashy555__seagull-on-beach.mp3
## -------> C:\Users\MaxKarolyi\OneDrive - NEWITY\Desktop\327447__freelibras__birds-chirping-on-a-tree-3.wav