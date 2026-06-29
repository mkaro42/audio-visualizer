import librosa 
import numpy as np 
import matplotlib.pyplot as plt
import soundfile as sf 

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


if __name__ == "__main__":
    file_path = input("Enter the path to your audio file: ")
    audio, sample_rate = load_audio(file_path)
    plot_waveform(audio, sample_rate)
    plot_spectrogram(audio, sample_rate)

## testing waveform file path ---> C:\Users\MaxKarolyi\OneDrive - NEWITY\Desktop\ocean.wav