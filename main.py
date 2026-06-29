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

if __name__ == "__main__":
    file_path = input("Enter the path to your audio file: ")
    audio, sample_rate = load_audio(file_path)

