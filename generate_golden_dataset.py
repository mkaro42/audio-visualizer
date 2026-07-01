import numpy as np
import soundfile as sf
import json
import os
from audio_filter import apply_filter
from eq_llm import get_eq_suggestion
from dotenv import load_dotenv

load_dotenv()

# Generate white noise
SAMPLE_RATE = 44100
DURATION = 5  # seconds
output_dir = "golden_dataset"

os.makedirs(output_dir, exist_ok=True)

def generate_white_noise(duration, sample_rate):
    noise = np.random.uniform(-1, 1, duration * sample_rate).astype(np.float32)
    return noise

# Define golden filters
golden_filters = {
    "highpass_5k": {
        "prompt": "apply a highpass filter above 5000 hz",
        "effect_name": "Highpass 5k",
        "bands": [{"frequency": 5000, "filter_type": "highpass", "gain": 0}]
    },
    "lowpass_5k": {
        "prompt": "apply a lowpass filter below 5000 hz",
        "effect_name": "Lowpass 5k",
        "bands": [{"frequency": 5000, "filter_type": "lowpass", "gain": 0}]
    },
    "bandpass_1k_2k": {
        "prompt": "apply a bandpass filter between 1000 and 2000 hz",
        "effect_name": "Bandpass 1k-2k",
        "bands": [{"frequency": 1500, "filter_type": "bandpass", "gain": 0}]
    },
    "bandpass_combo": {
        "prompt": "apply a bandpass between 1000 and 2000 hz and another between 7000 and 8000 hz",
        "effect_name": "Bandpass Combo 1k-2k and 7k-8k",
        "bands": [
            {"frequency": 1500, "filter_type": "bandpass", "gain": 0},
            {"frequency": 7500, "filter_type": "bandpass", "gain": 0}
        ]
    },
    "band_reject": {
        "prompt": "apply a band reject filter removing frequencies between 1000 and 5000 hz",
        "effect_name": "Band Reject 1k-5k",
        "bands": [
            {"frequency": 500, "filter_type": "lowpass", "gain": 0},
            {"frequency": 5000, "filter_type": "highpass", "gain": 0}
        ]
    }
}

if __name__ == "__main__":
    audio_stats = {"sample_rate": SAMPLE_RATE, "duration": DURATION}
    
    print("Generating white noise...")
    white_noise = generate_white_noise(DURATION, SAMPLE_RATE)
    sf.write(f"{output_dir}/white_noise.wav", white_noise, SAMPLE_RATE)
    print("White noise saved!")

    llm_outputs = {}

    for name, params in golden_filters.items():
        print(f"\nProcessing {name}...")
        
        # apply golden filter
        filtered = apply_filter(white_noise, SAMPLE_RATE, params)
        sf.write(f"{output_dir}/{name}.wav", filtered, SAMPLE_RATE)
        print(f"Saved {name}.wav")
        
        # get raw LLM output for same prompt
        print(f"Getting LLM output for: {params['prompt']}")
        llm_result = get_eq_suggestion(params["prompt"], audio_stats)
        llm_outputs[name] = {
            "prompt": params["prompt"],
            "llm_response": llm_result
        }
        print(f"LLM returned: {llm_result['effect_name']}")

    # save ground truth JSON
    with open(f"{output_dir}/golden_params.json", "w") as f:
        json.dump(golden_filters, f, indent=4)
    print("\nGolden params saved!")

    # save raw LLM outputs JSON
    with open(f"{output_dir}/llm_outputs.json", "w") as f:
        json.dump(llm_outputs, f, indent=4)
    print("LLM outputs saved!")

    print(f"\nGolden dataset generated successfully!")
    print(f"Files saved to {output_dir}/")