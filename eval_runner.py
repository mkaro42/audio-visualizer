import json
import numpy as np
import soundfile as sf
from langsmith import Client, traceable
from langsmith.evaluation import evaluate
from eq_llm import get_eq_suggestion
from audio_filter import apply_filter
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

SAMPLE_RATE = 44100
GOLDEN_DIR = "golden_dataset"

# load golden params
with open(f"{GOLDEN_DIR}/golden_params.json", "r") as f:
    golden_filters = json.load(f)

# --- Evaluators ---

def evaluate_filter_type(run, example):
    expected_bands = example.outputs["bands"]
    actual_bands = run.outputs["bands"]
    
    expected_types = [b["filter_type"] for b in expected_bands]
    actual_types = [b["filter_type"] for b in actual_bands]
    
    correct = sum(1 for e, a in zip(expected_types, actual_types) if e == a)
    score = correct / max(len(expected_types), len(actual_types))
    
    return {"key": "filter_type_accuracy", "score": score}

def evaluate_frequency_accuracy(run, example):
    expected_bands = example.outputs["bands"]
    actual_bands = run.outputs["bands"]
    
    scores = []
    for expected, actual in zip(expected_bands, actual_bands):
        expected_freq = expected["frequency"]
        actual_freq = actual["frequency"]
        
        # score based on how close the frequency is (within 20% is good)
        diff = abs(expected_freq - actual_freq) / expected_freq
        score = max(0, 1 - diff)
        scores.append(score)
    
    return {"key": "frequency_accuracy", "score": np.mean(scores) if scores else 0}

def evaluate_band_count(run, example):
    expected_count = len(example.outputs["bands"])
    actual_count = len(run.outputs["bands"])
    
    score = 1.0 if expected_count == actual_count else 0.0
    
    return {"key": "band_count_match", "score": score}

# --- Dataset creation ---

def create_langsmith_dataset():
    client = Client()
    
    dataset_name = "audio-eq-golden-dataset3"
    
    datasets = list(client.list_datasets())
    existing = [d for d in datasets if d.name == dataset_name]
    
    if existing:
        print(f"Dataset '{dataset_name}' already exists, using existing.")
        return existing[0]
    
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Golden dataset for audio EQ evaluation",
        metadata={"project": os.getenv("LANGSMITH_PROJECT")}
    )
    
    for name, params in golden_filters.items():
        client.create_example(
            inputs={"prompt": params["prompt"]},
            outputs={
                "effect_name": params["effect_name"],
                "bands": params["bands"]
            },
            dataset_id=dataset.id
        )
        print(f"Added example: {name}")
    
    print(f"\nDataset '{dataset_name}' created with {len(golden_filters)} examples!")
    return dataset
# --- Target function ---

@traceable
def eq_target(inputs):
    audio_stats = {"sample_rate": SAMPLE_RATE, "duration": 5.0}
    result = get_eq_suggestion(inputs["prompt"], audio_stats)
    return result

# --- Run evals ---

if __name__ == "__main__":
    print("Setting up LangSmith dataset...")
    dataset = create_langsmith_dataset()
    
    print("\nRunning evaluations...")
    results = evaluate(
        eq_target,
        data=dataset.name,
        evaluators=[
            evaluate_filter_type,
            evaluate_frequency_accuracy,
            evaluate_band_count
        ],
        experiment_prefix="golden-filter-eval"
    )
    
    print("\nEvaluation complete! Check LangSmith portal for results.")