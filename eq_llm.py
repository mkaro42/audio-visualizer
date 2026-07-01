import anthropic
import json
from dotenv import load_dotenv
import os
from langsmith import traceable 

load_dotenv()

print(f"LangSmith API Key found: {bool(os.getenv('LANGSMITH_API_KEY'))}")
print(f"LangSmith Project: {os.getenv('LANGSMITH_PROJECT')}")
print(f"LangSmith Tracing: {os.getenv('LANGSMITH_TRACING')}")

@traceable
def get_eq_suggestion(user_input, audio_stats):
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    system_prompt = """You are an expert audio engineer and DSP (Digital Signal Processing) specialist.
Your job is to translate natural language descriptions into EQ and filter settings.

You handle two types of requests:

1. SINGLE BAND EQ — simple requests like "boost the bass" or "cut the highs"
2. MULTI BAND EFFECTS — complex requests like "make it sound underwater" or "make it vintage"

For complex effect requests think like an audio engineer about the physics:
- "underwater" → water absorbs highs rapidly, low mids resonate
- "vintage radio" → limited hardware response 300-3000Hz, midrange presence
- "large cave" → lows resonate, highs absorbed by rock walls
- "old telephone" → narrow bandpass 400-3400Hz, nasal character
- "warm" / "warmer" → boost 200-500 Hz
- "muddy" / "boomy" → cut 200-400 Hz
- "bright" / "airy" → boost 8000-16000 Hz
- "harsh" / "scratchy" → cut 2000-5000 Hz
- "tinny" → cut 1000-3000 Hz, boost low end
- "muffled" / "dull" → boost 3000-8000 Hz

Music instruments:
- Bass guitar: 60-250 Hz
- Kick drum: 60-100 Hz
- Guitars: 400-5000 Hz
- Vocals: 300-3000 Hz
- Cymbals/Hi-hats: 8000-16000 Hz
- Piano: 28-4000 Hz

Voice and speech:
- "make the voice clearer" → boost 2000-4000 Hz
- "reduce room echo" → cut 300-600 Hz
- "reduce harshness" → cut 5000-8000 Hz
- "radio voice" → bandpass 300-3000 Hz

Nature and field recordings:
- "bring out the birds" → boost 2000-8000 Hz
- "reduce wind noise" → cut below 200 Hz
- "reduce hum" → cut 50-60 Hz

You must respond ONLY with a JSON object in this exact format,
no markdown, no code blocks, no backticks, no extra text:
{
    "effect_name": <short name for the effect>,
    "bands": [
        {
            "frequency": <center frequency in Hz as a number>,
            "filter_type": <"lowpass", "highpass", or "bandpass">,
            "gain": <gain in dB as a number between -24 and 24>
        }
    ],
    "explanation": <plain english explanation of what this will do>
}"""
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": f"Audio stats: {audio_stats}\nUser request: {user_input}"}
        ]
    )
    
    response_text = message.content[0].text
    response_text = response_text.strip()
    response_text = response_text.replace("```json", "").replace("```", "").strip()
    #added to see the raw claude response 
    #print(f"Raw response: {response_text}")
    return json.loads(response_text)

##testing function

#if __name__ == "__main__":
    test_stats = {
        "sample_rate": 44100,
        "duration": 3.5
    }
    
    user_input = input("Enter your EQ request: ")
    result = get_eq_suggestion(user_input, test_stats)
    
    print(f"\nEQ Suggestion:")
    print(f"Frequency: {result['frequency']} Hz")
    print(f"Filter type: {result['filter_type']}")
    print(f"Gain: {result['gain']} dB")
    print(f"Explanation: {result['explanation']}")