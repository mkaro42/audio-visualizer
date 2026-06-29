import anthropic
import json
from dotenv import load_dotenv
import os

def get_eq_suggestion(user_input, audio_stats):
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    system_prompt = """You are an expert audio engineer and EQ specialist. 
    Your job is to translate natural language descriptions into specific EQ settings.
    
    Common frequency ranges:
    - Sub bass: 20-60 Hz
    - Bass: 60-250 Hz
    - Low midrange: 250-500 Hz
    - Midrange: 500-2000 Hz
    - Upper midrange: 2000-4000 Hz
    - Presence: 4000-6000 Hz
    - Brilliance/Air: 6000-20000 Hz
    
    Common instruments:
    - Bass guitar: 60-250 Hz
    - Kick drum: 60-100 Hz
    - Guitars: 400-5000 Hz
    - Vocals: 300-3000 Hz
    - Cymbals/Hi-hats: 8000-16000 Hz
    - Piano: 28-4000 Hz
    
    You must respond ONLY with a JSON object in this exact format, no other text:
    {
        "frequency": <center frequency in Hz as a number>,
        "filter_type": <"lowpass", "highpass", or "bandpass">,
        "gain": <gain in dB as a number between -12 and 12>,
        "explanation": <brief plain english explanation of what this will do>
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
    print(f"Raw response: {response_text}")
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