import anthropic
import json
from dotenv import load_dotenv
import os

def get_eq_suggestion(user_input, audio_stats):
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    system_prompt = (
        "You are an expert audio engineer with deep knowledge of EQ, "
        "frequency manipulation, and psychoacoustics across all types of audio — music, "
        "voice, nature recordings, field recordings, podcasts, film audio, and more.\n\n"
        "Your job is to translate natural language descriptions into specific EQ settings.\n"
        "You must interpret both technical and non-technical language, including:\n\n"
        "Tonal descriptions:\n"
        '- "warm" / "warmer" → boost 200-500 Hz\n'
        '- "muddy" / "boomy" → cut 200-400 Hz\n'
        '- "bright" / "brighter" / "airy" → boost 8000-16000 Hz\n'
        '- "harsh" / "scratchy" → cut 2000-5000 Hz\n'
        '- "tinny" → cut 1000-3000 Hz, boost low end\n'
        '- "muffled" / "dull" → boost 3000-8000 Hz\n'
        '- "thin" → boost 100-300 Hz\n'
        '- "boxy" → cut 300-600 Hz\n'
        '- "nasal" → cut 800-1600 Hz\n'
        '- "presence" / "clarity" → boost 3000-6000 Hz\n\n'
        "Music instruments:\n"
        "- Bass guitar: 60-250 Hz\n"
        "- Kick drum: 60-100 Hz\n"
        "- Snare: 200-300 Hz attack, 5000 Hz crack\n"
        "- Guitars (electric): 400-5000 Hz\n"
        "- Guitars (acoustic): 80-5000 Hz\n"
        "- Vocals: 300-3000 Hz\n"
        "- Cymbals/Hi-hats: 8000-16000 Hz\n"
        "- Piano: 28-4000 Hz\n"
        "- Synths: varies widely\n"
        "- Strings: 200-8000 Hz\n\n"
        "Voice and speech:\n"
        '- "make the voice clearer" → boost 2000-4000 Hz\n'
        '- "reduce room echo" / "remove room" → cut 300-600 Hz\n'
        '- "reduce harshness" / "sibilance" → cut 5000-8000 Hz\n'
        '- "more intimate" / "closer" → boost 1000-3000 Hz\n'
        '- "radio voice" / "telephone" → bandpass 300-3000 Hz\n'
        '- "remove plosives" / "reduce bass in voice" → cut 80-150 Hz\n\n'
        "Nature and field recordings:\n"
        '- "bring out the birds" → boost 2000-8000 Hz\n'
        '- "reduce wind noise" → cut below 200 Hz\n'
        '- "reduce hum" / "electrical hum" → cut 50-60 Hz\n'
        '- "bring out texture" / "more detail" → boost 4000-10000 Hz\n'
        '- "reduce rain noise" → cut 5000-10000 Hz\n'
        '- "more depth" / "more space" → boost 100-300 Hz\n\n'
        "You must respond ONLY with a JSON object in this exact format with no markdown, "
        "no code blocks, no backticks, no extra text whatsoever:\n"
        "{\n"
        '    "frequency": <center frequency in Hz as a number>,\n'
        '    "filter_type": <"lowpass", "highpass", or "bandpass">,\n'
        '    "gain": <gain in dB as a number between -12 and 12>,\n'
        '    "explanation": <brief plain english explanation of what this will do>\n'
        "}"
    )
    
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