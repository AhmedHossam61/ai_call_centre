import google.generativeai as genai
import whisper
from elevenlabs import generate, set_api_key
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize services
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
set_api_key(os.getenv('ELEVENLABS_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash')
whisper_model = whisper.load_model("base")

# Define accent voices (ElevenLabs voice IDs or Azure voice names)
ACCENTS = {
    'egyptian': 'voice_id_or_name_for_egyptian',
    'sudanese': 'voice_id_or_name_for_sudanese',
    'levantine': 'voice_id_or_name_for_levantine',
    'gulf': 'voice_id_or_name_for_gulf'
}

def process_voice_input(audio_file, accent='egyptian'):
    # 1. Transcribe
    result = whisper_model.transcribe(audio_file, language='ar')
    user_text = result['text']
    
    # 2. Get AI response
    response = model.generate_content(user_text)
    ai_text = response.text
    
    # 3. Generate speech with accent
    audio = generate(
        text=ai_text,
        voice=ACCENTS[accent],
        model="eleven_multilingual_v2"
    )
    
    return audio, ai_text

# Example usage
# audio_output, text_response = process_voice_input("user_audio.wav", accent='egyptian')