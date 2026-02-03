"""
AI Call Center with Enhanced STT and Optimized Audio
- Auto-stops recording when you finish speaking (VAD)
- Faster, more natural TTS output
"""

import google.generativeai as genai
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import os
from dotenv import load_dotenv
import tempfile
import uuid
from session import CallSession
from dialect_detector import DialectDetector
from response_generator import ResponseGenerator
from enhanced_stt import EnhancedSTT
from vad_recorder import VADRecorder, SimpleRecorder
from optimized_tts import OptimizedTTS
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig

add_safe_globals([XttsConfig])

# Load environment variables
load_dotenv()

# Configuration
SAMPLE_RATE = 16000
WHISPER_MODEL_SIZE = "medium"  # medium for better accuracy

# Recording settings
USE_VAD = True  # Auto-stop when you finish speaking (recommended)
MAX_RECORDING_DURATION = 30  # Maximum seconds if using VAD
SIMPLE_RECORDING_DURATION = 10  # Fixed duration if not using VAD
SILENCE_DURATION = 1.5  # Seconds of silence before auto-stop

# TTS settings
TTS_SPEED = 1.2  # 1.0=normal, 1.2=faster and more natural
TTS_TEMPERATURE = 0.7  # 0.5=neutral, 0.7=natural, 0.9=very expressive


class CallCenterAgent:
    """Main AI Call Center Agent with Enhanced STT"""
    
    def __init__(self):
        """Initialize all components"""
        print("=" * 60)
        print("AI Call Center - Enhanced Audio - Initializing...")
        print("=" * 60)
        
        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("✓ Gemini 2.5 Flash initialized")
        
        # Initialize Enhanced STT
        print(f"Loading Enhanced STT...")
        self.stt = EnhancedSTT(model_size=WHISPER_MODEL_SIZE, enable_noise_reduction=True)
        print(f"✓ Enhanced STT loaded ({WHISPER_MODEL_SIZE} model)")
        
        # Initialize Audio Recorder with VAD
        if USE_VAD:
            self.recorder = VADRecorder(aggressiveness=2)
            print("✓ Voice Activity Detection (VAD) enabled")
            print("  - Auto-stops when you finish speaking")
            print(f"  - Max duration: {MAX_RECORDING_DURATION}s")
            print(f"  - Silence threshold: {SILENCE_DURATION}s")
        else:
            self.recorder = SimpleRecorder()
            print(f"✓ Simple recorder (fixed {SIMPLE_RECORDING_DURATION}s duration)")
        
        # Initialize Optimized TTS
        print("Loading Optimized TTS...")
        self.tts = OptimizedTTS(use_gpu=True)
        print(f"✓ Optimized TTS loaded")
        print(f"  - Speed: {TTS_SPEED}x (faster = more natural)")
        print(f"  - Temperature: {TTS_TEMPERATURE} (natural expressiveness)")
        
        # Initialize dialect detector and response generator
        self.dialect_detector = DialectDetector(self.gemini_model)
        self.response_generator = ResponseGenerator(self.gemini_model)
        print("✓ Dialect detector & response generator ready")
        
        print("\n" + "=" * 60)
        print("All systems ready!")
        print("=" * 60 + "\n")
    
    def record_audio(self):
        """Record audio from microphone (with VAD or fixed duration)"""
        if USE_VAD:
            # Use VAD - auto-stops when you finish speaking
            audio = self.recorder.record_with_vad(
                max_duration=MAX_RECORDING_DURATION,
                silence_duration=SILENCE_DURATION
            )
        else:
            # Use fixed duration
            audio = self.recorder.record(duration=SIMPLE_RECORDING_DURATION)
        
        return audio
    
    def transcribe_audio(self, audio_data):
        """
        Convert speech to text using Enhanced STT
        (replaces basic Whisper transcription)
        """
        print("🔄 Transcribing with enhancements...")
        text = self.stt.transcribe(audio_data)
        return text
    
    def synthesize_speech(self, text, reference_audio_path=None):
        """Convert text to speech with optimized settings"""
        print("🔊 Generating speech...")
        
        try:
            # Use optimized TTS with natural speed and expressiveness
            output_path = self.tts.synthesize(
                text=text,
                reference_voice=reference_audio_path,
                speed=TTS_SPEED,  # Faster = more natural
                temperature=TTS_TEMPERATURE  # Natural expressiveness
            )
            
            return output_path
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            # Create silent audio as fallback
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                output_path = f.name
            silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
            sf.write(output_path, silence, SAMPLE_RATE)
            return output_path
    
    def play_audio(self, file_path):
        """Play audio file through speakers"""
        try:
            audio_data, sample_rate = sf.read(file_path)
            print("▶️  Playing response...")
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"❌ Playback error: {e}")
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def handle_call(self):
        """Main call handling loop"""
        session = CallSession(str(uuid.uuid4()))
        
        print("\n" + "=" * 60)
        print("NEW CALL STARTED")
        print(f"Session ID: {session.session_id}")
        print("=" * 60)
        print("\nPress Ctrl+C to end call")
        print("Press Enter to start each turn\n")
        
        reference_audio_path = None
        turn_count = 0
        
        try:
            while True:
                input("\n[Press Enter when customer is ready to speak...]")
                turn_count += 1
                print(f"\n--- Turn {turn_count} ---")
                
                # 1. RECORD customer speech
                audio = self.record_audio()
                
                # Save first audio for voice cloning
                if turn_count == 1 and reference_audio_path is None:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        sf.write(f.name, audio, SAMPLE_RATE)
                        reference_audio_path = f.name
                    print("✓ Voice sample saved for cloning")
                
                # 2. TRANSCRIBE to text (Enhanced STT)
                customer_text = self.transcribe_audio(audio)
                
                if not customer_text or len(customer_text) < 2:
                    print("⚠️  No speech detected, please try again")
                    continue
                
                print(f"📝 Customer: {customer_text}")
                
                # 3. DETECT DIALECT
                if not session.dialect_locked:
                    dialect, confidence = self.dialect_detector.detect(customer_text)
                    session.lock_dialect(dialect, confidence)
                
                # Display dialect status
                if session.dialect_locked:
                    print(f"🔒 Dialect: {session.detected_dialect} (locked)")
                else:
                    print(f"🔍 Dialect: {session.detected_dialect or 'detecting...'} "
                          f"(confidence: {session.dialect_confidence:.2f})")
                
                # 4. GENERATE RESPONSE in dialect
                response_text = self.response_generator.generate(
                    user_query=customer_text,
                    dialect=session.detected_dialect or 'msa',
                    context=session.get_context()
                )
                print(f"💬 Agent: {response_text}")
                
                # Store conversation
                session.add_interaction(customer_text, response_text)
                
                # 5. SYNTHESIZE speech
                audio_path = self.synthesize_speech(response_text, reference_audio_path)
                
                # 6. PLAY response
                self.play_audio(audio_path)
                
                print(f"✓ Turn {turn_count} complete")
        
        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("CALL ENDED")
            print("=" * 60)
            
            # Show statistics
            stats = session.get_stats()
            print(f"\nCall Statistics:")
            print(f"  Session ID: {stats['session_id']}")
            print(f"  Total turns: {stats['turns']}")
            print(f"  Detected dialect: {stats['dialect']}")
            print(f"  Dialect locked: {'Yes' if stats['locked'] else 'No'}")
            print(f"  Final confidence: {stats['confidence']:.2f}")
            
            # Cleanup
            if reference_audio_path and os.path.exists(reference_audio_path):
                os.unlink(reference_audio_path)
            
            print("\n📞 Thank you for using AI Call Center\n")


def main():
    """Main entry point"""
    try:
        agent = CallCenterAgent()
        agent.handle_call()
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
