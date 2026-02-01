"""
AI Call Center with Automatic Dialect Detection
Main application entry point
"""

import google.generativeai as genai
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
from TTS.api import TTS
import torch
import os
from dotenv import load_dotenv
import tempfile
import uuid
from session import CallSession
from dialect_detector import DialectDetector
from response_generator import ResponseGenerator

# Load environment variables
load_dotenv()

# Configuration
SAMPLE_RATE = 16000
RECORDING_DURATION = 5  # Seconds to record per turn
WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large


class CallCenterAgent:
    """Main AI Call Center Agent with dialect detection"""
    
    def __init__(self):
        """Initialize all components"""
        print("=" * 60)
        print("AI Call Center - Initializing...")
        print("=" * 60)
        
        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("✓ Gemini 2.5 Flash initialized")
        
        # Initialize Whisper (Local STT)
        print(f"Loading Whisper ({WHISPER_MODEL_SIZE})...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        print("✓ Whisper STT loaded")
        
        # Initialize TTS
        print("Loading TTS (XTTS-v2)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print(f"✓ TTS loaded on {device}")
        
        # Initialize dialect detector and response generator
        self.dialect_detector = DialectDetector(self.gemini_model)
        self.response_generator = ResponseGenerator(self.gemini_model)
        print("✓ Dialect detector & response generator ready")
        
        print("\n" + "=" * 60)
        print("All systems ready!")
        print("=" * 60 + "\n")
    
    def record_audio(self, duration=RECORDING_DURATION):
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
        
        Returns:
            numpy array of audio samples
        """
        print(f"🎤 Listening for {duration} seconds...")
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("✓ Recording complete")
        return audio.flatten()
    
    def transcribe_audio(self, audio_data):
        """
        Convert speech to text using Whisper
        
        Args:
            audio_data: Audio samples
        
        Returns:
            Transcribed text
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio_data, SAMPLE_RATE)
            temp_path = f.name
        
        print("🔄 Transcribing...")
        result = self.whisper_model.transcribe(temp_path, language='ar')
        
        # Cleanup
        os.unlink(temp_path)
        
        return result['text'].strip()
    
    def synthesize_speech(self, text, reference_audio_path=None):
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            reference_audio_path: Optional reference for voice cloning
        
        Returns:
            Path to generated audio file
        """
        print("🔊 Generating speech...")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            output_path = f.name
        
        try:
            if reference_audio_path and os.path.exists(reference_audio_path):
                # Clone customer's voice (optional feature)
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=reference_audio_path,
                    language="ar",
                    file_path=output_path
                )
            else:
                # Use default Arabic voice
                self.tts.tts_to_file(
                    text=text,
                    language="ar",
                    file_path=output_path
                )
            
            return output_path
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            # Create silent audio as fallback
            silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
            sf.write(output_path, silence, SAMPLE_RATE)
            return output_path
    
    def play_audio(self, file_path):
        """
        Play audio file through speakers
        
        Args:
            file_path: Path to audio file
        """
        try:
            audio_data, sample_rate = sf.read(file_path)
            print("▶️  Playing response...")
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"❌ Playback error: {e}")
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def handle_call(self):
        """Main call handling loop"""
        # Create new session
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
                
                # Save first audio for voice cloning (optional)
                if turn_count == 1 and reference_audio_path is None:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        sf.write(f.name, audio, SAMPLE_RATE)
                        reference_audio_path = f.name
                    print("✓ Voice sample saved for cloning")
                
                # 2. TRANSCRIBE to text
                customer_text = self.transcribe_audio(audio)
                
                if not customer_text or len(customer_text) < 2:
                    print("⚠️  No speech detected, please try again")
                    continue
                
                print(f"📝 Customer: {customer_text}")
                
                # 3. DETECT DIALECT (only if not locked)
                if not session.dialect_locked:
                    dialect, confidence = self.dialect_detector.detect(customer_text)
                    session.lock_dialect(dialect, confidence)
                
                # Display current dialect status
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
            
            # Show call statistics
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
        # Initialize agent
        agent = CallCenterAgent()
        
        # Handle call
        agent.handle_call()
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
