"""
Optimized Text-to-Speech Module
Faster, more natural Arabic speech synthesis
"""

from TTS.api import TTS
import torch
import numpy as np
import soundfile as sf
import tempfile
import os


class OptimizedTTS:
    """
    Optimized TTS for natural, fast Arabic speech
    """
    
    def __init__(self, use_gpu=True):
        """
        Initialize Optimized TTS
        
        Args:
            use_gpu: Use GPU if available (much faster)
        """
        print("🔊 Loading Optimized TTS...")
        
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        
        # Load XTTS model
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.device = device
        
        print(f"✓ Optimized TTS loaded on {device}")
        if device == "cpu":
            print("⚠️  Running on CPU - TTS will be slower. Use GPU for best performance.")
    
    def synthesize(self, 
                   text, 
                   output_path=None,
                   reference_voice=None,
                   speed=1.2,
                   temperature=0.7):
        """
        Synthesize speech from text
        
        Args:
            text: Arabic text to synthesize
            output_path: Path to save audio (None = return audio array)
            reference_voice: Path to reference audio for voice cloning (optional)
            speed: Speech speed multiplier (1.0=normal, 1.2=faster, 0.8=slower)
                   Recommended: 1.2-1.3 for natural pace
            temperature: Expressiveness (0.1=robotic, 0.7=natural, 1.0=very expressive)
                        Recommended: 0.7 for natural speech
        
        Returns:
            Path to audio file or audio array
        """
        # Clean text
        text = text.strip()
        if not text:
            print("⚠️  Empty text, skipping TTS")
            return None
        
        # Create temp output if not specified
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        try:
            # Synthesize with optimized settings
            if reference_voice and os.path.exists(reference_voice):
                # Voice cloning mode
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=reference_voice,
                    language="ar",
                    file_path=output_path,
                    speed=speed,
                    temperature=temperature,
                    enable_text_splitting=True  # Handle long text better
                )
            else:
                # Default Arabic voice
                self.tts.tts_to_file(
                    text=text,
                    language="ar",
                    file_path=output_path,
                    speed=speed,
                    temperature=temperature,
                    enable_text_splitting=True
                )
            
            # Speed up audio post-processing (optional)
            if speed != 1.0:
                audio, sr = sf.read(output_path)
                # Audio is already sped up by XTTS, no need for additional processing
            
            return output_path
            
        except Exception as e:
            print(f"❌ TTS synthesis error: {e}")
            # Create silent audio as fallback
            silence = np.zeros(16000 * 2, dtype=np.float32)
            sf.write(output_path, silence, 16000)
            return output_path
    
    def synthesize_fast(self, text, reference_voice=None):
        """
        Quick synthesis with default optimizations
        
        Args:
            text: Text to synthesize
            reference_voice: Optional reference audio path
        
        Returns:
            Path to audio file
        """
        return self.synthesize(
            text=text,
            reference_voice=reference_voice,
            speed=1.2,  # Slightly faster for natural pace
            temperature=0.7  # Natural expressiveness
        )


class FastTTS:
    """
    Ultra-fast TTS using Piper (if available)
    Trade-off: No voice cloning, but 10x faster
    """
    
    def __init__(self):
        """Initialize Fast TTS"""
        try:
            from piper import PiperVoice
            print("🔊 Loading Fast TTS (Piper)...")
            # Note: You need to download Arabic Piper model
            # This is a placeholder - adjust model path as needed
            self.voice = None
            self.available = False
            print("⚠️  Piper model not configured. Using XTTS instead.")
        except ImportError:
            self.available = False
            print("⚠️  Piper not installed. Install with: pip install piper-tts")
    
    def synthesize(self, text, output_path=None):
        """Fast synthesis (if available)"""
        if not self.available:
            raise NotImplementedError("Fast TTS not available")
        
        # Piper synthesis would go here
        pass


# Convenience functions
def create_optimized_tts(use_gpu=True):
    """
    Create optimized TTS instance
    
    Args:
        use_gpu: Use GPU if available
    
    Returns:
        OptimizedTTS instance
    """
    return OptimizedTTS(use_gpu=use_gpu)


# TTS Settings Presets
TTS_PRESETS = {
    'natural': {
        'speed': 1.2,
        'temperature': 0.7,
        'description': 'Natural conversational pace (recommended)'
    },
    'fast': {
        'speed': 1.4,
        'temperature': 0.6,
        'description': 'Fast but still natural'
    },
    'expressive': {
        'speed': 1.0,
        'temperature': 0.9,
        'description': 'Slower but more expressive'
    },
    'neutral': {
        'speed': 1.0,
        'temperature': 0.5,
        'description': 'Standard neutral voice'
    }
}


if __name__ == "__main__":
    # Test
    tts = create_optimized_tts()
    
    print("\nAvailable presets:")
    for name, settings in TTS_PRESETS.items():
        print(f"  {name}: {settings['description']}")
        print(f"    speed={settings['speed']}, temperature={settings['temperature']}")
