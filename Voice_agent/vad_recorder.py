"""
Voice Activity Detection (VAD) Recorder
Automatically stops recording when user stops speaking
"""

import sounddevice as sd
import numpy as np
import queue
import threading

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("⚠️  webrtcvad not installed. Install with: pip install webrtcvad")


class VADRecorder:
    """
    Records audio with Voice Activity Detection
    Stops automatically when user stops speaking
    """
    
    def __init__(self, sample_rate=16000, aggressiveness=2):
        """
        Initialize VAD Recorder
        
        Args:
            sample_rate: Audio sample rate (16000 Hz required for VAD)
            aggressiveness: VAD sensitivity (0-3, higher = more aggressive)
                           0 = least aggressive (more forgiving)
                           3 = most aggressive (strict)
                           Recommended: 2 for normal environments
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30  # WebRTC VAD requires 10, 20, or 30 ms frames
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(aggressiveness)
            self.vad_enabled = True
        else:
            self.vad_enabled = False
            print("⚠️  VAD disabled - will use timeout instead")
    
    def record_with_vad(self, 
                        max_duration=30,
                        silence_duration=1.5,
                        pre_speech_buffer=0.3):
        """
        Record audio and stop when silence detected
        
        Args:
            max_duration: Maximum recording time (seconds)
            silence_duration: How long silence before stopping (seconds)
            pre_speech_buffer: Keep this much audio before speech starts (seconds)
        
        Returns:
            numpy array of audio
        """
        if not self.vad_enabled:
            return self._record_with_timeout(max_duration)
        
        print("🎤 Listening... (speak now, will auto-stop when you finish)")
        
        # Buffers
        audio_queue = queue.Queue()
        pre_speech_frames = []
        speech_frames = []
        
        # Silence detection
        num_silent_frames = 0
        silent_frames_threshold = int(silence_duration * 1000 / self.frame_duration_ms)
        pre_buffer_frames = int(pre_speech_buffer * 1000 / self.frame_duration_ms)
        
        # State
        speech_started = False
        recording = True
        
        def audio_callback(indata, frames, time, status):
            """Called for each audio chunk"""
            if status:
                print(f"Audio callback status: {status}")
            if recording:
                audio_queue.put(indata.copy())
        
        # Start recording stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.frame_size,
            callback=audio_callback
        )
        
        with stream:
            max_frames = int(max_duration * 1000 / self.frame_duration_ms)
            
            for i in range(max_frames):
                try:
                    # Get frame from queue
                    frame = audio_queue.get(timeout=1.0)
                    frame_bytes = frame.tobytes()
                    
                    # Check if frame contains speech
                    is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                    
                    if not speech_started:
                        # Before speech starts, keep in pre-buffer
                        pre_speech_frames.append(frame)
                        if len(pre_speech_frames) > pre_buffer_frames:
                            pre_speech_frames.pop(0)
                        
                        if is_speech:
                            # Speech started! Add pre-buffer to main recording
                            speech_started = True
                            speech_frames.extend(pre_speech_frames)
                            speech_frames.append(frame)
                            print("✓ Speech detected")
                    else:
                        # After speech started
                        speech_frames.append(frame)
                        
                        if is_speech:
                            num_silent_frames = 0
                        else:
                            num_silent_frames += 1
                        
                        # Stop if silence detected for threshold duration
                        if num_silent_frames > silent_frames_threshold:
                            if len(speech_frames) > 10:  # Ensure we have some audio
                                print("✓ Silence detected, stopping recording")
                                recording = False
                                break
                
                except queue.Empty:
                    print("⚠️  Audio queue timeout")
                    break
        
        # Combine all frames
        if speech_frames:
            audio = np.concatenate(speech_frames, axis=0).flatten()
        elif pre_speech_frames:
            # Fallback: use pre-buffer if no speech detected
            audio = np.concatenate(pre_speech_frames, axis=0).flatten()
        else:
            # No audio at all
            audio = np.zeros(self.sample_rate, dtype=np.int16)
        
        # Convert from int16 to float32
        audio = audio.astype(np.float32) / 32768.0
        
        duration = len(audio) / self.sample_rate
        print(f"✓ Recorded {duration:.1f} seconds")
        
        return audio
    
    def _record_with_timeout(self, duration):
        """Fallback: simple timeout-based recording"""
        print(f"🎤 Listening for {duration} seconds...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        return audio.flatten()


class SimpleRecorder:
    """
    Simple recorder with adjustable timeout
    No VAD - just records for specified duration
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def record(self, duration=10):
        """
        Record for fixed duration
        
        Args:
            duration: Recording time in seconds
        
        Returns:
            numpy array of audio
        """
        print(f"🎤 Listening for up to {duration} seconds...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("✓ Recording complete")
        return audio.flatten()


# Convenience functions
def create_vad_recorder(aggressiveness=2):
    """
    Create VAD recorder instance
    
    Args:
        aggressiveness: 0-3 (0=lenient, 3=strict)
    
    Returns:
        VADRecorder instance
    """
    return VADRecorder(aggressiveness=aggressiveness)


def create_simple_recorder():
    """
    Create simple timeout-based recorder
    
    Returns:
        SimpleRecorder instance
    """
    return SimpleRecorder()
