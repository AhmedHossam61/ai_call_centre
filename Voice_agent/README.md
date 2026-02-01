# AI Call Center with Automatic Dialect Detection

Real-time Arabic dialect detection and response system using Gemini 2.5 Flash, Whisper, and XTTS.

## Features

✅ **Automatic Dialect Detection** - Detects Egyptian, Gulf, Levantine, Moroccan, and MSA  
✅ **LLM-Based Classification** - Uses Gemini 2.5 Flash (no training required)  
✅ **Session Dialect Locking** - Maintains consistent dialect throughout call  
✅ **Local Processing** - Whisper STT and XTTS TTS run on your machine  
✅ **Real-Time Conversation** - Microphone input → Speaker output  
✅ **Context-Aware Responses** - Remembers conversation history  

## Quick Start

### 1. Installation

```bash
# Clone or download the project files
cd ai-call-center

# Install dependencies
pip install -r requirements.txt

# This will install:
# - Gemini API client
# - Whisper (local STT)
# - XTTS (local TTS)
# - Audio processing libraries
```

**Note:** First run will download models (~2-3GB total):
- Whisper: ~150MB
- XTTS: ~2GB

### 2. Configuration

```bash
# Create .env file from template
cp .env.template .env

# Edit .env and add your Gemini API key
nano .env
```

Get your Gemini API key from: https://makersuite.google.com/app/apikey

### 3. Run

```bash
python main.py
```

## Usage

```
AI Call Center - Initializing...
============================================================
✓ Gemini 2.5 Flash initialized
✓ Whisper STT loaded
✓ TTS loaded on cuda
✓ Dialect detector & response generator ready

All systems ready!
============================================================

NEW CALL STARTED
Session ID: a1b2c3d4-5678-90ef-ghij-klmnopqrstuv
============================================================

Press Ctrl+C to end call
Press Enter to start each turn

[Press Enter when customer is ready to speak...]

--- Turn 1 ---
🎤 Listening for 5 seconds...
✓ Recording complete
🔄 Transcribing...
📝 Customer: ازيك يا فندم، عايز استفسر عن الخدمة
🔍 Detected: egyptian (confidence: 0.95)
   Reasoning: استخدام "ازيك" و"عايز" من اللهجة المصرية
✓ Dialect locked: egyptian (confidence: 0.95)
🔒 Dialect: egyptian (locked)
💬 Agent: أهلاً بيك! اتفضل قول لي، عايز تعرف ايه بالظبط؟
🔊 Generating speech...
▶️  Playing response...
✓ Turn 1 complete

[Press Enter when customer is ready to speak...]

--- Turn 2 ---
🎤 Listening for 5 seconds...
...
```

## Project Structure

```
ai-call-center/
├── main.py                    # Main application
├── session.py                 # Session management
├── dialect_detector.py        # Gemini-based dialect detection
├── response_generator.py      # Dialect-aware response generation
├── requirements.txt           # Python dependencies
├── .env                       # Configuration (create from .env.template)
├── .env.template             # Configuration template
└── README.md                 # This file
```

## System Requirements

### Minimum
- **CPU:** 4 cores
- **RAM:** 8GB
- **Storage:** 5GB free space
- **OS:** Windows, macOS, or Linux

### Recommended
- **GPU:** NVIDIA with 6GB+ VRAM (for faster TTS)
- **RAM:** 16GB
- **Storage:** 10GB free space

### Without GPU
The system works on CPU-only, just slower:
- Whisper: ~2-3 seconds (vs <1s on GPU)
- TTS: ~5-8 seconds (vs 2-3s on GPU)

## Supported Dialects

1. **Egyptian (مصري)**
   - Keywords: ازيك، عايز، ايه، دا، دي
   
2. **Gulf/Khaleeji (خليجي)**
   - Keywords: شلونك، شنو، ويش، تبغى
   
3. **Levantine/Shami (شامي)**
   - Keywords: كيفك، شو، هيك، بدك
   
4. **Moroccan/Maghrebi (مغربي)**
   - Keywords: كيفاش، واش، بزاف
   
5. **Modern Standard Arabic (فصحى)**
   - Keywords: كيف حالك، ماذا، هذا

## How It Works

### 1. Speech-to-Text (Whisper - Local)
Customer speaks → Audio recorded → Whisper transcribes to Arabic text

### 2. Dialect Detection (Gemini 2.5 Flash)
Transcribed text → Gemini analyzes linguistic features → Detects dialect

### 3. Session Locking
Once confidence ≥ 80% → Dialect locked for entire conversation

### 4. Response Generation (Gemini 2.5 Flash)
Customer query + Detected dialect → Gemini generates response in same dialect

### 5. Text-to-Speech (XTTS - Local)
Response text → XTTS synthesizes speech → Plays through speakers

## Configuration

### Adjust Recording Duration

In `main.py`, change:
```python
RECORDING_DURATION = 5  # Seconds per turn
```

### Change Whisper Model Size

Trade-off: Smaller = faster but less accurate

In `main.py`:
```python
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
```

**Recommendations:**
- `tiny`: Very fast, ~60% accuracy
- `base`: Fast, ~75% accuracy ✅ **Default**
- `small`: Moderate, ~85% accuracy
- `medium`: Slow, ~90% accuracy

### Adjust Dialect Lock Threshold

In `session.py`:
```python
self.lock_threshold = 0.8  # 80% confidence required
```

## Cost Analysis

**Per Call (5 minutes, ~10 turns):**
- Whisper STT: $0 (local)
- Gemini 2.5 Flash: ~$0.001
  - Dialect detection: 10 calls × $0.00005 = $0.0005
  - Response generation: 10 calls × $0.00005 = $0.0005
- TTS: $0 (local with XTTS)

**Total: ~$0.001 per call** (essentially free!)

**For 1000 calls/day:**
- Daily: $1
- Monthly: ~$30

## Troubleshooting

### Microphone Not Working

```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Set specific device in main.py
sd.default.device = 1  # Use device ID from list
```

### No Audio Output

```bash
# Test speakers
python -c "import sounddevice as sd; import numpy as np; sd.play(np.random.randn(16000), 16000); sd.wait()"
```

### Slow Performance

1. **Use smaller Whisper model:** Change to `tiny` or `base`
2. **Reduce recording duration:** Set to 3 seconds
3. **Use GPU:** Enable CUDA if available

### Dialect Detection Issues

**Problem:** Wrong dialect detected  
**Solution:** Text might be too short or ambiguous. System will improve accuracy over multiple turns.

**Problem:** Confidence too low  
**Solution:** Ask customer to speak more (longer utterances help)

## Advanced Features

### Add Custom Business Context

In `response_generator.py`, modify the system context:

```python
system_context = """
أنت مساعد في شركة [اسم الشركة].
نحن نقدم خدمات: [قائمة الخدمات]
أوقات العمل: [الأوقات]
"""

response = generator.generate(
    user_query=query,
    dialect=dialect,
    system_context=system_context
)
```

### Multiple Concurrent Calls

For production with multiple agents:

```python
import threading

def handle_multiple_calls(num_calls=5):
    threads = []
    for i in range(num_calls):
        agent = CallCenterAgent()
        thread = threading.Thread(target=agent.handle_call)
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
```

### Call Recording & Analytics

Add to `session.py`:

```python
def save_call_recording(self, filepath):
    """Save conversation to file"""
    import json
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'session_id': self.session_id,
            'dialect': self.detected_dialect,
            'conversation': self.conversation_history
        }, f, ensure_ascii=False, indent=2)
```

## API Integration

### Flask REST API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
agent = CallCenterAgent()

@app.route('/detect_dialect', methods=['POST'])
def detect_dialect():
    text = request.json['text']
    dialect, confidence = agent.dialect_detector.detect(text)
    return jsonify({
        'dialect': dialect,
        'confidence': confidence
    })

@app.route('/generate_response', methods=['POST'])
def generate_response():
    query = request.json['query']
    dialect = request.json['dialect']
    response = agent.response_generator.generate(query, dialect)
    return jsonify({'response': response})
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### Environment Variables

For production, use environment variables instead of .env:

```bash
export GEMINI_API_KEY="your_key"
export RECORDING_DURATION="3"
export WHISPER_MODEL="base"
```

## Support & Contributing

For issues or questions, please check:
1. This README
2. Code comments in source files
3. The comprehensive guide: `AI_CALL_CENTER_GUIDE.md`

## License

This project is for educational and commercial use.

---

**Ready to start?** Just run `python main.py` and begin testing!
