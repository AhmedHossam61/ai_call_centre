@echo off
REM Fix PyTorch Version for TTS Compatibility
REM Fixes TorchCodec/FFmpeg error

echo ==============================================
echo PyTorch Version Fix for TTS
echo RTX 3050 Compatible Setup
echo ==============================================

echo.
echo Current issue: PyTorch 2.9.1 + CUDA 13.0 incompatible with TTS
echo Solution: Install PyTorch 2.5.1 + CUDA 11.8
echo.

pause

REM Step 1: Uninstall incompatible versions
echo.
echo Step 1: Removing incompatible packages...
pip uninstall torch torchvision torchaudio TTS -y

REM Step 2: Install compatible PyTorch
echo.
echo Step 2: Installing PyTorch 2.5.1 with CUDA 11.8...
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

REM Step 3: Verify GPU
echo.
echo Step 3: Verifying GPU detection...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

REM Step 4: Install TTS
echo.
echo Step 4: Installing TTS library...
pip install TTS==0.22.0

REM Step 5: Install other dependencies
echo.
echo Step 5: Installing additional dependencies...
pip install google-generativeai openai-whisper sounddevice soundfile numpy noisereduce librosa webrtcvad python-dotenv

REM Final test
echo.
echo ==============================================
echo Testing TTS on GPU...
echo ==============================================

python -c "import torch; from TTS.api import TTS; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to('cuda' if torch.cuda.is_available() else 'cpu'); print('TTS loaded on:', 'cuda' if torch.cuda.is_available() else 'cpu')"

echo.
if %ERRORLEVEL% EQU 0 (
    echo ==============================================
    echo ✓ INSTALLATION SUCCESSFUL!
    echo ==============================================
    echo.
    echo Your RTX 3050 is ready for fast TTS!
    echo.
    echo Run your app with:
    echo   python main.py
    echo.
    echo You should see:
    echo   "✓ Optimized TTS loaded on cuda"
    echo.
) else (
    echo ==============================================
    echo ❌ INSTALLATION FAILED
    echo ==============================================
    echo.
    echo Try manual installation:
    echo 1. pip uninstall torch torchvision torchaudio -y
    echo 2. pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo 3. pip install TTS
    echo.
    echo See FIX_PYTORCH_VERSION.md for details
    echo.
)

pause
