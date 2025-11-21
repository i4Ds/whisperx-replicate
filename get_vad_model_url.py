from whisperx.vads.silero import Silero as Silero


vad = Silero(vad_onset=0.5, chunk_size=30.0)
import whisperx


import os
os.environ['HF_HOME'] = 'hf_home'
model = whisperx.load_model('i4ds/daily-brook-134', device='cuda')