#!/bin/bash
# MODEL PATH
MODEL_PATH=i4ds/whisper4sg-srg-v2-full-mc-de-sg-corpus-v2

# To download it to the cache, because there is no internet in the container.
python3 -c "import os; os.environ['HF_HOME'] = 'hf_home'; from faster_whisper import WhisperModel; WhisperModel('$MODEL_PATH')"
