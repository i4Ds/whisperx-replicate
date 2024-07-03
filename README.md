# Whisper-SG whisperX on Replicate


# Model Information
https://huggingface.co/i4ds/whisper4sg-srg-v2-full-mc-de-sg-corpus-v2

# How to use
Install COG and Setup Replicate.

Use Locally:
- Run `download_model.sh`
- Predict with `cog predict -i audio_file=@<PATH TO FILE>`

Deploy to Replicate:
- Run `download_model.sh` 
- cog push `YOUR REPOSITORY ON REPLICATE` (best to read the guide on replicate)

# Citation

```
@misc{bain2023whisperx,
      title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio}, 
      author={Max Bain and Jaesung Huh and Tengda Han and Andrew Zisserman},
      year={2023},
      eprint={2303.00747},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```