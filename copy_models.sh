#!/bin/bash


# make folder
mkdir -p models

# Copy the model to the models directory
# Basically, copy the model from where it's cached to a fixed folder in this repository.
cp -rL "/home/kenfus/.cache/huggingface/hub/models--i4ds--daily-brook-134/snapshots/65904462d6e5fffa8e33539109565f72e9f29d7f/" models/faster-whisper-large-v3-turbo

