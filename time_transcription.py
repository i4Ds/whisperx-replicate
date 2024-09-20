import matplotlib.pyplot as plt
from pydub import AudioSegment
import os
import tempfile
from use_replicate_model import transcribe_audio
import sys

def concatenate_audio(file_path, num_repeats):
    audio = AudioSegment.from_file(file_path)
    return audio * num_repeats

def warm_up_model(file_path):
    print("Warming up the model...")
    for _ in range(3):
        with open(file_path, "rb") as file:
            transcribe_audio(file)
    print("Warm-up complete.")

def main(file_path, max_repeats=10):
    warm_up_model(file_path)

    base_audio = AudioSegment.from_file(file_path)
    base_length = len(base_audio) / 1000  # Length in seconds

    lengths = []
    durations = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(1, max_repeats + 1):  # Test with 1 to max_repeats repetitions
            temp_file = os.path.join(temp_dir, f"temp_{i}.flac")
            concatenated_audio = concatenate_audio(file_path, i)
            concatenated_audio.export(temp_file, format="flac")

            print(f"Transcribing {i}x length...")
            with open(temp_file, "rb") as file:
                output = transcribe_audio(file)
            
            duration = output['transcribe_ms'] / 1000  # Convert to seconds

            lengths.append(base_length * i)
            durations.append(duration)

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, durations, marker='o')
    plt.xlabel("Audio Length (seconds)")
    plt.ylabel("Transcription Time (seconds)")
    plt.title("Transcription Time vs Audio Length")
    plt.grid(True)
    plt.savefig("transcription_time_plot.png")
    plt.show()

    print("Lengths (seconds):", lengths)
    print("Durations (seconds):", durations)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <audio_file_path> [max_repeats]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    max_repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    main(file_path, max_repeats)
