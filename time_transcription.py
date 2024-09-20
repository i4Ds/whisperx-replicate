import matplotlib.pyplot as plt
from pydub import AudioSegment
import os
import tempfile
from use_replicate_model import transcribe_audio
import sys
import statistics

def concatenate_audio(file_path, num_repeats):
    audio = AudioSegment.from_file(file_path)
    return audio * num_repeats

def warm_up_model(file_path):
    print("Warming up the model...")
    for _ in range(3):
        with open(file_path, "rb") as file:
            transcribe_audio(file)
    print("Warm-up complete.")

def main(file_path, max_repeats=10, reps_per_length=3):
    """
    Main function to benchmark transcription times for concatenated audio files of increasing length.

    Args:
    - file_path (str): Path to the input audio file to be concatenated and transcribed.
    - max_repeats (int, optional): The maximum number of times to repeat the base audio for benchmarking. Defaults to 10.
    - reps_per_length (int, optional): The number of transcription repetitions to perform for each audio length to average the results. Defaults to 3.

    Logic:
    1. Warm up the model by transcribing the base audio file 3 times to simulate real-world performance.
    2. Concatenate the base audio file to increase its length, from 1x up to max_repeats times the original length.
    3. For each concatenated length, export the audio as a temporary file and transcribe it using the `transcribe_audio` function.
    4. Perform the transcription `reps_per_length` times for each concatenated audio length to calculate the average transcription time.
    5. Store the audio lengths and their corresponding average transcription times.
    6. Plot the relationship between the audio length and average transcription time.
    7. Save and display the plot, showing how transcription time scales with increasing audio length.

    Example Usage:
    python script.py <audio_file_path> [max_repeats] [reps_per_length]

    Outputs:
    - A plot showing the average transcription time vs. audio length.
    - Prints lists of audio lengths and corresponding average transcription times to the console.
    """
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
            duration_list = []
            for j in range(reps_per_length):
                print(f"  Repetition {j+1}/{reps_per_length}")
                with open(temp_file, "rb") as file:
                    output = transcribe_audio(file)
                duration = output['transcribe_ms'] / 1000  # Convert to seconds
                duration_list.append(duration)
                print(duration['transcription'])

            avg_duration = statistics.mean(duration_list)
            lengths.append(base_length * i)
            durations.append(avg_duration)

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, durations, marker='o')
    plt.xlabel("Audio Length (seconds)")
    plt.ylabel("Average Transcription Time (seconds)")
    plt.title(f"Average Transcription Time vs Audio Length ({reps_per_length} reps per length)")
    plt.grid(True)
    plt.savefig("transcription_time_plot.png")
    plt.show()

    print("Lengths (seconds):", lengths)
    print("Average Durations (seconds):", durations)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <audio_file_path> [max_repeats] [reps_per_length]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    max_repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    reps_per_length = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    main(file_path, max_repeats, reps_per_length)
