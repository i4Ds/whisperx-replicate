from cog import BasePredictor, Input, Path, BaseModel
import pysubs2

import gc
import whisperx
import time

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"
whisper_arch = "./models/faster-whisper-large-v3-turbo"

class Output(BaseModel):
    segments: str
    detected_language: str


class Predictor(BasePredictor):
    def predict(
            self,
            audio_file: Path = Input(description="Audio file"),
            language: str = Input(
                description="Language is fixed to German (ISO 'de')",
                default="de",
                choices=["de"]),
            batch_size: int = Input(
                description="Parallelization of input audio transcription",
                default=8),
            temperature: float = Input(
                description="Temperature to use for sampling",
                default=0),
            vad_onset: float = Input(
                description="VAD onset",
                default=0.500),
            vad_offset: float = Input(
                description="VAD offset",
                default=0.363),
            align_output: bool = Input(
                description="Aligns whisper output to get accurate word-level timestamps",
                default=False),
            debug: bool = Input(
                description="Print out compute/inference times and memory usage information",
                default=True)
    ) -> Output:
            language = "de"  # hardcode to German regardless of user input

            asr_options = {
                "temperatures": [temperature],
            }

            vad_options = {
                "vad_onset": vad_onset,
                "vad_offset": vad_offset
            }

            start_time = time.time_ns() / 1e6

            model = whisperx.load_model(whisper_arch, device, compute_type=compute_type, language=language,
                                        asr_options=asr_options, vad_options=vad_options)

            model.model.feat_kwargs["feature_size"] = 128
            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            audio = whisperx.load_audio(audio_file)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load audio: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} ms")

            gc.collect()
            del model

            if align_output:
                model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            # Prettify to SRT format
            output = pysubs2.load_from_whisper(result)
            return Output(
                segments=output.to_string(format_="srt"),
                detected_language=detected_language
            )

