"""
CTC-based forced alignment for word-level timestamps.

Uses wav2vec2 models and a CTC alignment algorithm adapted from WhisperX.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from faster_whisper.audio import decode_audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi-vlsp2020",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
}


@dataclass
class AlignedWord:
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class AlignedSegment:
    text: str
    start: float
    end: float
    words: List[AlignedWord]
    speaker: Optional[str] = None
    avg_logprob: Optional[float] = None

    def to_dict(self) -> dict:
        data = {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [word.to_dict() for word in self.words],
        }
        if self.speaker is not None:
            data["speaker"] = self.speaker
        if self.avg_logprob is not None:
            data["avg_logprob"] = self.avg_logprob
        return data


@dataclass
class AlignmentResult:
    segments: List[AlignedSegment]
    word_segments: List[AlignedWord]

    def to_dict(self) -> dict:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "word_segments": [word.to_dict() for word in self.word_segments],
        }


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class CharSegment:
    label: str
    start: int
    end: int
    score: float


class CTCAligner:
    """CTC-based forced alignment using wav2vec2 models."""

    def __init__(
        self,
        language: str = "de",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_name: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        self.language = language
        self.device = device
        self.model_name = model_name
        self.model_dir = model_dir
        self._model = None
        self._metadata = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        model_name = self.model_name
        if model_name is None:
            if self.language in DEFAULT_ALIGN_MODELS_TORCH:
                model_name = DEFAULT_ALIGN_MODELS_TORCH[self.language]
            elif self.language in DEFAULT_ALIGN_MODELS_HF:
                model_name = DEFAULT_ALIGN_MODELS_HF[self.language]
            else:
                raise ValueError(f"No default alignment model for language: {self.language}")

        logger.info("Loading alignment model: %s", model_name)

        if model_name in torchaudio.pipelines.__all__:
            bundle = torchaudio.pipelines.__dict__[model_name]
            self._model = bundle.get_model(dl_kwargs={"model_dir": self.model_dir}).to(self.device)
            labels = bundle.get_labels()
            align_dictionary = {char.lower(): index for index, char in enumerate(labels)}
            pipeline_type = "torchaudio"
        else:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=self.model_dir)
            self._model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=self.model_dir).to(self.device)
            align_dictionary = {
                char.lower(): code for char, code in processor.tokenizer.get_vocab().items()
            }
            pipeline_type = "huggingface"

        self._metadata = {
            "language": self.language,
            "dictionary": align_dictionary,
            "type": pipeline_type,
        }
        logger.info("Alignment model loaded: %s", pipeline_type)

    def align(
        self,
        segments: List[dict],
        audio: Union[str, Path, np.ndarray, torch.Tensor],
    ) -> AlignmentResult:
        self._load_model()

        if isinstance(audio, (str, Path)):
            audio = decode_audio(str(audio), sampling_rate=SAMPLE_RATE)
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)

        max_duration = audio.shape[1] / SAMPLE_RATE
        model_dictionary = self._metadata["dictionary"]
        model_lang = self._metadata["language"]
        model_type = self._metadata["type"]

        aligned_segments: List[AlignedSegment] = []
        all_words: List[AlignedWord] = []

        for segment in segments:
            t1, t2 = segment["start"], segment["end"]
            text = segment["text"]
            speaker = segment.get("speaker")
            avg_logprob = segment.get("avg_logprob")

            if t1 >= max_duration:
                aligned_segments.append(
                    AlignedSegment(
                        text=text,
                        start=t1,
                        end=t2,
                        words=[],
                        speaker=speaker,
                        avg_logprob=avg_logprob,
                    )
                )
                continue

            num_leading = len(text) - len(text.lstrip())
            num_trailing = len(text) - len(text.rstrip())

            clean_char, clean_cdx = [], []
            for cdx, char in enumerate(text):
                char_ = (
                    char.lower().replace(" ", "|")
                    if model_lang not in LANGUAGES_WITHOUT_SPACES
                    else char.lower()
                )
                if cdx < num_leading or cdx > len(text) - num_trailing - 1:
                    continue
                clean_char.append(char_ if char_ in model_dictionary else "*")
                clean_cdx.append(cdx)

            if not clean_char:
                aligned_segments.append(
                    AlignedSegment(
                        text=text,
                        start=t1,
                        end=t2,
                        words=[],
                        speaker=speaker,
                        avg_logprob=avg_logprob,
                    )
                )
                continue

            text_clean = "".join(clean_char)
            tokens = [model_dictionary.get(char, -1) for char in text_clean]

            f1, f2 = int(t1 * SAMPLE_RATE), int(t2 * SAMPLE_RATE)
            waveform_segment = audio[:, f1:f2]

            if waveform_segment.shape[-1] < 400:
                lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(self.device)
                waveform_segment = torch.nn.functional.pad(
                    waveform_segment, (0, 400 - waveform_segment.shape[-1])
                )
            else:
                lengths = None

            with torch.inference_mode():
                if model_type == "torchaudio":
                    emissions, _ = self._model(waveform_segment.to(self.device), lengths=lengths)
                else:
                    emissions = self._model(waveform_segment.to(self.device)).logits
                emissions = torch.log_softmax(emissions, dim=-1)

            emission = emissions[0].cpu().detach()
            blank_id = next(
                (code for char, code in model_dictionary.items() if char in ["[pad]", "<pad>"]),
                0,
            )

            trellis = self._get_trellis(emission, tokens, blank_id)
            path = self._backtrack(trellis, emission, tokens, blank_id)

            if path is None:
                aligned_segments.append(
                    AlignedSegment(
                        text=text,
                        start=t1,
                        end=t2,
                        words=[],
                        speaker=speaker,
                        avg_logprob=avg_logprob,
                    )
                )
                continue

            char_segments = self._merge_repeats(path, text_clean)
            ratio = (t2 - t1) / (trellis.size(0) - 1)

            char_segments_arr = []
            word_idx = 0
            for cdx, char in enumerate(text):
                start, end, score = None, None, None
                if cdx in clean_cdx:
                    char_seg = char_segments[clean_cdx.index(cdx)]
                    start = round(char_seg.start * ratio + t1, 3)
                    end = round(char_seg.end * ratio + t1, 3)
                    score = round(char_seg.score, 3)
                char_segments_arr.append(
                    {
                        "char": char,
                        "start": start,
                        "end": end,
                        "score": score,
                        "word_idx": word_idx,
                    }
                )
                if model_lang in LANGUAGES_WITHOUT_SPACES or cdx == len(text) - 1 or text[cdx + 1] == " ":
                    word_idx += 1

            char_df = pd.DataFrame(char_segments_arr)
            words = []
            for _, word_group in char_df.groupby("word_idx"):
                word_text = "".join(word_group["char"]).strip()
                if not word_text:
                    continue

                start_values = word_group["start"].dropna()
                end_values = word_group["end"].dropna()
                score_values = word_group["score"].dropna()

                word = AlignedWord(
                    word=word_text,
                    start=float(start_values.min()) if not start_values.empty else None,
                    end=float(end_values.max()) if not end_values.empty else None,
                    score=float(score_values.mean()) if not score_values.empty else None,
                )
                words.append(word)
                all_words.append(word)

            aligned_segments.append(
                AlignedSegment(
                    text=text,
                    start=t1,
                    end=t2,
                    words=words,
                    speaker=speaker,
                    avg_logprob=avg_logprob,
                )
            )

        return AlignmentResult(segments=aligned_segments, word_segments=all_words)

    def _get_trellis(self, emission: torch.Tensor, tokens: List[int], blank_id: int) -> torch.Tensor:
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.full((num_frame + 1, num_tokens + 1), -float("inf"))
        trellis[:, 0] = 0
        for frame_index in range(num_frame):
            trellis[frame_index + 1, 1:] = torch.maximum(
                trellis[frame_index, 1:] + emission[frame_index, blank_id],
                trellis[frame_index, :-1] + emission[frame_index, tokens],
            )
        return trellis

    def _backtrack(
        self,
        trellis: torch.Tensor,
        emission: torch.Tensor,
        tokens: List[int],
        blank_id: int,
    ) -> Optional[List[Point]]:
        time_index = trellis.size(0) - 1
        token_index = trellis.size(1) - 1
        path = []

        while token_index > 0:
            if time_index <= 0:
                return None

            stayed = trellis[time_index - 1, token_index] + emission[time_index - 1, blank_id]
            changed = trellis[time_index - 1, token_index - 1] + emission[time_index - 1, tokens[token_index - 1]]

            time_index -= 1
            if changed > stayed:
                token_index -= 1
                path.append(
                    Point(
                        token_index=token_index,
                        time_index=time_index,
                        score=emission[time_index, tokens[token_index]].exp().item(),
                    )
                )
            else:
                path.append(
                    Point(
                        token_index=token_index - 1,
                        time_index=time_index,
                        score=emission[time_index, blank_id].exp().item(),
                    )
                )

        return list(reversed(path))

    def _merge_repeats(self, path: List[Point], transcript: str) -> List[CharSegment]:
        i1, segments = 0, []
        while i1 < len(path):
            i2 = i1
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1

            score = sum(point.score for point in path[i1:i2]) / (i2 - i1)
            segments.append(
                CharSegment(
                    label=transcript[path[i1].token_index],
                    start=path[i1].time_index,
                    end=path[i2 - 1].time_index + 1,
                    score=score,
                )
            )
            i1 = i2
        return segments


def save_alignment_log(
    result: AlignmentResult, output_path: Union[str, Path], audio_path: Optional[str] = None
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_data = result.to_dict()
    if audio_path:
        log_data["audio_file"] = str(audio_path)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(log_data, handle, indent=2, ensure_ascii=False)

    logger.info("Saved alignment log to %s", output_path)
