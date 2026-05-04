"""
Voice Activity Detection providers.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from local_cache import DEFAULT_PYANNOTE_VAD_MODEL, resolve_hf_hub_snapshot

from .base import SpeechSegment, VADProvider
from .utils import get_device, load_audio_for_pyannote

logger = logging.getLogger(__name__)


class PyAnnoteVAD(VADProvider):
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import VoiceActivityDetection

            logger.info("Loading PyAnnote VAD model...")
            model_path = resolve_hf_hub_snapshot(DEFAULT_PYANNOTE_VAD_MODEL, "pytorch_model.bin")
            model = Model.from_pretrained(model_path)
            self._pipeline = VoiceActivityDetection(segmentation=model)
            self._pipeline.to(torch.device(self.device))

            min_duration_on = self.params.get("min_duration_on")
            min_duration_off = self.params.get("min_duration_off")

            unsupported = []
            for key in ("threshold", "onset", "offset", "segmentation"):
                if key in self.params:
                    unsupported.append(key)
            if unsupported:
                logger.warning(
                    "PyAnnote VAD does not support %s in this environment. Ignoring these parameters.",
                    ", ".join(unsupported),
                )

            hyper_parameters = {
                "min_duration_on": min_duration_on if min_duration_on is not None else 0.0,
                "min_duration_off": min_duration_off if min_duration_off is not None else 0.0,
            }
            logger.info("PyAnnote VAD hyper-parameters: %s", hyper_parameters)
            self._pipeline.instantiate(hyper_parameters)
            logger.info("PyAnnote VAD model loaded successfully")
        return self._pipeline

    def detect_speech(
        self,
        audio_path: Union[str, Path],
        min_duration: float = 0.5,
        merge_threshold: float = 0.3,
    ) -> List[SpeechSegment]:
        pipeline = self._load_pipeline()

        logger.info("Running VAD on %s", audio_path)
        vad_result = pipeline(load_audio_for_pyannote(audio_path))

        segments = []
        for item in vad_result.itertracks():
            segment = item[0] if isinstance(item, tuple) else item
            segments.append(SpeechSegment(start=segment.start, end=segment.end))

        segments = self._post_process_segments(segments, min_duration, merge_threshold)
        logger.info("Detected %s speech segments (PyAnnote)", len(segments))
        return segments


class SileroVAD(VADProvider):
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)

    def detect_speech(
        self,
        audio_path: Union[str, Path],
        min_duration: float = 0.5,
        merge_threshold: float = 0.3,
    ) -> List[SpeechSegment]:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        audio = decode_audio(str(audio_path), sampling_rate=16000)

        if isinstance(self.params, VadOptions):
            vad_options = self.params
        elif self.params:
            vad_params = {
                "threshold": self.params.get("threshold", self.params.get("onset", 0.5)),
                "neg_threshold": self.params.get("neg_threshold"),
                "min_speech_duration_ms": self.params.get("min_speech_duration_ms", 250),
                "max_speech_duration_s": self.params.get("max_speech_duration_s", float("inf")),
                "min_silence_duration_ms": self.params.get("min_silence_duration_ms", 2000),
                "speech_pad_ms": self.params.get("speech_pad_ms", 400),
            }
            logger.info("Silero VAD options: %s", vad_params)
            vad_options = VadOptions(**vad_params)
        else:
            vad_options = VadOptions(threshold=0.5, neg_threshold=0.365)

        speech_timestamps = get_speech_timestamps(audio, vad_options)
        segments = [
            SpeechSegment(start=float(ts["start"]) / 16000.0, end=float(ts["end"]) / 16000.0)
            for ts in speech_timestamps
        ]

        segments = self._post_process_segments(segments, min_duration, merge_threshold)
        logger.info("Detected %s speech segments (Silero)", len(segments))
        return segments


class SpeechBrainVAD(VADProvider):
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)

    def detect_speech(
        self,
        audio_path: Union[str, Path],
        min_duration: float = 0.5,
        merge_threshold: float = 0.3,
    ) -> List[SpeechSegment]:
        try:
            from speechbrain.inference.VAD import VAD
        except Exception as exc:
            raise RuntimeError("SpeechBrain VAD requires speechbrain and torchaudio") from exc

        audio_path = Path(audio_path)
        run_device = get_device(self.device)
        run_opts = {"device": run_device} if run_device else None

        vad_model = self.params.get("vad_model", "speechbrain/vad-crdnn-libriparty")
        vad = VAD.from_hparams(
            source=vad_model,
            savedir=self.params.get("vad_savedir", "pretrained_sb_vad"),
            run_opts=run_opts,
        )
        speech_segs = vad.get_speech_segments(
            str(audio_path),
            activation_th=self.params.get("vad_threshold", 0.5),
            deactivation_th=self.params.get("vad_deactivation_th", 0.25),
            len_th=self.params.get("min_speech_duration", 0.2),
            close_th=self.params.get("min_silence_duration", 0.2),
        )
        if len(speech_segs) == 0:
            logger.warning("No speech detected by SpeechBrain VAD")
            return []

        segments = [SpeechSegment(start=float(start), end=float(end)) for (start, end) in speech_segs]
        segments = self._post_process_segments(segments, min_duration, merge_threshold)
        logger.info("Detected %s speech segments (SpeechBrain)", len(segments))
        return segments


class NemoVAD(VADProvider):
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)

    def detect_speech(
        self,
        audio_path: Union[str, Path],
        min_duration: float = 0.5,
        merge_threshold: float = 0.3,
    ) -> List[SpeechSegment]:
        try:
            import tempfile

            from omegaconf import OmegaConf
            from nemo.collections.asr.models import ClusteringDiarizer
        except Exception as exc:
            raise RuntimeError("NeMo VAD requires nemo_toolkit[asr] and omegaconf") from exc

        audio_path = Path(audio_path)
        workdir = Path(self.params.get("workdir", tempfile.mkdtemp(prefix="nemo_vad_")))
        workdir.mkdir(parents=True, exist_ok=True)

        manifest_path = workdir / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            handle.write(
                (
                    '{"audio_filepath": "%s", "offset": 0, "duration": null, '
                    '"label": "infer", "text": "-", "num_speakers": null, '
                    '"rttm_filepath": null, "uem_filepath": null}\n'
                )
                % str(audio_path.resolve())
            )

        cfg = OmegaConf.create(
            {
                "sample_rate": 16000,
                "batch_size": 1,
                "num_workers": 0,
                "verbose": False,
                "diarizer": {
                    "manifest_filepath": str(manifest_path),
                    "out_dir": str(workdir / "pred_rttms"),
                    "oracle_vad": False,
                    "collar": 0.25,
                    "ignore_overlap": True,
                    "vad": {
                        "model_path": self.params.get("vad_model", "vad_marblenet"),
                        "parameters": self.params.get(
                            "vad_params",
                            {
                                "onset": 0.8,
                                "offset": 0.6,
                                "pad_offset": -0.05,
                                "window_length_in_sec": 0.15,
                                "shift_length_in_sec": 0.01,
                                "smoothing": False,
                                "overlap": 0.875,
                            },
                        ),
                    },
                    "speaker_embeddings": {
                        "model_path": self.params.get("speaker_model", "titanet_large"),
                        "parameters": self.params.get(
                            "speaker_params",
                            {
                                "save_embeddings": False,
                                "window_length_in_sec": [1.5, 1.0, 0.5, 0.25],
                                "shift_length_in_sec": [0.75, 0.5, 0.25, 0.125],
                                "multiscale_weights": [1.0, 1.0, 1.0, 1.2],
                            },
                        ),
                    },
                    "clustering": {"parameters": {"oracle_num_speakers": False}},
                },
                "device": self.device,
            }
        )

        diarizer = ClusteringDiarizer(cfg=cfg)
        diarizer.diarize()

        pred_dir = Path(cfg.diarizer.out_dir)
        rttm_files = list(pred_dir.rglob("*.rttm"))
        if not rttm_files:
            return []

        from .utils import parse_rttm

        segments = parse_rttm(rttm_files[0])
        segments = self._post_process_segments(segments, min_duration, merge_threshold)
        logger.info("Detected %s speech segments (NeMo)", len(segments))
        return segments


class VADFactory:
    PROVIDERS = {
        "pyannote": PyAnnoteVAD,
        "silero": SileroVAD,
        "speechbrain": SpeechBrainVAD,
        "nemo": NemoVAD,
    }

    @classmethod
    def create(
        cls,
        method: str,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ) -> VADProvider:
        if method not in cls.PROVIDERS:
            supported = ", ".join(sorted(cls.PROVIDERS))
            raise ValueError(f"Unknown VAD method '{method}'. Supported methods: {supported}")
        return cls.PROVIDERS[method](
            device=device,
            params=params,
            use_auth_token=use_auth_token,
        )
