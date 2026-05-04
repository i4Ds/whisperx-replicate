"""
Speaker diarization providers.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from local_cache import DEFAULT_PYANNOTE_DIARIZATION_MODEL, resolve_hf_hub_snapshot

from .base import DiarizationProvider, DiarizationResult, SpeechSegment
from .utils import get_device, load_audio_for_pyannote, parse_rttm

logger = logging.getLogger(__name__)


class PyAnnoteDiarization(DiarizationProvider):
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
            from pyannote.audio import Pipeline

            pipeline_name = self.params.get("pipeline_name", DEFAULT_PYANNOTE_DIARIZATION_MODEL)
            model_path = resolve_hf_hub_snapshot(pipeline_name, "config.yaml")

            logger.info("Loading PyAnnote diarization pipeline from %s", model_path)
            self._pipeline = Pipeline.from_pretrained(model_path)
            self._pipeline.to(torch.device(self.device))
            pipeline_params = self.params.get("pipeline_params")
            if pipeline_params:
                logger.info("Instantiating PyAnnote diarization pipeline with parameters")
                self._pipeline.instantiate(pipeline_params)
            logger.info("PyAnnote diarization pipeline loaded successfully")
        return self._pipeline

    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        pipeline = self._load_pipeline()

        logger.info("Running diarization on %s", audio_path)
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        diarization_output = pipeline(load_audio_for_pyannote(audio_path), **kwargs)
        diarization = diarization_output.speaker_diarization

        segments = []
        speakers_set = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeechSegment(start=turn.start, end=turn.end, speaker=speaker))
            speakers_set.add(speaker)

        segments.sort(key=lambda segment: segment.start)
        logger.info(
            "Diarization complete: %s segments, %s speakers",
            len(segments),
            len(speakers_set),
        )
        return DiarizationResult(segments=segments, num_speakers=len(speakers_set))


class NemoClusteringDiarization(DiarizationProvider):
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)

    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        try:
            from nemo.collections.asr.models import ClusteringDiarizer
            from omegaconf import OmegaConf
        except Exception as exc:
            raise RuntimeError("NeMo diarization requires nemo_toolkit[asr] and omegaconf") from exc

        audio_path = Path(audio_path)
        workdir = Path(self.params.get("workdir", audio_path.parent / "nemo_out"))
        workdir.mkdir(parents=True, exist_ok=True)

        manifest_path = workdir / "manifest.jsonl"
        entry = {
            "audio_filepath": str(audio_path.resolve()),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

        min_spk = min_speakers if min_speakers is not None else int(self.params.get("min_speakers", 1))
        max_spk = max_speakers if max_speakers is not None else int(self.params.get("max_speakers", 10))

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
                    "clustering": {
                        "parameters": {
                            "oracle_num_speakers": num_speakers is not None,
                            "max_num_speakers": max_spk,
                            "min_num_speakers": min_spk,
                            "max_rp_threshold": 0.25,
                            "sparse_search_volume": 60,
                            "embeddings_per_chunk": 10000,
                            "chunk_cluster_count": 80,
                        }
                    },
                },
                "device": self.device,
            }
        )
        cfg_overrides = self.params.get("cfg_overrides")
        if cfg_overrides:
            cfg = OmegaConf.merge(cfg, cfg_overrides)

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        diarizer = ClusteringDiarizer(cfg=cfg)
        diarizer.diarize()

        rttm_dir = Path(cfg.diarizer.out_dir)
        rttm_files = list(rttm_dir.rglob("*.rttm"))
        if not rttm_files:
            raise RuntimeError("NeMo diarization did not produce any RTTM output")

        uri = audio_path.stem
        segments = parse_rttm(rttm_files[0], uri=None)
        speakers_set = {segment.speaker for segment in segments if segment.speaker}
        return DiarizationResult(segments=segments, num_speakers=len(speakers_set))


class SpeechBrainDiarization(DiarizationProvider):
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)

    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        try:
            import torchaudio
            from speechbrain.inference.VAD import VAD
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.processing import diarization as diar
        except Exception as exc:
            raise RuntimeError("SpeechBrain diarization requires speechbrain, torchaudio, numpy") from exc

        audio_path = Path(audio_path)
        run_device = get_device(self.device)
        run_opts = {"device": run_device} if run_device else None

        wav, sr = self._load_mono_16k(audio_path)

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
            raise RuntimeError("No speech detected by SpeechBrain VAD")

        spk_model = self.params.get("speaker_model", "speechbrain/spkrec-ecapa-voxceleb")
        spk = EncoderClassifier.from_hparams(
            source=spk_model,
            savedir=self.params.get("speaker_savedir", "pretrained_sb_ecapa"),
            run_opts=run_opts,
        )

        win = float(self.params.get("win", 1.5))
        hop = float(self.params.get("hop", 0.75))
        min_spk = min_speakers if min_speakers is not None else int(self.params.get("min_spk", 2))
        max_spk = max_speakers if max_speakers is not None else int(self.params.get("max_spk", 10))
        pval = float(self.params.get("pval", 0.30))

        embs, seg_meta = self._extract_embeddings(wav, sr, speech_segs, spk, win, hop)
        if len(embs) < max(2, min_spk):
            raise RuntimeError(f"Too few segments ({len(embs)}) to cluster")

        clustering_method = self.params.get("clustering_method", "Spec_Clust_unorm")
        clustering_cls = getattr(diar, clustering_method, None)
        if clustering_cls is None:
            raise RuntimeError(f"Unknown SpeechBrain clustering method: {clustering_method}")

        clust = clustering_cls(min_num_spkrs=min_spk, max_num_spkrs=max_spk)
        oracle_k = num_speakers if num_speakers is not None else min(max_spk, max(min_spk, 2))
        clust.do_spec_clust(np.stack(embs, axis=0), k_oracle=oracle_k, p_val=pval)
        labels = clust.labels_.astype(int)

        segments = [
            SpeechSegment(start=start, end=end, speaker=f"spk{label}")
            for (start, end), label in zip(seg_meta, labels)
        ]
        speakers_set = {segment.speaker for segment in segments if segment.speaker}
        return DiarizationResult(segments=segments, num_speakers=len(speakers_set))

    def _load_mono_16k(self, audio_path: Path):
        import torchaudio

        wav, sr = torchaudio.load(str(audio_path))
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000
        return wav, sr

    def _extract_embeddings(self, wav, sr, speech_segs, speaker_model, win, hop):
        embs = []
        seg_meta = []
        for start, end in speech_segs:
            start = float(start)
            end = float(end)
            cursor = start
            while cursor < end:
                chunk_end = min(cursor + win, end)
                s0 = int(cursor * sr)
                s1 = int(chunk_end * sr)
                if s1 - s0 < int(0.2 * sr):
                    cursor += hop
                    continue
                chunk = wav[:, s0:s1]
                emb = speaker_model.encode_batch(chunk).detach().cpu().numpy().squeeze()
                embs.append(emb)
                seg_meta.append((cursor, chunk_end))
                cursor += hop
        return embs, seg_meta


class DiarizationFactory:
    PROVIDERS = {
        "pyannote": PyAnnoteDiarization,
        "nemo": NemoClusteringDiarization,
        "speechbrain": SpeechBrainDiarization,
    }

    @classmethod
    def create(
        cls,
        method: str,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ) -> DiarizationProvider:
        if method not in cls.PROVIDERS:
            supported = ", ".join(sorted(cls.PROVIDERS))
            raise ValueError(
                f"Unknown diarization method '{method}'. Supported methods: {supported}"
            )
        return cls.PROVIDERS[method](
            device=device,
            params=params,
            use_auth_token=use_auth_token,
        )
