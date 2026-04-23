import argparse
import logging
from pathlib import Path
import shutil
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_cache import (
    ALIGNMENT_MODELS_DIR,
    DEFAULT_ALIGNMENT_MODEL,
    DEFAULT_PYANNOTE_DIARIZATION_MODEL,
    DEFAULT_PYANNOTE_VAD_MODEL,
    DEFAULT_WHISPER_MODEL,
    SPEECHBRAIN_MODELS_DIR,
    WHISPER_MODELS_DIR,
    configure_local_caches,
    get_default_hf_token,
    normalize_model_name,
)

configure_local_caches()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_global_hf_snapshot(repo_id: str) -> Path | None:
    repo_cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
    )
    if not repo_cache_dir.exists():
        return None

    snapshots = [path for path in repo_cache_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return None

    return max(snapshots, key=lambda path: path.stat().st_mtime)


def download_hf_repo(repo_id: str, destination_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = destination_root / normalize_model_name(repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Caching %s into %s", repo_id, local_dir)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        cached_snapshot = find_global_hf_snapshot(repo_id)
        if not cached_snapshot:
            raise
        logger.warning(
            "Falling back to the existing global HF cache for %s after download failure: %s",
            repo_id,
            exc,
        )
        shutil.copytree(cached_snapshot, local_dir, dirs_exist_ok=True)
    return local_dir


def prefetch_silero() -> None:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    logger.info("Prefetching Silero VAD weights")
    dummy_audio = np.zeros(16000, dtype=np.float32)
    get_speech_timestamps(dummy_audio, VadOptions())


def prefetch_alignment(model_name: str) -> None:
    import torchaudio

    logger.info("Prefetching alignment model %s", model_name)
    if model_name in torchaudio.pipelines.__all__:
        bundle = torchaudio.pipelines.__dict__[model_name]
        bundle.get_model(dl_kwargs={"model_dir": str(ALIGNMENT_MODELS_DIR)})
    else:
        download_hf_repo(model_name, ALIGNMENT_MODELS_DIR)


def prefetch_pyannote(token: str | None, include_diarization: bool) -> None:
    if not token:
        logger.warning("Skipping pyannote downloads because no Hugging Face token was found")
        return

    try:
        from pyannote.audio import Model, Pipeline
        from pyannote.audio.pipelines import VoiceActivityDetection

        logger.info("Prefetching pyannote VAD model")
        segmentation = Model.from_pretrained(DEFAULT_PYANNOTE_VAD_MODEL, use_auth_token=token)
        vad_pipeline = VoiceActivityDetection(segmentation=segmentation)
        vad_pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})

        if include_diarization:
            logger.info("Prefetching pyannote diarization pipeline")
            Pipeline.from_pretrained(DEFAULT_PYANNOTE_DIARIZATION_MODEL, token=token)
    except Exception as exc:
        logger.warning("Skipping pyannote downloads because authentication or access failed: %s", exc)


def prefetch_speechbrain() -> None:
    from speechbrain.inference.VAD import VAD
    from speechbrain.inference.speaker import EncoderClassifier

    logger.info("Prefetching SpeechBrain VAD and speaker models")
    VAD.from_hparams(
        source="speechbrain/vad-crdnn-libriparty",
        savedir=str(SPEECHBRAIN_MODELS_DIR / "vad-crdnn-libriparty"),
    )
    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(SPEECHBRAIN_MODELS_DIR / "spkrec-ecapa-voxceleb"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model assets into repo-local caches")
    parser.add_argument(
        "--whisper-model",
        action="append",
        default=[DEFAULT_WHISPER_MODEL],
        help="Whisper model repo id to cache locally. Can be specified multiple times.",
    )
    parser.add_argument(
        "--alignment-model",
        default=DEFAULT_ALIGNMENT_MODEL,
        help="Alignment model bundle/repo to prefetch",
    )
    parser.add_argument(
        "--skip-silero",
        action="store_true",
        help="Skip Silero VAD prefetch",
    )
    parser.add_argument(
        "--skip-pyannote",
        action="store_true",
        help="Skip pyannote VAD/diarization prefetch",
    )
    parser.add_argument(
        "--skip-pyannote-diarization",
        action="store_true",
        help="Only cache pyannote VAD, not the diarization pipeline",
    )
    parser.add_argument(
        "--include-speechbrain",
        action="store_true",
        help="Also prefetch SpeechBrain VAD and speaker models",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Optional Hugging Face token. Falls back to saved local login.",
    )
    args = parser.parse_args()

    token = args.hf_token or get_default_hf_token()

    whisper_models = []
    for model_name in args.whisper_model:
        if model_name not in whisper_models:
            whisper_models.append(model_name)

    for model_name in whisper_models:
        download_hf_repo(model_name, WHISPER_MODELS_DIR)

    prefetch_alignment(args.alignment_model)

    if not args.skip_silero:
        prefetch_silero()

    if not args.skip_pyannote:
        prefetch_pyannote(token, include_diarization=not args.skip_pyannote_diarization)

    if args.include_speechbrain:
        prefetch_speechbrain()

    logger.info("Asset prefetch complete")


if __name__ == "__main__":
    main()
