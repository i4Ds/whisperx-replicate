import os
import re
import site
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODEL_STORE = REPO_ROOT / "model-store"
HF_HOME = MODEL_STORE / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
TRANSFORMERS_CACHE = HF_HOME / "transformers"
TORCH_HOME = MODEL_STORE / "torch"
XDG_CACHE_HOME = MODEL_STORE / "xdg"
WHISPER_MODELS_DIR = MODEL_STORE / "whisper"
ALIGNMENT_MODELS_DIR = MODEL_STORE / "alignment"
SPEECHBRAIN_MODELS_DIR = MODEL_STORE / "speechbrain"
NEMO_MODELS_DIR = MODEL_STORE / "nemo"
OUTPUTS_DIR = REPO_ROOT / "outputs"

DEFAULT_WHISPER_MODEL = "i4ds/daily-brook-134"
DEFAULT_ALIGNMENT_MODEL = "VOXPOPULI_ASR_BASE_10K_DE"
DEFAULT_PYANNOTE_VAD_MODEL = "pyannote/segmentation-3.0"
DEFAULT_PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def ensure_pytorch_nvidia_libraries_first() -> None:
    """Relaunch Python with PyTorch's bundled NVIDIA libraries first.

    Cog's CUDA base image can expose an older cuDNN through the dynamic linker.
    PyTorch wheels ship their matching cuDNN under site-packages/nvidia, so put
    those directories first before torch is imported.
    """
    if os.name != "posix" or sys.platform != "linux":
        return
    if os.environ.get("STT4SG_NVIDIA_LIBS_FIRST") == "1":
        return

    roots = []
    try:
        roots.extend(Path(path) for path in site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    roots.extend(Path(path) for path in sys.path if path)

    library_dirs = []
    for root in dict.fromkeys(path for path in roots if path.exists()):
        nvidia_root = root / "nvidia"
        if nvidia_root.exists():
            library_dirs.extend(path for path in nvidia_root.glob("*/lib") if path.is_dir())
        torch_lib = root / "torch" / "lib"
        if torch_lib.exists():
            library_dirs.append(torch_lib)

    library_dirs = list(dict.fromkeys(library_dirs))
    if not library_dirs:
        return

    current_paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    preferred_paths = [str(path) for path in library_dirs]
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        preferred_paths + [path for path in current_paths if path not in preferred_paths]
    )
    os.environ["STT4SG_NVIDIA_LIBS_FIRST"] = "1"
    os.execv(sys.executable, getattr(sys, "orig_argv", [sys.executable, *sys.argv]))


def resolve_hf_hub_snapshot(repo_id: str, filename: str | None = None) -> str:
    """Return a local path for a cached HF Hub model file.

    If *filename* is given, returns the path to that specific file inside the
    snapshot directory (e.g. ``"pytorch_model.bin"`` or ``"config.yaml"``).
    If *filename* is omitted, returns the snapshot directory itself.

    Falls back to *repo_id* so callers still work when the model is not
    pre-bundled (HF hub will download it at runtime if needed).
    """
    repo_dir = HF_HUB_CACHE / f"models--{repo_id.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        snapshot_hash = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = repo_dir / "snapshots" / snapshot_hash
        if snapshot_dir.exists():
            if filename is None:
                return str(snapshot_dir)
            candidate = snapshot_dir / filename
            if candidate.exists():
                return str(candidate)
    return repo_id


def configure_local_caches() -> None:
    for path in (
        MODEL_STORE,
        HF_HOME,
        HF_HUB_CACHE,
        TRANSFORMERS_CACHE,
        TORCH_HOME,
        XDG_CACHE_HOME,
        WHISPER_MODELS_DIR,
        ALIGNMENT_MODELS_DIR,
        SPEECHBRAIN_MODELS_DIR,
        NEMO_MODELS_DIR,
        OUTPUTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
    os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    patch_torchaudio_compat()


def patch_torchaudio_compat() -> None:
    # Only patch if torchaudio is already fully imported — never trigger the
    # import here, as doing so at module-load time causes a circular-import
    # error ("partially initialized module 'torchaudio' has no attribute 'lib'").
    import sys
    torchaudio = sys.modules.get("torchaudio")
    if torchaudio is None:
        return

    if not hasattr(torchaudio, "_stt4sg_audio_backend"):
        torchaudio._stt4sg_audio_backend = "ffmpeg"

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["ffmpeg", "soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: getattr(
            torchaudio, "_stt4sg_audio_backend", None
        )
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: setattr(
            torchaudio, "_stt4sg_audio_backend", backend
        )


def get_default_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
    except Exception:
        token = None

    if not token:
        for token_path in (
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".huggingface" / "token",
        ):
            try:
                raw_token = token_path.read_text(encoding="utf-8").strip()
            except Exception:
                raw_token = ""
            if raw_token:
                token = raw_token
                break

    if token:
        os.environ.setdefault("HF_TOKEN", token)
    return token


def normalize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", model_name.strip("/"))


def resolve_local_dir(root: Path, model_name: str) -> Path | None:
    candidate = root / normalize_model_name(model_name)
    return candidate if candidate.exists() else None


def resolve_whisper_model_spec(model_name: str) -> str:
    if not model_name:
        model_name = DEFAULT_WHISPER_MODEL

    path = Path(model_name)
    if path.exists():
        return str(path)

    local_dir = resolve_local_dir(WHISPER_MODELS_DIR, model_name)
    if local_dir:
        return str(local_dir)

    return model_name


def resolve_alignment_model_spec(model_name: str | None) -> str | None:
    if not model_name:
        return None

    path = Path(model_name)
    if path.exists():
        return str(path)

    local_dir = resolve_local_dir(ALIGNMENT_MODELS_DIR, model_name)
    if local_dir:
        return str(local_dir)

    return model_name


ensure_pytorch_nvidia_libraries_first()
configure_local_caches()
