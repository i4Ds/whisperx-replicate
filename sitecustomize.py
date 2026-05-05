"""Runtime library path fixes loaded before application imports."""

import os
import site
import sys
from pathlib import Path


def _ensure_pytorch_nvidia_libraries_first() -> None:
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


_ensure_pytorch_nvidia_libraries_first()
