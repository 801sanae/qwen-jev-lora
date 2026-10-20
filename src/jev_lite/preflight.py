"""Preflight: NVIDIA CUDA 사용 가능 여부를 로깅하고, 불가하면 계속 진행할지 묻는다.
실행: uv run -m jev_lite.preflight "vLLM 서빙"
비대화형(파이프·CI)에서는 묻지 않고 중단한다. 강제 진행: JEV_ALLOW_NO_CUDA=1
"""
import logging, os, shutil, subprocess, sys

log = logging.getLogger("jev_lite.preflight")
ALLOW_ENV = "JEV_ALLOW_NO_CUDA"


def _nvidia_smi() -> list[str] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return [l.strip() for l in out.splitlines() if l.strip()] or None


def check_cuda() -> bool:
    smi = _nvidia_smi()
    log.info("nvidia-smi: %s", "; ".join(smi) if smi else "없음 (드라이버 미설치 또는 WSL GPU 미연결)")
    try:
        import torch
    except ImportError:
        log.info("torch: 미설치 -> nvidia-smi 결과로만 판단")
        return smi is not None
    ok = torch.cuda.is_available()
    log.info("torch %s (CUDA build: %s), cuda.is_available=%s", torch.__version__, torch.version.cuda, ok)
    if smi and torch.version.cuda is None:
        log.warning("GPU는 있으나 torch가 CPU 빌드입니다. CUDA 빌드 torch를 설치하세요.")
    for i in range(torch.cuda.device_count() if ok else 0):
        prop = torch.cuda.get_device_properties(i)
        log.info("GPU%d: %s, sm%d%d, VRAM %.1fGiB", i, prop.name, prop.major, prop.minor, prop.total_memory / 2**30)
    return ok


def confirm_continue() -> bool:
    if os.getenv(ALLOW_ENV) == "1":
        log.warning("%s=1 -> CUDA 없이 계속 진행", ALLOW_ENV)
        return True
    if not sys.stdin.isatty():
        log.error("비대화형 환경이라 확인할 수 없어 중단합니다 (강제 진행: %s=1)", ALLOW_ENV)
        return False
    try:
        return input("CUDA 없이 계속 실행할까요? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def require_cuda(purpose: str) -> bool:
    """CUDA 가능 -> True. 불가 시 사용자에게 묻고, 승인하면 False(CPU 진행), 거부하면 exit(1)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [preflight] %(message)s")
    if check_cuda():
        log.info("CUDA 사용 가능 -> %s 시작", purpose)
        return True
    log.warning("CUDA 사용 불가: %s 이(가) CPU로 실행되어 매우 느리거나 실패할 수 있습니다.", purpose)
    if not confirm_continue():
        log.error("중단합니다.")
        sys.exit(1)
    return False


if __name__ == "__main__":
    require_cuda(sys.argv[1] if len(sys.argv) > 1 else "작업")
