"""jev_lite 패키지. Windows 콘솔(cp949)에서 한글 로그가 깨지지 않도록 stdio 를 UTF-8 로 맞춘다."""
import sys

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
