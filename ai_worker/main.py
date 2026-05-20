import signal
import time

_running = True


def _handle_shutdown(_signum: int, _frame: object) -> None:
    global _running
    _running = False


def main() -> None:
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    print("AI Worker started. Queue/OCR/ML/LLM jobs are not wired yet.", flush=True)
    while _running:
        time.sleep(30)
    print("AI Worker stopped.", flush=True)


if __name__ == "__main__":
    main()
