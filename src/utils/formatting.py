
import time


def format_duration(seconds: float) -> str:

    if seconds < 60:
        return f"{seconds:.2f}s"

    if seconds < 3600:
        minutes = int(seconds // 60)
        remainder = seconds - minutes * 60
        return f"{minutes}m {remainder:04.1f}s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds - hours * 3600 - minutes * 60
    return f"{hours}h {minutes:02d}m {remainder:04.1f}s"


def print_stage_header(title: str, stage_number: int, total_stages: int) -> None:

    print()
    print("=" * 78)
    print(f"[{stage_number}/{total_stages}] {title}")
    print("=" * 78)


def print_elapsed(stage_start: float) -> None:

    elapsed = time.time() - stage_start
    print(f"Stage completed in {format_duration(elapsed)}")
