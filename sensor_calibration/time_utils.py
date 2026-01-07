from datetime import datetime


def timestamp_str() -> str:
    """Return a standard timestamp string (local time) for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
