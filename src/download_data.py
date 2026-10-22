"""Optional: re-download the original dataset and verify its checksum.

The dataset is already included in data/, so this script is NOT needed to run the project.
Usage:  python -m src.download_data
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.request

from . import config
from .preprocessing import file_sha256

EXPECTED_SHA256 = "1ae9329c3e1032acbb8dd080bfc397c44507006812a715d2788f172820489a1c"


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = config.RAW_FILE
    print(f"Downloading {config.DATA_URL}")
    try:
        urllib.request.urlretrieve(config.DATA_URL, target)
    except Exception as exc:  # network errors vary widely; report and stop
        raise SystemExit(f"Download failed: {exc}")
    digest = file_sha256(target)
    status = "matches" if digest == EXPECTED_SHA256 else "DIFFERS FROM"
    print(f"SHA-256 {digest} {status} the checksum of the file used in the report.")
    info = {"url": config.DATA_URL, "download_date_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
            "sha256": digest, "bytes": target.stat().st_size}
    config.DOWNLOAD_INFO_FILE.write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
