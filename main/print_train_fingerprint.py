import argparse
import hashlib
import json
import os
from pathlib import Path


TRAIN_CODE_FINGERPRINT_KEY = "train_code_fingerprint"


def compute_fingerprint(base_dir: Path) -> str:
    train_path = base_dir / "train.py"
    with open(train_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def sync_test_configs(base_dir: Path, fingerprint: str) -> int:
    test_dir = base_dir / "test"
    if not test_dir.exists():
        return 0

    updated = 0
    for config_path in sorted(test_dir.glob("*.json")):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config[TRAIN_CODE_FINGERPRINT_KEY] = fingerprint
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=True)
            f.write("\n")
        updated += 1
        print(f"updated: {config_path.name}")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Print train.py fingerprint and sync test configs by default.")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the fingerprint. Do not overwrite JSON configs under main/test/.",
    )
    args = parser.parse_args()

    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    fingerprint = compute_fingerprint(base_dir)
    print(fingerprint)

    if not args.print_only:
        updated = sync_test_configs(base_dir, fingerprint)
        print(f"synced_test_configs={updated}")


if __name__ == "__main__":
    main()
