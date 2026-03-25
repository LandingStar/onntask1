"""
Batch training script: runs train.py for each config in batch_config/.

Place an optional overall_config.json in batch_config/ to configure:
  - Batch-level options: max_parallel (default 1)
  - Shared training parameters that are merged into every individual config.

Individual config values take priority over overall_config values.
overall_config.json itself is NOT treated as a training config.

Usage:
    python batch_train.py
"""
import os
import sys
import json
import shutil
import subprocess
import time
import concurrent.futures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_CONFIG_DIR = os.path.join(BASE_DIR, 'batch_config')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
TRAIN_SCRIPT = os.path.join(BASE_DIR, 'train.py')
OVERALL_CONFIG_NAME = 'overall_config.json'

# Directory for storing temporary generated configs
TEMP_CONFIG_DIR = os.path.join(BASE_DIR, '.temp_batch_configs')

# Keys reserved for batch_train control (not merged into training configs)
BATCH_KEYS = {'max_parallel'}


def load_overall_config():
    """Load overall_config.json. Returns (batch_opts, shared_training_params)."""
    path = os.path.join(BATCH_CONFIG_DIR, OVERALL_CONFIG_NAME)
    if not os.path.exists(path):
        return {}, {}
    with open(path, 'r') as f:
        overall = json.load(f)
    batch_opts = {k: overall[k] for k in BATCH_KEYS if k in overall}
    shared_params = {k: v for k, v in overall.items() if k not in BATCH_KEYS}
    return batch_opts, shared_params


def get_config_files():
    """Get sorted list of .json training configs (excluding overall_config.json)."""
    if not os.path.isdir(BATCH_CONFIG_DIR):
        print(f"Batch config directory not found: {BATCH_CONFIG_DIR}")
        return []
    files = [f for f in os.listdir(BATCH_CONFIG_DIR)
             if f.endswith('.json') and f != OVERALL_CONFIG_NAME]
    files.sort()
    return files


def merge_config(shared_params, individual_path):
    """Merge shared params with individual config. Individual values take priority."""
    with open(individual_path, 'r') as f:
        individual = json.load(f)
    merged = {**shared_params, **individual}
    return merged


def run_one(cfg_file, shared_params, idx, total):
    """Run a single training. Returns (cfg_file, status, detail)."""
    cfg_path = os.path.join(BATCH_CONFIG_DIR, cfg_file)

    try:
        merged = merge_config(shared_params, cfg_path)
        # If exp_name is set in the individual config, use it. Otherwise, use the filename without .json
        exp_name = merged.get('exp_name', os.path.splitext(cfg_file)[0])
        # Force the exp_name in the merged config so train.py picks it up correctly
        merged['exp_name'] = exp_name
        # Important: set batch_train to False in the merged config so we don't infinitely loop
        merged['batch_train'] = False
    except json.JSONDecodeError as e:
        print(f"  [{idx}/{total}] ERROR: Invalid JSON in {cfg_file}: {e}")
        return (cfg_file, 'SKIPPED', 'Invalid JSON')

    # Ensure temp directory exists
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

    # Write merged config to a unique temp file in the temp directory
    tmp_config = os.path.join(TEMP_CONFIG_DIR, f'config_batch_{idx}.json')
    with open(tmp_config, 'w') as f:
        json.dump(merged, f, indent=4)

    print(f"\n{'='*60}")
    print(f"[{idx}/{total}] {cfg_file}  (exp_name: {exp_name})")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        cmd = [sys.executable, TRAIN_SCRIPT, tmp_config, "--is-subprocess"]
        print(f"  [{idx}/{total}] Launching: {' '.join(cmd)}")
        
        proc = subprocess.run(
            cmd,
            cwd=BASE_DIR,
        )
        elapsed = time.time() - start_time
        status = 'OK' if proc.returncode == 0 else f'FAIL (code {proc.returncode})'
        print(f"  [{idx}/{total}] Finished: {status} in {elapsed:.1f}s")
        return (cfg_file, status, f'{elapsed:.1f}s')
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [{idx}/{total}] ERROR: {e}")
        return (cfg_file, 'ERROR', str(e))
    finally:
        if os.path.exists(tmp_config):
            os.remove(tmp_config)


def main():
    print(f"\n[BATCH INIT] Starting batch_train.py")
    batch_opts, shared_params = load_overall_config()
    max_parallel = batch_opts.get('max_parallel', 1)

    config_files = get_config_files()
    if not config_files:
        print("[BATCH INIT] No training config files found in batch_config/. Nothing to do.")
        return

    total = len(config_files)

    if shared_params:
        print(f"Shared params from {OVERALL_CONFIG_NAME}:")
        for k, v in shared_params.items():
            print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print(f"Batch Training: {total} config(s), max_parallel={max_parallel}")
    print(f"{'='*60}")

    results = []

    try:
        if max_parallel <= 1:
            for idx, cfg_file in enumerate(config_files, 1):
                result = run_one(cfg_file, shared_params, idx, total)
                results.append(result)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = {}
                for idx, cfg_file in enumerate(config_files, 1):
                    future = executor.submit(run_one, cfg_file, shared_params, idx, total)
                    futures[future] = cfg_file
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            order = {name: i for i, name in enumerate(config_files)}
            results.sort(key=lambda r: order.get(r[0], 0))
    finally:
        # Clean up temp directory if empty or delete entirely
        if os.path.exists(TEMP_CONFIG_DIR):
            try:
                shutil.rmtree(TEMP_CONFIG_DIR)
            except Exception as e:
                print(f"Warning: Could not remove temporary config directory {TEMP_CONFIG_DIR}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("Batch Training Summary")
    print(f"{'='*60}")
    for cfg_file, status, detail in results:
        print(f"  {cfg_file:40s} {status:12s} {detail}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
