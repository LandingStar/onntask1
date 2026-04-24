"""
Batch training script: runs train.py for each config in batch_config/.

Place an optional overall_config.json in batch_config/ to configure:
  - Batch-level options: max_parallel (default 1)
  - Shared training parameters that are merged into every individual config.

Individual config values take priority over overall_config values.
overall_config.json itself is NOT treated as a training config.

Usage:
    python batch_train.py
    python batch_train.py --train-script train_boost.py
"""
import os
# Force thread limits BEFORE any other heavy imports to prevent RLIMIT_NPROC explosion in child processes
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import shutil
import subprocess
import time
import concurrent.futures
import hashlib
import argparse
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_CONFIG_DIR = os.path.join(BASE_DIR, 'batch_config')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
DEFAULT_TRAIN_SCRIPT = os.path.join(BASE_DIR, 'train.py')
OVERALL_CONFIG_NAME = 'overall_config.json'

# Directory for storing temporary generated configs
TEMP_CONFIG_DIR = os.path.join(BASE_DIR, '.temp_batch_configs')

# Keys reserved for batch_train control (not merged into training configs)
BATCH_KEYS = {'max_parallel', 'use_ddp', 'nproc_per_node', 'master_port'}
TRAIN_CODE_FINGERPRINT_KEY = 'train_code_fingerprint'
INLINE_MODULE_CACHE = {}


def parse_batch_args():
    parser = argparse.ArgumentParser(description="Run batch training configs with an optional training entry script.")
    parser.add_argument(
        '--train-script',
        default='',
        help='Path to the training script entrypoint. Empty means using train.py.'
    )
    return parser.parse_args()


def resolve_train_script(train_script_arg):
    script_value = str(train_script_arg or '').strip()
    if not script_value:
        return DEFAULT_TRAIN_SCRIPT
    if os.path.isabs(script_value):
        return script_value
    return os.path.abspath(os.path.join(BASE_DIR, script_value))


def get_fingerprint_key(train_script_path):
    script_name = os.path.splitext(os.path.basename(train_script_path))[0]
    if script_name == 'train':
        return TRAIN_CODE_FINGERPRINT_KEY
    return f'{script_name}_code_fingerprint'


def compute_train_code_fingerprint(train_script_path):
    with open(train_script_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def validate_train_code_fingerprint(config_obj, train_script_path):
    fingerprint_key = get_fingerprint_key(train_script_path)
    expected = str(config_obj.get(fingerprint_key, '')).strip()
    actual = compute_train_code_fingerprint(train_script_path)

    # Sidecar scripts such as train_boost.py may use their own fingerprint key.
    # If the dedicated key is absent, we treat it as "no fingerprint enforcement"
    # so existing configs can still be reused intentionally.
    if fingerprint_key != TRAIN_CODE_FINGERPRINT_KEY and not expected:
        return True, expected, actual, fingerprint_key

    if not expected:
        return True, expected, actual, fingerprint_key
    return expected == actual, expected, actual, fingerprint_key


def load_inline_train_module(train_script_path):
    abs_path = os.path.abspath(train_script_path)
    if abs_path in INLINE_MODULE_CACHE:
        return INLINE_MODULE_CACHE[abs_path]

    module_name = f"batch_train_entry_{hashlib.sha1(abs_path.encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module from {abs_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    INLINE_MODULE_CACHE[abs_path] = module
    return module


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


def run_one(cfg_file, shared_params, batch_opts, train_script_path, idx, total):
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

    fingerprint_ok, expected_fingerprint, actual_fingerprint, fingerprint_key = validate_train_code_fingerprint(
        merged,
        train_script_path
    )
    if expected_fingerprint:
        print(f"  [{idx}/{total}] {os.path.basename(train_script_path)} fingerprint: {actual_fingerprint}")
    if not fingerprint_ok:
        print(
            f"  [{idx}/{total}] WARNING: Fingerprint mismatch for {cfg_file}. "
            f"Expected {expected_fingerprint} in {fingerprint_key}, current {os.path.basename(train_script_path)} is {actual_fingerprint}. Skipping."
        )
        return (cfg_file, 'SKIPPED', f'Fingerprint mismatch ({expected_fingerprint} != {actual_fingerprint})')

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
        use_ddp = batch_opts.get('use_ddp', False)
        nproc_per_node = batch_opts.get('nproc_per_node', 'gpu')
        # We assign a unique port per run to avoid port collision if running max_parallel > 1
        base_port = batch_opts.get('master_port', 29500)
        master_port = base_port + idx

        if use_ddp:
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={nproc_per_node}",
                f"--master_port={master_port}",
                train_script_path, tmp_config, "--is-subprocess"
            ]
            print(f"  [{idx}/{total}] Launching: {' '.join(cmd)}")
            proc = subprocess.run(cmd, cwd=BASE_DIR)
            returncode = proc.returncode
        else:
            # Instead of spawning a subprocess, run inline to share RAM cache
            print(f"  [{idx}/{total}] Running inline via {os.path.basename(train_script_path)}: {tmp_config}")
            train_module = load_inline_train_module(train_script_path)
            # Set sys.argv so the chosen training script parses the right config
            sys.argv = [train_script_path, tmp_config]
            try:
                train_module.main()
                returncode = 0
            except Exception as e:
                print(f"  [{idx}/{total}] ERROR running inline: {e}")
                returncode = 1
                
        elapsed = time.time() - start_time
        status = 'OK' if returncode == 0 else f'FAIL (code {returncode})'
        print(f"  [{idx}/{total}] Finished: {status} in {elapsed:.1f}s")
        return (cfg_file, status, f'{elapsed:.1f}s')
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [{idx}/{total}] ERROR: {e}")
        return (cfg_file, 'ERROR', str(e))
    finally:
        # Keep temp configs around for debugging if needed, or selectively remove them
        # Previously this was: os.remove(tmp_config)
        # But we'll leave it out for a moment to ensure it's not being deleted prematurely by concurrent runs
        pass


def main():
    args = parse_batch_args()
    train_script_path = resolve_train_script(args.train_script)
    if not os.path.exists(train_script_path):
        print(f"[BATCH INIT] Training script not found: {train_script_path}")
        sys.exit(1)

    print(f"\n[BATCH INIT] Starting batch_train.py")
    print(f"[BATCH INIT] Training entry script: {train_script_path}")
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
                result = run_one(cfg_file, shared_params, batch_opts, train_script_path, idx, total)
                results.append(result)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = {}
                for idx, cfg_file in enumerate(config_files, 1):
                    future = executor.submit(run_one, cfg_file, shared_params, batch_opts, train_script_path, idx, total)
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
