import os
import shutil
import zipfile
import re
from datetime import datetime

def archive_and_cleanup(results_dir, keep_recent=3, keep_total=15):
    """
    Archives older training results and cleans up very old ones.
    
    Args:
        results_dir (str): Path to the results directory.
        keep_recent (int): Number of most recent results to keep as folders (uncompressed).
        keep_total (int): Total number of results to keep (folders + zips). 
                          Everything older than this count will be deleted.
    """
    if not os.path.exists(results_dir):
        print(f"[Archive] Results directory {results_dir} does not exist. Skipping cleanup.")
        return

    print(f"\n--- Running Result Archiving & Cleanup ---")
    print(f"Target: {results_dir}")
    print(f"Policy: Keep recent {keep_recent} raw, compress others, keep max {keep_total} total.")

    # Regex to match default naming format: name_YYYYMMDD_HHMM
    # Examples: default_run_5det_20231027_1030, exp1_20231027_1030.zip
    # We look for the date pattern at the end (ignoring .zip extension)
    name_pattern = re.compile(r'^(.+)_(\d{8}_\d{4})(?:\.zip)?$')

    items = []
    for entry in os.listdir(results_dir):
        full_path = os.path.join(results_dir, entry)
        
        # Check if it matches pattern
        match = name_pattern.match(entry)
        if not match:
            # Skip files/folders that don't match the naming convention
            continue
            
        # Parse timestamp for sorting
        timestamp_str = match.group(2)
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
        except ValueError:
            continue
            
        is_zip = entry.endswith('.zip')
        items.append({
            'path': full_path,
            'name': entry,
            'time': timestamp,
            'is_zip': is_zip,
            'base_name': match.group(1) + '_' + match.group(2) # name without extension
        })

    # Sort by time descending (newest first)
    items.sort(key=lambda x: x['time'], reverse=True)

    # Process items
    for i, item in enumerate(items):
        # 1. Delete if beyond keep_total
        if i >= keep_total:
            print(f"[Delete] Deleting old result: {item['name']}")
            try:
                if os.path.isdir(item['path']):
                    shutil.rmtree(item['path'])
                else:
                    os.remove(item['path'])
            except Exception as e:
                print(f"  Error deleting {item['name']}: {e}")
            continue

        # 2. Keep recent as is (folders)
        if i < keep_recent:
            # Ideally these should be folders. If it's already a zip, we leave it alone.
            continue

        # 3. Compress intermediate items (index keep_recent to keep_total-1)
        # If it's already a zip, do nothing.
        # If it's a folder, compress it and delete the folder.
        if not item['is_zip'] and os.path.isdir(item['path']):
            zip_path = os.path.join(results_dir, item['base_name'] + ".zip")
            print(f"[Archive] Compressing result: {item['name']} -> {os.path.basename(zip_path)}")
            
            try:
                # Create Zip
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Walk through the directory
                    for root, dirs, files in os.walk(item['path']):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Archive name should be relative to the results_dir so it unzips into a folder
                            # We want the zip to contain the folder "exp_name_date/..."
                            # So we calculate relpath from results_dir
                            arcname = os.path.relpath(file_path, results_dir)
                            zipf.write(file_path, arcname)
                
                # Verify zip creation before deleting original folder
                if os.path.exists(zip_path):
                    shutil.rmtree(item['path'])
            except Exception as e:
                print(f"  Failed to archive {item['name']}: {e}")
                # If zip was created partially, try to remove it
                if os.path.exists(zip_path):
                    try:
                        os.remove(zip_path)
                    except:
                        pass

if __name__ == "__main__":
    # Test block
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Default results dir relative to this script
    test_results_dir = os.path.join(BASE_DIR, 'results')
    archive_and_cleanup(test_results_dir)
