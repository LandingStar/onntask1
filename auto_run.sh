#!/bin/bash

# Configuration
MAX_RETRIES=10
RETRY_DELAY=5  # Seconds to wait between retries
LOG_FILE="auto_run.log"

# Navigate to the correct directory (assuming the script is run from or near task1)
# We use the script's directory as the base to ensure relative paths work
cd "$(dirname "$0")"
git config --global --add safe.directory /public/home/landingstar/onn_training/onntask1
# Initialize or clear the log file
echo "=== Auto Pull and Run Script Started at $(date) ===" > "$LOG_FILE"

# Function to perform git update
update_repo() {
    echo "[INFO] Fetching latest changes from origin..." >> "$LOG_FILE"
    git fetch origin master >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Git fetch failed." >> "$LOG_FILE"
        return 1
    fi

    echo "[INFO] Force resetting to origin/master..." >> "$LOG_FILE"
    git reset --hard origin/master >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Git reset failed." >> "$LOG_FILE"
        return 1
    fi
    
    # Optional: clean untracked files
    # git clean -fd >> "$LOG_FILE" 2>&1

    return 0
}

# Retry Loop
attempt=1
success=false

while [ $attempt -le $MAX_RETRIES ]; do
    echo "[INFO] Attempt $attempt of $MAX_RETRIES to update repository..." >> "$LOG_FILE"
    
    if update_repo; then
        echo "[SUCCESS] Repository updated successfully on attempt $attempt." >> "$LOG_FILE"
        success=true
        break
    else
        echo "[WARNING] Attempt $attempt failed." >> "$LOG_FILE"
        if [ $attempt -lt $MAX_RETRIES ]; then
            echo "[INFO] Waiting $RETRY_DELAY seconds before next attempt..." >> "$LOG_FILE"
            sleep $RETRY_DELAY
        fi
    fi
    
    attempt=$((attempt + 1))
done

# Execute script if update was successful
if [ "$success" = true ]; then
    echo "[INFO] Starting training script: python ai_refined_5_detectors/train.py" >> "$LOG_FILE"
    echo "---------------------------------------------------" >> "$LOG_FILE"
    
    # Run the python script and append output to the same log file
    # 2>&1 redirects stderr to stdout so both go to the log
    python ai_refined_5_detectors/train.py >> "$LOG_FILE" 2>&1
    
    TRAIN_EXIT_CODE=$?
    echo "---------------------------------------------------" >> "$LOG_FILE"
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "[SUCCESS] Training script completed successfully at $(date)." >> "$LOG_FILE"
    else
        echo "[ERROR] Training script exited with code $TRAIN_EXIT_CODE at $(date)." >> "$LOG_FILE"
    fi
else
    echo "[FATAL] Failed to update repository after $MAX_RETRIES attempts. Aborting." >> "$LOG_FILE"
    exit 1
fi

echo "=== Script Finished ===" >> "$LOG_FILE"
exit 0