#!/bin/bash

# Create a timestamped folder name (format: YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NEW_FOLDER="archive_${TIMESTAMP}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source directory (attention_weights)
SOURCE_DIR="${SCRIPT_DIR}/attention_weights"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Create the new folder inside attention_weights
echo "Creating folder: ${SOURCE_DIR}/${NEW_FOLDER}"
mkdir -p "${SOURCE_DIR}/${NEW_FOLDER}"

# Move only files (not folders) to the new folder
echo "Moving files from attention_weights to ${NEW_FOLDER}..."
moved_count=0
for item in "${SOURCE_DIR}"/*; do
    # Only move files, skip all directories (including archive folders)
    if [ -f "$item" ]; then
        item_name=$(basename "$item")
        echo "  Moving: $item_name"
        mv "$item" "${SOURCE_DIR}/${NEW_FOLDER}/"
        ((moved_count++))
    fi
done

echo "Done! Moved $moved_count items to: ${SOURCE_DIR}/${NEW_FOLDER}"
