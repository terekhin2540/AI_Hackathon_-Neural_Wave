#!/bin/bash

# This script creates symlinks to the original images to avoid copying.


if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <file_with_paths> <target_directory>"
  exit 1
fi

# Arguments
FILE_WITH_PATHS="$1" # text file with aligned/unaligned image paths
TARGET_DIR="$2" # where we wanna have these symlinks

# Check if the target dir exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Target directory does not exist: $TARGET_DIR"
  exit 1
fi

# Read file line by line
while IFS= read -r line
do
  # Check the source file for existence
  if [ ! -f "$line" ]; then
    echo "Source file does not exist: $line"
    continue
  fi

  # Get the name of file and folder
  directory=$(basename "$(dirname "$line")")
  filename=$(basename "$line")

  # Form a name of the symbolic link
  symlink_name="${directory}_${filename}"

  # Create symbolic link
  ln -s "$line" "$TARGET_DIR/$symlink_name"
done < "$FILE_WITH_PATHS"
