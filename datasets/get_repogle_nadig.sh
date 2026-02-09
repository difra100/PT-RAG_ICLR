#!/bin/bash
# ======================================================
# Download Replogle–Nadig Preprint dataset from Hugging Face
# ======================================================

# Directory where files will be stored
TARGET_DIR="./Replogle_Nadig_dataset"

# Create directory if it doesn’t exist
mkdir -p "$TARGET_DIR"

# Base URL for dataset files
BASE_URL="https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint/resolve/main"

# List of files to download
FILES=(
  "replogle.h5ad"
)

# Loop over each file
for FILE in "${FILES[@]}"; do
  echo "⬇️  Downloading $FILE..."
  wget -c "$BASE_URL/$FILE" -O "$TARGET_DIR/$FILE"
done

echo "✅ All downloads completed! Files saved in: $TARGET_DIR"
