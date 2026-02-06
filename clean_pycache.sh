#!/bin/bash

START=$(date +%s)
echo "Cleaning __pycache__ directories and .pyc files (excluding all .venv and .git folders)..."

# 強制削除（書き込み保護されていても削除）
find . -type d \( -name '.venv' -o -name '.git' \) -prune -o \
    -type d -name '__pycache__' -print -exec rm -rf {} + 2>/dev/null

find . -type d \( -name '.venv' -o -name '.git' \) -prune -o \
    -type f -name '*.pyc' -print -exec rm -f {} + 2>/dev/null

END=$(date +%s)
echo "Cleanup complete in $((END - START)) seconds"