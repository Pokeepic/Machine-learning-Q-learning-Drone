#!/bin/bash
set -e

PROJECT_DIR="ArduinoControl"
FQBN="arduino:avr:uno"

echo "User        : $USER"
echo "Project     : $PROJECT_DIR"
echo "Board       : $FQBN"
echo "-----------------------------------"

# ---------- Check arduino-cli ----------
if ! command -v arduino-cli >/dev/null 2>&1; then
    echo "ERROR: arduino-cli not found in PATH"
    echo "Fix  : sudo apt install arduino-cli"
    exit 1
fi

# ---------- Detect board ----------
echo "Detecting connected Arduino..."
BOARD_INFO=$(arduino-cli board list | grep -E "ttyACM|ttyUSB" || true)

if [ -z "$BOARD_INFO" ]; then
    echo "ERROR: No Arduino detected"
    echo "Checks:"
    echo " - USB cable supports data"
    echo " - Board is powered"
    echo " - Device appears under /dev/ttyACM* or /dev/ttyUSB*"
    exit 1
fi

echo "$BOARD_INFO"

# ---------- Extract port ----------
PORT=$(echo "$BOARD_INFO" | awk '{print $1}' | head -n 1)

if [ -z "$PORT" ]; then
    echo "ERROR: Unable to resolve serial port"
    exit 1
fi

echo "Using port : $PORT"

# ---------- Compile ----------
echo "Compiling sketch..."
arduino-cli compile --fqbn "$FQBN" "$PROJECT_DIR"

# ---------- Upload ----------
echo "Uploading firmware..."
arduino-cli upload -p "$PORT" --fqbn "$FQBN" "$PROJECT_DIR"

# ---------- Run Python ----------
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found"
    exit 1
fi

echo "Starting Python controller..."
python3 main.py
