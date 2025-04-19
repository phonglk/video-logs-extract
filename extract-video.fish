#!/usr/bin/env fish

# Get the directory where the script is located
set SCRIPT_DIR (dirname (status -f))

# Check if .env file exists
if not test -f "$SCRIPT_DIR/.env"
    echo "Error: .env file not found!"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "cp .env.example .env"
    exit 1
end

# Activate virtual environment if it exists
if test -d "$SCRIPT_DIR/venv"
    source "$SCRIPT_DIR/venv/bin/activate.fish"
end

# Run the program
python "$SCRIPT_DIR/main.py" 