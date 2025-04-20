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

# Set stage-specific environment variables
set -x STAGE "stage0"
set -x CONFIG_FILE "$SCRIPT_DIR/config/default_config.yaml"

# Create data directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/data"

# Run the program (python-dotenv will load .env)
python -m src.stage0.main 