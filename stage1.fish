#!/usr/bin/env fish

# Get the directory where the script is located
set SCRIPT_DIR (dirname (status -f))

# Define CLI options
argparse 'reset' 'no-resume' -- $argv

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
set -x STAGE "stage1"
set -x CONFIG_FILE "$SCRIPT_DIR/config/default_config.yaml"

# Setup resume options based on arguments
set CMD "python $SCRIPT_DIR/main.py stage1"

# Handle --reset option
if set -q _flag_reset
    set CMD "$CMD --reset"
end

# Handle --no-resume option
if set -q _flag_no_resume
    set CMD "$CMD --no-resume"
end

# Run the program with the constructed command
eval $CMD 