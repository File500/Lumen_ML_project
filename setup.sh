##!/bin/bash
#
## Define colors for output
#GREEN='\033[0;32m'
#YELLOW='\033[1;33m'
#NC='\033[0m' # No Color
#
#ENV_NAME="gambit_env"
#REQUIREMENTS_FILE="requirements.txt"
#
#cd "$(dirname "$0")"
#
## Check if the environment directory exists
#if [ ! -d "$ENV_NAME" ]; then
#    echo -e "${YELLOW}Creating new virtual environment: $ENV_NAME${NC}"
#
#    # Create virtual environment
#    python -m venv $ENV_NAME
#
#    # Check if environment was created successfully
#    if [ ! -d "$ENV_NAME" ]; then
#        echo "Failed to create virtual environment. Exiting."
#        exit 1
#    fi
#
#    # Activate the environment
#    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win64" ]]; then
#        # Windows
#        $ENV_NAME\Scripts\activate || $ENV_NAME\Scripts\Activate.ps1
#    else
#        # Linux/macOS
#        source $ENV_NAME/bin/activate
#    fi
#
#    # Check if requirements file exists
#    if [ -f "$REQUIREMENTS_FILE" ]; then
#        echo -e "${YELLOW}Installing packages from $REQUIREMENTS_FILE${NC}"
#
#        # Upgrade pip
#        pip install --upgrade pip
#
#        pip install -r requirements.txt
#
#        # Install PyTorch with CUDA support
#        echo -e "${YELLOW}Installing PyTorch with CUDA support${NC}"
#        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#
#        echo -e "${GREEN}Installation complete!${NC}"
#    else
#        echo "Warning: $REQUIREMENTS_FILE not found. No packages were installed."
#    fi
#else
#    echo -e "${GREEN}Environment $ENV_NAME already exists.${NC}"
#
#    # Activate the environment
#    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win64" ]]; then
#        # Windows
#        $ENV_NAME\Scripts\activate || $ENV_NAME\Scripts\Activate.ps1
#    else
#        # Linux/macOS
#        source $ENV_NAME/bin/activate
#    fi
#fi
#
#echo -e "${GREEN}Environment $ENV_NAME is now active.${NC}"
#echo -e "To deactivate, run: ${YELLOW}deactivate${NC}"