#!/bin/bash
#
# Install NVIDIA Nsight Systems (nsys) - Updated URLs
#

set -e

INSTALL_DIR="$HOME/.local/nsight-systems"

echo "========================================================================"
echo "Installing NVIDIA Nsight Systems"
echo "========================================================================"
echo ""
echo "NVIDIA frequently changes download URLs. Let me try multiple versions..."
echo ""

# Create install directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Try multiple versions until one works
VERSIONS=(
    "2024_5_1/NsightSystems-linux-cli-public-2024.5.1.91-3441458.tar.gz"
    "2024_4_1/NsightSystems-linux-cli-public-2024.4.1.42-3380207.tar.gz"
    "2024_2_1/NsightSystems-linux-cli-public-2024.2.1.106-3453501.tar.gz"
    "2023_4_4/NsightSystems-linux-cli-public-2023.4.4.32-3372216.tar.gz"
)

SUCCESS=0

for VERSION_PATH in "${VERSIONS[@]}"; do
    URL="https://developer.download.nvidia.com/devtools/nsight-systems/${VERSION_PATH}"

    echo "Trying: ${URL}"

    if wget --spider "$URL" 2>/dev/null; then
        echo "Found working URL!"
        echo "Downloading..."

        wget --progress=bar:force "$URL" -O nsight-systems.tar.gz

        echo "Extracting..."
        tar -xzf nsight-systems.tar.gz
        rm nsight-systems.tar.gz

        SUCCESS=1
        break
    else
        echo "  Not available, trying next..."
    fi
done

if [ $SUCCESS -eq 0 ]; then
    echo ""
    echo "ERROR: Could not find a working download URL"
    echo ""
    echo "Please download manually from:"
    echo "  https://developer.nvidia.com/nsight-systems/get-started"
    echo ""
    echo "Download the 'Linux CLI' version and extract to:"
    echo "  ${INSTALL_DIR}"
    echo ""
    exit 1
fi

# Find the extracted directory
EXTRACTED_DIR=$(ls -d nsight-systems-*/ 2>/dev/null | head -1)

if [ -z "$EXTRACTED_DIR" ]; then
    echo "Error: Could not find extracted directory"
    ls -la
    exit 1
fi

echo "Found: $EXTRACTED_DIR"

# Create symlink in ~/.local/bin
mkdir -p "$HOME/.local/bin"
ln -sf "$(realpath ${EXTRACTED_DIR}/bin/nsys)" "$HOME/.local/bin/nsys"

# Add to PATH if not already there
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    echo ""
    echo "Adding to PATH..."
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "========================================================================"
echo "Installation Complete!"
echo "========================================================================"
echo ""
echo "nsys installed to: $HOME/.local/bin/nsys"
echo ""
echo "To use nsys in your current shell:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo ""
echo "Verify installation:"
echo "  nsys --version"
echo ""
echo "========================================================================"
