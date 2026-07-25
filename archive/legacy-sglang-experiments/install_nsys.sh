#!/bin/bash
#
# Install NVIDIA Nsight Systems (nsys) locally without sudo
#

set -e

INSTALL_DIR="$HOME/.local/nsight-systems"

echo "========================================================================"
echo "Installing NVIDIA Nsight Systems"
echo "========================================================================"
echo ""
echo "Installation directory: ${INSTALL_DIR}"
echo ""

# Create install directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download Nsight Systems 2026.2.1
echo "Downloading Nsight Systems 2026.2.1..."
wget --progress=bar:force \
    "https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_2/NsightSystems-linux-public-2026.2.1.210-3763964.run" \
    -O nsight-systems.run

# Make it executable
chmod +x nsight-systems.run

# Run installer to extract to our local directory
echo ""
echo "Installing to ${INSTALL_DIR}..."

# Try non-interactive installation
# The .run file usually supports --accept --quiet --prefix or --target
./nsight-systems.run --accept --quiet --prefix="${INSTALL_DIR}" || \
    ./nsight-systems.run --accept --target="${INSTALL_DIR}" || \
    {
        echo "Interactive installation required. The installer will prompt you."
        echo "When asked for installation directory, enter: ${INSTALL_DIR}"
        ./nsight-systems.run
    }

# Clean up installer
rm nsight-systems.run

# Find nsys binary
NSYS_BIN=$(find "${INSTALL_DIR}" -name nsys -type f | head -1)

if [ -z "$NSYS_BIN" ]; then
    echo "Error: Could not find nsys binary after installation"
    echo "Contents of ${INSTALL_DIR}:"
    ls -la "${INSTALL_DIR}"
    exit 1
fi

echo "Found nsys at: $NSYS_BIN"

# Create symlink in ~/.local/bin
mkdir -p "$HOME/.local/bin"
ln -sf "$(realpath $NSYS_BIN)" "$HOME/.local/bin/nsys"

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
echo "Or start a new shell (it's already in your .bashrc)"
echo ""
echo "Verify installation:"
echo "  \$HOME/.local/bin/nsys --version"
echo ""
echo "========================================================================"
