#!/bin/bash
# Compare CUDA kernels between two nsys profiles

if [ $# -ne 2 ]; then
    echo "Usage: $0 <profile1.nsys-rep> <profile2.nsys-rep>"
    echo ""
    echo "Example:"
    echo "  $0 nsys_batch_16_profile.nsys-rep nsys_batch_32_profile.nsys-rep"
    exit 1
fi

PROFILE1="$1"
PROFILE2="$2"

if [ ! -f "$PROFILE1" ]; then
    echo "ERROR: Profile 1 not found: $PROFILE1"
    exit 1
fi

if [ ! -f "$PROFILE2" ]; then
    echo "ERROR: Profile 2 not found: $PROFILE2"
    exit 1
fi

echo "========================================================================"
echo "Comparing CUDA kernels between two nsys profiles"
echo "========================================================================"
echo "Profile 1: $PROFILE1"
echo "Profile 2: $PROFILE2"
echo ""

# Export kernel lists
echo "Extracting kernel names from Profile 1..."
nsys stats --report cuda_gpu_kern_sum --format csv "$PROFILE1" 2>/dev/null | \
    grep -v "^#" | grep -v "^Time" | cut -d',' -f8 | sort | uniq > /tmp/kernels1.txt

echo "Extracting kernel names from Profile 2..."
nsys stats --report cuda_gpu_kern_sum --format csv "$PROFILE2" 2>/dev/null | \
    grep -v "^#" | grep -v "^Time" | cut -d',' -f8 | sort | uniq > /tmp/kernels2.txt

# Compare
echo ""
echo "========================================================================"
echo "Kernels in BOTH profiles (same kernels used):"
echo "========================================================================"
comm -12 /tmp/kernels1.txt /tmp/kernels2.txt | nl

echo ""
echo "========================================================================"
echo "Kernels ONLY in Profile 1:"
echo "========================================================================"
ONLY1=$(comm -23 /tmp/kernels1.txt /tmp/kernels2.txt)
if [ -z "$ONLY1" ]; then
    echo "(none)"
else
    echo "$ONLY1" | nl
fi

echo ""
echo "========================================================================"
echo "Kernels ONLY in Profile 2:"
echo "========================================================================"
ONLY2=$(comm -13 /tmp/kernels1.txt /tmp/kernels2.txt)
if [ -z "$ONLY2" ]; then
    echo "(none)"
else
    echo "$ONLY2" | nl
fi

echo ""
echo "========================================================================"
echo "Full kernel statistics for Profile 1:"
echo "========================================================================"
nsys stats --report cuda_gpu_kern_sum "$PROFILE1"

echo ""
echo "========================================================================"
echo "Full kernel statistics for Profile 2:"
echo "========================================================================"
nsys stats --report cuda_gpu_kern_sum "$PROFILE2"

# Cleanup
rm -f /tmp/kernels1.txt /tmp/kernels2.txt

echo ""
echo "========================================================================"
echo "Comparison complete!"
echo "========================================================================"
echo ""
echo "To view in GUI:"
echo "  nsys-ui $PROFILE1 &"
echo "  nsys-ui $PROFILE2 &"
