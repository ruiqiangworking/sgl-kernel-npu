#!/bin/bash
set -e

# Configuration options
DEBUG_MODE="OFF"

# Parse command line arguments
while getopts ":dh" opt; do
    case ${opt} in
        d )
            DEBUG_MODE="ON"
            echo "Debug mode enabled"
            ;;
        h )
            echo "Usage: ./build.sh [options]"
            echo "Options:"
            echo "  -d    Enable debug mode"
            echo "  -h    Show this help message"
            echo "Default SOC_VERSION: Ascend910_93"
            exit 0
            ;;
        \? )
            echo "Error: unknown flag: -$OPTARG" 1>&2
            echo "Run './build.sh -h' for more information."
            exit 1
            ;;
    esac
done

shift $((OPTIND -1))

export DEBUG_MODE=$DEBUG_MODE

# SOC Version (can be overridden with first argument)
SOC_VERSION="${1:-Ascend910_9382}"

# Setup environment paths
if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi

if [ -n "$ASCEND_INCLUDE_DIR" ]; then
    ASCEND_INCLUDE_DIR=$ASCEND_INCLUDE_DIR
else
    ASCEND_INCLUDE_DIR=${_ASCEND_INSTALL_PATH}/aarch64-linux/include
fi

if [ -n "$SHMEM_HOME_PATH" ]; then
    _SHMEM_HOME_PATH=$SHMEM_HOME_PATH
else
    _SHMEM_HOME_PATH=/usr/local/Ascend/shmem/latest
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
export SHMEM_HOME_PATH=${_SHMEM_HOME_PATH}

echo "========================================="
echo "DeepEP Standalone Build Script"
echo "========================================="
echo "Ascend path: ${ASCEND_HOME_PATH}"
echo "SOC Version: ${SOC_VERSION}"
echo "Build mode: $([ "$DEBUG_MODE" == "ON" ] && echo "Debug" || echo "Release")"
echo "========================================="

# Source Ascend environment
if [ -f "$(dirname ${ASCEND_HOME_PATH})/set_env.sh" ]; then
    source $(dirname ${ASCEND_HOME_PATH})/set_env.sh
else
    echo "Warning: Ascend environment script not found"
fi

# Get current directory
CURRENT_DIR=$(pwd)
VERSION="1.0.0"
OUTPUT_DIR=$CURRENT_DIR/output
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/lib

echo "Output path: ${OUTPUT_DIR}"

# Compile options
COMPILE_OPTIONS=""
if [ "$DEBUG_MODE" == "ON" ]; then
    COMPILE_OPTIONS="-DCMAKE_BUILD_TYPE=Debug"
else
    COMPILE_OPTIONS="-DCMAKE_BUILD_TYPE=Release"
fi

# Function: Build DeepEP Adapter (C++ extension)
function build_deepep_adapter()
{
    echo ""
    echo "========================================="
    echo "Building DeepEP Adapter Module"
    echo "========================================="
    
    BUILD_DIR="$CURRENT_DIR/build"
    mkdir -p "$BUILD_DIR"
    
    cd "$CURRENT_DIR" || exit
    
    # Configure CMake
    echo "Configuring CMake..."
    cmake $COMPILE_OPTIONS \
          -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
          -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
          -DASCEND_INCLUDE_DIR="$ASCEND_INCLUDE_DIR" \
          -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH" \
          -DSOC_VERSION="$SOC_VERSION" \
          -B "$BUILD_DIR" \
          -S .
    
    # Build
    echo "Building adapter module..."
    cmake --build "$BUILD_DIR" -j8
    
    # Install to output directory
    echo "Installing to output directory..."
    cmake --build "$BUILD_DIR" --target install
    
    cd - > /dev/null
    echo "DeepEP adapter build completed successfully!"
}

# Function: Build DeepEP Kernels
function build_deepep_kernels()
{
    echo ""
    echo "========================================="
    echo "Building DeepEP Kernels"
    echo "========================================="
    
    KERNEL_DIR="csrc/deepep/ops"
    CUSTOM_OPP_DIR="${CURRENT_DIR}/python/deep_ep/deep_ep"

    cd "$KERNEL_DIR" || exit

    # Make build script executable
    chmod +x build.sh
    if [ -f "cmake/util/gen_ops_filter.sh" ]; then
        chmod +x cmake/util/gen_ops_filter.sh
    fi
    
    # Build kernels
    echo "Running kernel build script..."
    ./build.sh

    # Find the generated run package
    custom_opp_file=$(find ./build_out -maxdepth 1 -type f -name "custom_opp*.run" 2>/dev/null | head -n 1)
    
    if [ -z "$custom_opp_file" ]; then
        echo "Error: Cannot find run package in ./build_out"
        exit 1
    else
        echo "Found run package: $custom_opp_file"
        chmod +x "$custom_opp_file"
    fi
    
    # Install custom operators
    echo "Installing custom operators to: $CUSTOM_OPP_DIR"
    rm -rf "$CUSTOM_OPP_DIR"/vendors
    "$custom_opp_file" --install-path="$CUSTOM_OPP_DIR"
    
    # Copy built libraries to output directory
    if [ -d "./build_out" ]; then
        echo "Copying libraries to output directory..."
        find ./build_out -name "*.so" -exec cp -v {} "$OUTPUT_DIR/lib/" \;
    fi
    
    cd - > /dev/null
    echo "DeepEP kernels build completed successfully!"
}

# Function: Make DeepEP Package
function make_deepep_package()
{
    echo ""
    echo "========================================="
    echo "Building DeepEP Python Package"
    echo "========================================="
    
    cd python/deep_ep || exit

    # Copy libraries from output directory
    if [ -d "${OUTPUT_DIR}/lib" ] && [ "$(ls -A ${OUTPUT_DIR}/lib)" ]; then
        echo "Copying libraries from ${OUTPUT_DIR}/lib to deep_ep/"
        cp -v ${OUTPUT_DIR}/lib/* "$CURRENT_DIR"/python/deep_ep/deep_ep/ 2>/dev/null || true
    fi

    # Clean previous builds
    echo "Cleaning previous builds..."
    rm -rf "$CURRENT_DIR"/python/deep_ep/build
    rm -rf "adapter
    build_deepep_adapter
    
    # Build $CURRENT_DIR"/python/deep_ep/dist
    rm -rf "$CURRENT_DIR"/python/deep_ep/deep_ep.egg-info
    
    # Build wheel package
    echo "Building wheel package..."
    python3 setup.py clean --all
    python3 setup.py bdist_wheel
    
    # Move wheel to output directory
    if [ -d "$CURRENT_DIR"/python/deep_ep/dist ]; then
        echo "Moving wheel package to output directory..."
        mv -v "$CURRENT_DIR"/python/deep_ep/dist/deep_ep*.whl ${OUTPUT_DIR}/
        rm -rf "$CURRENT_DIR"/python/deep_ep/dist
    fi
    
    cd - > /dev/null
    echo "DeepEP package build completed successfully!"
}

# Main function
function main()
{
    echo ""
    echo "Starting DeepEP standalone build process..."
    echo ""
    
    # Check if wheel is installed
    if pip3 show wheel > /dev/null 2>&1; then
        echo "Python wheel package is already installed"
    else
        echo "Installing Python wheel package..."
        pip3 install wheel==0.45.1
    fi
    
    # Build adapter
    build_deepep_adapter
    
    # Build kernels
    build_deepep_kernels
    
    # Build package
    make_deepep_package
    
    echo ""
    echo "========================================="
    echo "Build completed successfully!"
    echo "========================================="
    echo "Output directory: ${OUTPUT_DIR}"
    echo ""
    echo "Generated files:"
    ls -lh ${OUTPUT_DIR}/*.whl 2>/dev/null || echo "No wheel files found"
    echo ""
    echo "To install the package, run:"
    echo "  pip3 install ${OUTPUT_DIR}/deep_ep*.whl"
    echo ""
}

# Run main function
main
