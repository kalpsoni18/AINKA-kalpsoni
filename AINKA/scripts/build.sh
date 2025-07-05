#!/bin/bash

# AINKA Enhanced Build Script
# 
# This script builds the complete AI-Native Linux Kernel Assistant
# including the enhanced kernel module, eBPF programs, AI daemon,
# and CLI tools.
# 
# Copyright (C) 2024 AINKA Community
# Licensed under GPLv3

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AINKA_VERSION="0.2.0"
BUILD_DIR="build"
INSTALL_DIR="/opt/ainka"
KERNEL_MODULE_NAME="ainka_enhanced"
EBPF_DIR="kernel/ebpf"
DAEMON_DIR="daemon"
CLI_DIR="cli"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking build dependencies..."
    
    # Check for required packages
    local missing_packages=()
    
    # Kernel development packages
    if ! command -v make &> /dev/null; then
        missing_packages+=("make")
    fi
    
    if ! command -v gcc &> /dev/null; then
        missing_packages+=("gcc")
    fi
    
    # Rust toolchain
    if ! command -v cargo &> /dev/null; then
        missing_packages+=("rust")
    fi
    
    # eBPF tools
    if ! command -v bpftool &> /dev/null; then
        missing_packages+=("bpftool")
    fi
    
    if ! command -v clang &> /dev/null; then
        missing_packages+=("clang")
    fi
    
    if ! command -v llvm-strip &> /dev/null; then
        missing_packages+=("llvm")
    fi
    
    # System utilities
    if ! command -v sudo &> /dev/null; then
        missing_packages+=("sudo")
    fi
    
    if ! command -v systemctl &> /dev/null; then
        missing_packages+=("systemd")
    fi
    
    # Report missing packages
    if [ ${#missing_packages[@]} -ne 0 ]; then
        print_error "Missing required packages: ${missing_packages[*]}"
        print_status "Please install the missing packages:"
        print_status "sudo apt-get update && sudo apt-get install -y ${missing_packages[*]}"
        exit 1
    fi
    
    print_success "All dependencies satisfied"
}

# Function to check kernel headers
check_kernel_headers() {
    print_status "Checking kernel headers..."
    
    if [ ! -d "/usr/src/linux-headers-$(uname -r)" ]; then
        print_error "Kernel headers not found for current kernel: $(uname -r)"
        print_status "Please install kernel headers:"
        print_status "sudo apt-get install -y linux-headers-$(uname -r)"
        exit 1
    fi
    
    print_success "Kernel headers found"
}

# Function to build eBPF programs
build_ebpf() {
    print_status "Building eBPF programs..."
    
    if [ ! -d "$EBPF_DIR" ]; then
        print_error "eBPF directory not found: $EBPF_DIR"
        exit 1
    fi
    
    # Create build directory for eBPF
    mkdir -p "$BUILD_DIR/ebpf"
    
    # Build eBPF tracepoint programs
    if [ -f "$EBPF_DIR/ainka_tracepoints.c" ]; then
        print_status "Building tracepoint programs..."
        
        # Compile eBPF programs
        clang -O2 -g -Wall -target bpf -c "$EBPF_DIR/ainka_tracepoints.c" \
            -o "$BUILD_DIR/ebpf/ainka_tracepoints.o"
        
        if [ $? -eq 0 ]; then
            print_success "eBPF tracepoint programs built successfully"
        else
            print_error "Failed to build eBPF tracepoint programs"
            exit 1
        fi
    else
        print_warning "eBPF tracepoint source not found, skipping"
    fi
    
    # Generate skeleton files (if using libbpf)
    if command -v bpftool &> /dev/null; then
        print_status "Generating eBPF skeleton files..."
        
        for obj_file in "$BUILD_DIR/ebpf"/*.o; do
            if [ -f "$obj_file" ]; then
                local base_name=$(basename "$obj_file" .o)
                bpftool gen skeleton "$obj_file" > "$BUILD_DIR/ebpf/${base_name}.skel.h" 2>/dev/null || true
            fi
        done
    fi
    
    print_success "eBPF programs built successfully"
}

# Function to build kernel module
build_kernel_module() {
    print_status "Building enhanced kernel module..."
    
    if [ ! -f "kernel/Makefile" ]; then
        print_error "Kernel module Makefile not found"
        exit 1
    fi
    
    # Create build directory for kernel module
    mkdir -p "$BUILD_DIR/kernel"
    
    # Copy kernel module source to build directory
    cp -r kernel/* "$BUILD_DIR/kernel/"
    
    # Build kernel module
    cd "$BUILD_DIR/kernel"
    
    # Check if enhanced module exists, otherwise use basic module
    if [ -f "ainka_enhanced.c" ]; then
        print_status "Building enhanced kernel module..."
        make -f Makefile KERNEL_MODULE=ainka_enhanced
    else
        print_status "Building basic kernel module..."
        make -f Makefile
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Kernel module built successfully"
    else
        print_error "Failed to build kernel module"
        exit 1
    fi
    
    cd - > /dev/null
}

# Function to build Rust daemon
build_daemon() {
    print_status "Building enhanced AI daemon..."
    
    if [ ! -f "$DAEMON_DIR/Cargo.toml" ]; then
        print_error "Daemon Cargo.toml not found"
        exit 1
    fi
    
    # Create build directory for daemon
    mkdir -p "$BUILD_DIR/daemon"
    
    # Build daemon
    cd "$DAEMON_DIR"
    
    # Check if enhanced components exist
    if [ -f "src/ai_engine.rs" ] && [ -f "src/policy_engine.rs" ] && [ -f "src/telemetry_hub.rs" ] && [ -f "src/ipc_layer.rs" ]; then
        print_status "Building enhanced AI daemon with ML engine..."
        cargo build --release --features "enhanced"
    else
        print_status "Building basic daemon..."
        cargo build --release
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Daemon built successfully"
        cp target/release/ainka-daemon "../$BUILD_DIR/daemon/"
    else
        print_error "Failed to build daemon"
        exit 1
    fi
    
    cd - > /dev/null
}

# Function to build CLI tool
build_cli() {
    print_status "Building CLI tool..."
    
    if [ ! -f "$CLI_DIR/Cargo.toml" ]; then
        print_error "CLI Cargo.toml not found"
        exit 1
    fi
    
    # Create build directory for CLI
    mkdir -p "$BUILD_DIR/cli"
    
    # Build CLI
    cd "$CLI_DIR"
    cargo build --release
    
    if [ $? -eq 0 ]; then
        print_success "CLI tool built successfully"
        cp target/release/ainka-cli "../$BUILD_DIR/cli/"
    else
        print_error "Failed to build CLI tool"
        exit 1
    fi
    
    cd - > /dev/null
}

# Function to create installation package
create_package() {
    print_status "Creating installation package..."
    
    # Create package directory
    local package_dir="$BUILD_DIR/ainka-package"
    mkdir -p "$package_dir"
    
    # Copy built components
    cp -r "$BUILD_DIR/kernel" "$package_dir/"
    cp -r "$BUILD_DIR/ebpf" "$package_dir/"
    cp -r "$BUILD_DIR/daemon" "$package_dir/"
    cp -r "$BUILD_DIR/cli" "$package_dir/"
    
    # Copy documentation
    cp -r docs "$package_dir/" 2>/dev/null || true
    cp README.md "$package_dir/" 2>/dev/null || true
    cp QUICKSTART.md "$package_dir/" 2>/dev/null || true
    cp LICENSE "$package_dir/" 2>/dev/null || true
    
    # Create installation script
    cat > "$package_dir/install.sh" << 'EOF'
#!/bin/bash

# AINKA Installation Script
# Copyright (C) 2024 AINKA Community

set -e

INSTALL_DIR="/opt/ainka"
SERVICE_DIR="/etc/systemd/system"

echo "Installing AINKA Enhanced AI-Native Kernel Assistant..."

# Create installation directory
sudo mkdir -p "$INSTALL_DIR"

# Copy files
sudo cp -r * "$INSTALL_DIR/"

# Set permissions
sudo chmod +x "$INSTALL_DIR/daemon/ainka-daemon"
sudo chmod +x "$INSTALL_DIR/cli/ainka-cli"

# Create systemd service
sudo tee "$SERVICE_DIR/ainka-daemon.service" > /dev/null << 'SERVICE_EOF'
[Unit]
Description=AINKA Enhanced AI-Native Kernel Assistant
After=network.target

[Service]
Type=simple
User=root
ExecStart=/opt/ainka/daemon/ainka-daemon
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable ainka-daemon.service

echo "AINKA installation completed successfully!"
echo "To start the service: sudo systemctl start ainka-daemon"
echo "To check status: sudo systemctl status ainka-daemon"
EOF

    chmod +x "$package_dir/install.sh"
    
    # Create uninstall script
    cat > "$package_dir/uninstall.sh" << 'EOF'
#!/bin/bash

# AINKA Uninstallation Script
# Copyright (C) 2024 AINKA Community

set -e

INSTALL_DIR="/opt/ainka"
SERVICE_DIR="/etc/systemd/system"

echo "Uninstalling AINKA Enhanced AI-Native Kernel Assistant..."

# Stop and disable service
sudo systemctl stop ainka-daemon.service 2>/dev/null || true
sudo systemctl disable ainka-daemon.service 2>/dev/null || true

# Remove service file
sudo rm -f "$SERVICE_DIR/ainka-daemon.service"

# Reload systemd
sudo systemctl daemon-reload

# Remove installation directory
sudo rm -rf "$INSTALL_DIR"

echo "AINKA uninstallation completed successfully!"
EOF

    chmod +x "$package_dir/uninstall.sh"
    
    # Create tarball
    cd "$BUILD_DIR"
    tar -czf "ainka-enhanced-${AINKA_VERSION}.tar.gz" ainka-package/
    
    print_success "Installation package created: $BUILD_DIR/ainka-enhanced-${AINKA_VERSION}.tar.gz"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Test kernel module
    if [ -f "$BUILD_DIR/kernel/ainka_enhanced.ko" ]; then
        print_status "Testing kernel module..."
        # Load module for testing
        sudo insmod "$BUILD_DIR/kernel/ainka_enhanced.ko" 2>/dev/null || true
        
        # Check if module loaded
        if lsmod | grep -q "ainka_enhanced"; then
            print_success "Kernel module test passed"
            # Unload module
            sudo rmmod ainka_enhanced 2>/dev/null || true
        else
            print_warning "Kernel module test failed"
        fi
    fi
    
    # Test daemon
    if [ -f "$BUILD_DIR/daemon/ainka-daemon" ]; then
        print_status "Testing daemon..."
        # Test daemon version
        if "$BUILD_DIR/daemon/ainka-daemon" --version &> /dev/null; then
            print_success "Daemon test passed"
        else
            print_warning "Daemon test failed"
        fi
    fi
    
    # Test CLI
    if [ -f "$BUILD_DIR/cli/ainka-cli" ]; then
        print_status "Testing CLI..."
        # Test CLI version
        if "$BUILD_DIR/cli/ainka-cli" --version &> /dev/null; then
            print_success "CLI test passed"
        else
            print_warning "CLI test failed"
        fi
    fi
}

# Function to clean build artifacts
clean_build() {
    print_status "Cleaning build artifacts..."
    
    # Remove build directory
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        print_success "Build artifacts cleaned"
    fi
    
    # Clean Rust artifacts
    if [ -d "$DAEMON_DIR/target" ]; then
        cd "$DAEMON_DIR"
        cargo clean
        cd - > /dev/null
    fi
    
    if [ -d "$CLI_DIR/target" ]; then
        cd "$CLI_DIR"
        cargo clean
        cd - > /dev/null
    fi
}

# Function to show help
show_help() {
    echo "AINKA Enhanced Build Script v${AINKA_VERSION}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --clean         Clean build artifacts"
    echo "  -t, --test          Run tests after building"
    echo "  -p, --package       Create installation package"
    echo "  -e, --ebpf-only     Build only eBPF programs"
    echo "  -k, --kernel-only   Build only kernel module"
    echo "  -d, --daemon-only   Build only daemon"
    echo "  -a, --all           Build everything (default)"
    echo ""
    echo "Examples:"
    echo "  $0                  Build everything"
    echo "  $0 --clean          Clean build artifacts"
    echo "  $0 --test           Build and test"
    echo "  $0 --package        Build and create package"
    echo "  $0 --ebpf-only      Build only eBPF programs"
}

# Main build function
main_build() {
    print_status "Starting AINKA Enhanced build process..."
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    
    # Build components
    build_ebpf
    build_kernel_module
    build_daemon
    build_cli
    
    print_success "Build completed successfully!"
}

# Parse command line arguments
BUILD_ALL=true
BUILD_EBPF_ONLY=false
BUILD_KERNEL_ONLY=false
BUILD_DAEMON_ONLY=false
RUN_TESTS=false
CREATE_PACKAGE=false
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        -p|--package)
            CREATE_PACKAGE=true
            shift
            ;;
        -e|--ebpf-only)
            BUILD_ALL=false
            BUILD_EBPF_ONLY=true
            shift
            ;;
        -k|--kernel-only)
            BUILD_ALL=false
            BUILD_KERNEL_ONLY=true
            shift
            ;;
        -d|--daemon-only)
            BUILD_ALL=false
            BUILD_DAEMON_ONLY=true
            shift
            ;;
        -a|--all)
            BUILD_ALL=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
if [ "$CLEAN_BUILD" = true ]; then
    clean_build
    exit 0
fi

# Check dependencies
check_dependencies
check_kernel_headers

# Build based on options
if [ "$BUILD_ALL" = true ]; then
    main_build
elif [ "$BUILD_EBPF_ONLY" = true ]; then
    build_ebpf
elif [ "$BUILD_KERNEL_ONLY" = true ]; then
    build_kernel_module
elif [ "$BUILD_DAEMON_ONLY" = true ]; then
    build_daemon
fi

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    run_tests
fi

# Create package if requested
if [ "$CREATE_PACKAGE" = true ]; then
    create_package
fi

print_success "AINKA Enhanced build process completed!"
print_status "Build artifacts are in: $BUILD_DIR" 