#!/bin/bash

# AINKA Installation Script
# 
# This script installs AINKA on any Linux system with minimal dependencies.
# It provides a simple, user-friendly installation process.
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
INSTALL_DIR="/usr/local/bin"
CONFIG_DIR="/etc/ainka"
DATA_DIR="/var/lib/ainka"
LOG_DIR="/var/log/ainka"
SERVICE_NAME="ainka.service"

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

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    else
        echo "unknown"
    fi
}

# Function to install dependencies based on distribution
install_dependencies() {
    local distro=$(detect_distro)
    
    print_status "Detected distribution: $distro"
    print_status "Installing dependencies..."
    
    case $distro in
        "ubuntu"|"debian"|"linuxmint")
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                linux-headers-$(uname -r) \
                rustc \
                cargo \
                clang \
                llvm \
                bpftool \
                libbpf-dev \
                pkg-config \
                libssl-dev \
                curl \
                wget
            ;;
        "fedora"|"rhel"|"centos"|"rocky")
            sudo dnf install -y \
                gcc \
                gcc-c++ \
                make \
                kernel-devel \
                rust \
                cargo \
                clang \
                llvm \
                bpftool \
                libbpf-devel \
                pkgconfig \
                openssl-devel \
                curl \
                wget
            ;;
        "arch"|"manjaro")
            sudo pacman -S --noconfirm \
                base-devel \
                linux-headers \
                rust \
                cargo \
                clang \
                llvm \
                bpftool \
                libbpf \
                pkg-config \
                openssl \
                curl \
                wget
            ;;
        *)
            print_warning "Unknown distribution: $distro"
            print_status "Please install the following packages manually:"
            print_status "  - build-essential / gcc / base-devel"
            print_status "  - linux-headers-$(uname -r) / kernel-devel"
            print_status "  - rustc and cargo"
            print_status "  - clang and llvm"
            print_status "  - bpftool and libbpf-dev"
            print_status "  - pkg-config and libssl-dev"
            ;;
    esac
    
    print_success "Dependencies installed"
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root"
        print_status "The script will use sudo when needed"
        exit 1
    fi
}

# Function to create directories
create_directories() {
    print_status "Creating directories..."
    
    sudo mkdir -p "$INSTALL_DIR"
    sudo mkdir -p "$CONFIG_DIR"
    sudo mkdir -p "$DATA_DIR"
    sudo mkdir -p "$LOG_DIR"
    
    # Set permissions
    sudo chown -R $USER:$USER "$CONFIG_DIR"
    sudo chown -R $USER:$USER "$DATA_DIR"
    sudo chown -R $USER:$USER "$LOG_DIR"
    
    print_success "Directories created"
}

# Function to build AINKA
build_ainka() {
    print_status "Building AINKA..."
    
    # Check if we're in the AINKA directory
    if [ ! -f "Cargo.toml" ] && [ ! -f "daemon/Cargo.toml" ]; then
        print_error "Please run this script from the AINKA directory"
        exit 1
    fi
    
    # Build the daemon
    if [ -f "daemon/Cargo.toml" ]; then
        print_status "Building AINKA daemon..."
        cd daemon
        cargo build --release
        cd ..
    fi
    
    # Build kernel module if available
    if [ -f "kernel/Makefile" ]; then
        print_status "Building kernel module..."
        cd kernel
        make clean
        make
        cd ..
    fi
    
    print_success "AINKA built successfully"
}

# Function to install AINKA
install_ainka() {
    print_status "Installing AINKA..."
    
    # Install daemon
    if [ -f "daemon/target/release/ainka-daemon" ]; then
        sudo cp "daemon/target/release/ainka-daemon" "$INSTALL_DIR/"
        sudo chmod +x "$INSTALL_DIR/ainka-daemon"
        print_success "AINKA daemon installed"
    fi
    
    # Install kernel module if available
    if [ -f "kernel/ainka_enhanced.ko" ]; then
        sudo cp "kernel/ainka_enhanced.ko" "/lib/modules/$(uname -r)/kernel/drivers/"
        sudo depmod -a
        print_success "Kernel module installed"
    fi
    
    # Create configuration file
    if [ ! -f "$CONFIG_DIR/config.toml" ]; then
        cat > "$CONFIG_DIR/config.toml" << 'EOF'
# AINKA Configuration File

[daemon]
optimization_interval = 300
max_cycles = 1000
daemon_mode = true
interactive_mode = false
data_dir = "/var/lib/ainka"
log_dir = "/var/log/ainka"
pid_file = "/var/run/ainka.pid"

[monitoring]
cpu_interval = 5
memory_interval = 10
disk_interval = 15
network_interval = 10
process_interval = 30
detailed_monitoring = false
max_processes = 100
cpu_threshold = 80.0
memory_threshold = 85.0
disk_threshold = 90.0

[optimization]
enable_cpu_optimization = true
enable_memory_optimization = true
enable_io_optimization = true
enable_network_optimization = true
aggressive_mode = false
max_optimizations_per_cycle = 5
optimization_timeout = 30
enable_service_restart = false
enable_config_drift_detection = true

[ml]
enable_ml = true
learning_rate = 0.01
feature_count = 10
model_update_interval = 60
min_training_points = 100
max_model_age = 3600
enable_feature_importance = true
enable_anomaly_detection = true
anomaly_threshold = 2.0
model_dir = "/var/lib/ainka/models"

[logging]
level = "info"
format = "text"
enable_file_logging = true
enable_console_logging = true
log_file = "/var/log/ainka/ainka.log"
max_log_size = 100
log_rotation_count = 5
enable_structured_logging = false
EOF
        print_success "Configuration file created"
    fi
    
    # Create systemd service
    cat > "/tmp/$SERVICE_NAME" << EOF
[Unit]
Description=AINKA Intelligent Linux System Optimizer
After=network.target

[Service]
Type=simple
User=root
ExecStart=$INSTALL_DIR/ainka-daemon start --daemon
Restart=always
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv "/tmp/$SERVICE_NAME" "/etc/systemd/system/"
    sudo systemctl daemon-reload
    
    print_success "Systemd service created"
}

# Function to create user-friendly scripts
create_scripts() {
    print_status "Creating user-friendly scripts..."
    
    # Create ainka command
    cat > "/tmp/ainka" << 'EOF'
#!/bin/bash

# AINKA Command Line Interface
# Simple wrapper for the AINKA daemon

AINKA_DAEMON="/usr/local/bin/ainka-daemon"

if [ ! -f "$AINKA_DAEMON" ]; then
    echo "Error: AINKA daemon not found. Please install AINKA first."
    exit 1
fi

case "$1" in
    "start")
        echo "Starting AINKA..."
        sudo systemctl start ainka.service
        ;;
    "stop")
        echo "Stopping AINKA..."
        sudo systemctl stop ainka.service
        ;;
    "restart")
        echo "Restarting AINKA..."
        sudo systemctl restart ainka.service
        ;;
    "status")
        echo "AINKA Status:"
        sudo systemctl status ainka.service
        ;;
    "optimize")
        echo "Running immediate optimization..."
        sudo $AINKA_DAEMON optimize
        ;;
    "monitor")
        echo "Monitoring system..."
        sudo $AINKA_DAEMON monitor
        ;;
    "analyze")
        echo "Analyzing system..."
        sudo $AINKA_DAEMON analyze
        ;;
    "config")
        echo "Opening configuration..."
        sudo $AINKA_DAEMON config --show
        ;;
    "logs")
        echo "Showing logs..."
        sudo journalctl -u ainka.service -f
        ;;
    "help"|"--help"|"-h"|"")
        echo "AINKA - Intelligent Linux System Optimizer"
        echo ""
        echo "Usage: ainka <command>"
        echo ""
        echo "Commands:"
        echo "  start     - Start AINKA daemon"
        echo "  stop      - Stop AINKA daemon"
        echo "  restart   - Restart AINKA daemon"
        echo "  status    - Show AINKA status"
        echo "  optimize  - Run immediate optimization"
        echo "  monitor   - Monitor system performance"
        echo "  analyze   - Analyze system and show recommendations"
        echo "  config    - Show configuration"
        echo "  logs      - Show AINKA logs"
        echo "  help      - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ainka start      # Start AINKA"
        echo "  ainka optimize   # Optimize system immediately"
        echo "  ainka status     # Check AINKA status"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run 'ainka help' for usage information"
        exit 1
        ;;
esac
EOF
    
    sudo mv "/tmp/ainka" "$INSTALL_DIR/"
    sudo chmod +x "$INSTALL_DIR/ainka"
    
    print_success "User-friendly scripts created"
}

# Function to enable and start service
enable_service() {
    print_status "Enabling AINKA service..."
    
    sudo systemctl enable ainka.service
    
    print_success "AINKA service enabled"
    print_status "You can now use the following commands:"
    print_status "  ainka start    - Start AINKA"
    print_status "  ainka status   - Check status"
    print_status "  ainka optimize - Run optimization"
    print_status "  ainka help     - Show all commands"
}

# Function to show post-installation information
show_post_install_info() {
    echo ""
    print_success "AINKA v$AINKA_VERSION installed successfully!"
    echo ""
    echo "What's next?"
    echo "============="
    echo "1. Start AINKA:"
    echo "   ainka start"
    echo ""
    echo "2. Check status:"
    echo "   ainka status"
    echo ""
    echo "3. Run immediate optimization:"
    echo "   ainka optimize"
    echo ""
    echo "4. Monitor system:"
    echo "   ainka monitor"
    echo ""
    echo "5. View logs:"
    echo "   ainka logs"
    echo ""
    echo "6. Get help:"
    echo "   ainka help"
    echo ""
    echo "Configuration:"
    echo "=============="
    echo "Config file: $CONFIG_DIR/config.toml"
    echo "Data directory: $DATA_DIR"
    echo "Log directory: $LOG_DIR"
    echo ""
    echo "Service management:"
    echo "=================="
    echo "Start:   sudo systemctl start ainka"
    echo "Stop:    sudo systemctl stop ainka"
    echo "Restart: sudo systemctl restart ainka"
    echo "Status:  sudo systemctl status ainka"
    echo ""
    echo "For more information, visit: https://github.com/ainka-community/ainka"
    echo ""
}

# Main installation function
main() {
    echo "AINKA Intelligent Linux System Optimizer v$AINKA_VERSION"
    echo "========================================================"
    echo ""
    
    # Check if not running as root
    check_root
    
    # Install dependencies
    install_dependencies
    
    # Create directories
    create_directories
    
    # Build AINKA
    build_ainka
    
    # Install AINKA
    install_ainka
    
    # Create user-friendly scripts
    create_scripts
    
    # Enable service
    enable_service
    
    # Show post-installation information
    show_post_install_info
}

# Run main function
main "$@" 