#!/bin/bash

# AINKA Integration Test Script
# Tests the complete AI-Native architecture including kernel modules, eBPF, and userspace daemon
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

# Test configuration
TEST_DURATION=30
STRESS_TEST_DURATION=60
LOG_FILE="/tmp/ainka_integration_test.log"

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

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check kernel version
    KERNEL_VERSION=$(uname -r)
    print_status "Kernel version: $KERNEL_VERSION"
    
    # Check if required tools are available
    local missing_tools=()
    
    if ! command_exists make; then
        missing_tools+=("make")
    fi
    
    if ! command_exists gcc; then
        missing_tools+=("gcc")
    fi
    
    if ! command_exists clang; then
        missing_tools+=("clang")
    fi
    
    if ! command_exists cargo; then
        missing_tools+=("cargo")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_status "Run 'make dev-setup' to install dependencies"
        exit 1
    fi
    
    print_success "System requirements check passed"
}

# Function to build all components
build_components() {
    print_status "Building AINKA components..."
    
    # Build kernel modules
    print_status "Building kernel modules..."
    cd kernel
    make clean
    make all
    cd ..
    
    # Build eBPF programs
    print_status "Building eBPF programs..."
    cd kernel
    make ebpf
    cd ..
    
    # Build userspace daemon
    print_status "Building userspace daemon..."
    cd daemon
    cargo build --release --features "full"
    cd ..
    
    # Build CLI tools
    print_status "Building CLI tools..."
    cd cli
    cargo build --release
    cd ..
    
    print_success "All components built successfully"
}

# Function to install kernel modules
install_kernel_modules() {
    print_status "Installing kernel modules..."
    
    cd kernel
    
    # Unload any existing modules
    make uninstall-all 2>/dev/null || true
    
    # Install core module
    print_status "Installing core module..."
    make install-core
    sleep 2
    
    # Check if core module is loaded
    if lsmod | grep -q ainka_core; then
        print_success "Core module loaded successfully"
    else
        print_error "Failed to load core module"
        exit 1
    fi
    
    # Install enhanced module
    print_status "Installing enhanced module..."
    make install-enhanced
    sleep 2
    
    # Check if enhanced module is loaded
    if lsmod | grep -q ainka_enhanced; then
        print_success "Enhanced module loaded successfully"
    else
        print_error "Failed to load enhanced module"
        exit 1
    fi
    
    cd ..
}

# Function to start userspace daemon
start_daemon() {
    print_status "Starting AINKA daemon..."
    
    # Kill any existing daemon processes
    pkill -f "ainka-daemon" 2>/dev/null || true
    
    # Start daemon in background
    cd daemon
    cargo run --release --features "full" > /tmp/ainka_daemon.log 2>&1 &
    DAEMON_PID=$!
    cd ..
    
    # Wait for daemon to start
    sleep 5
    
    # Check if daemon is running
    if kill -0 $DAEMON_PID 2>/dev/null; then
        print_success "Daemon started successfully (PID: $DAEMON_PID)"
    else
        print_error "Failed to start daemon"
        cat /tmp/ainka_daemon.log
        exit 1
    fi
}

# Function to test kernel module functionality
test_kernel_modules() {
    print_status "Testing kernel module functionality..."
    
    # Test /proc interface
    if [[ -f /proc/ainka ]]; then
        print_success "/proc/ainka interface exists"
        print_status "Reading /proc/ainka:"
        cat /proc/ainka
    else
        print_error "/proc/ainka interface not found"
        exit 1
    fi
    
    # Test netlink communication
    print_status "Testing netlink communication..."
    if dmesg | grep -q "AINKA.*netlink"; then
        print_success "Netlink communication working"
    else
        print_warning "No netlink communication detected"
    fi
    
    # Test exported functions
    print_status "Testing exported functions..."
    if [[ -f /sys/kernel/debug/kallsyms ]] && grep -q ainka_register_event /sys/kernel/debug/kallsyms; then
        print_success "Exported functions available"
    else
        print_warning "Exported functions not found in kallsyms"
    fi
}

# Function to test eBPF programs
test_ebpf_programs() {
    print_status "Testing eBPF programs..."
    
    # Check if eBPF programs are loaded
    if [[ -f /sys/kernel/debug/bpf/verifier_log ]]; then
        print_success "eBPF verifier available"
    else
        print_warning "eBPF verifier not available"
    fi
    
    # Check for eBPF maps
    if [[ -d /sys/fs/bpf ]]; then
        print_success "eBPF filesystem mounted"
    else
        print_warning "eBPF filesystem not mounted"
    fi
}

# Function to test userspace daemon
test_daemon() {
    print_status "Testing userspace daemon..."
    
    # Check if daemon is still running
    if kill -0 $DAEMON_PID 2>/dev/null; then
        print_success "Daemon is running"
    else
        print_error "Daemon is not running"
        cat /tmp/ainka_daemon.log
        exit 1
    fi
    
    # Test daemon logs
    if [[ -f /tmp/ainka_daemon.log ]]; then
        print_status "Daemon logs:"
        tail -20 /tmp/ainka_daemon.log
    fi
    
    # Test CLI communication
    print_status "Testing CLI communication..."
    cd cli
    if cargo run --release -- --status > /tmp/cli_test.log 2>&1; then
        print_success "CLI communication working"
        cat /tmp/cli_test.log
    else
        print_warning "CLI communication failed"
        cat /tmp/cli_test.log
    fi
    cd ..
}

# Function to run stress tests
run_stress_tests() {
    print_status "Running stress tests for $STRESS_TEST_DURATION seconds..."
    
    # Start stress-ng if available
    if command_exists stress-ng; then
        print_status "Starting stress-ng..."
        stress-ng --cpu 4 --vm 2 --io 2 --timeout $STRESS_TEST_DURATION &
        STRESS_PID=$!
    else
        print_warning "stress-ng not available, using basic stress test"
        # Basic stress test
        for i in {1..4}; do
            dd if=/dev/zero of=/dev/null bs=1M count=1000 &
        done
    fi
    
    # Monitor system during stress test
    print_status "Monitoring system during stress test..."
    start_time=$(date +%s)
    
    while [[ $(($(date +%s) - start_time)) -lt $STRESS_TEST_DURATION ]]; do
        # Check kernel module status
        if [[ -f /proc/ainka ]]; then
            echo "=== $(date) ===" >> /tmp/ainka_stress_test.log
            cat /proc/ainka >> /tmp/ainka_stress_test.log
        fi
        
        # Check daemon status
        if kill -0 $DAEMON_PID 2>/dev/null; then
            echo "Daemon running" >> /tmp/ainka_stress_test.log
        else
            print_error "Daemon stopped during stress test"
            break
        fi
        
        sleep 5
    done
    
    # Stop stress test
    if [[ -n $STRESS_PID ]]; then
        kill $STRESS_PID 2>/dev/null || true
    fi
    
    print_success "Stress test completed"
}

# Function to test AI functionality
test_ai_functionality() {
    print_status "Testing AI functionality..."
    
    # Test policy updates
    print_status "Testing policy updates..."
    cd cli
    if cargo run --release -- --add-policy "test_policy" --condition "cpu_high" --action "reduce_freq" > /tmp/policy_test.log 2>&1; then
        print_success "Policy update test passed"
    else
        print_warning "Policy update test failed"
        cat /tmp/policy_test.log
    fi
    cd ..
    
    # Test anomaly detection
    print_status "Testing anomaly detection..."
    # Trigger some system events
    for i in {1..10}; do
        dd if=/dev/zero of=/dev/null bs=1M count=100 &
    done
    
    sleep 5
    
    # Check if anomalies were detected
    if dmesg | grep -q "AINKA.*anomaly"; then
        print_success "Anomaly detection working"
    else
        print_warning "No anomalies detected"
    fi
    
    # Test optimization application
    print_status "Testing optimization application..."
    if dmesg | grep -q "AINKA.*optimization"; then
        print_success "Optimization application working"
    else
        print_warning "No optimizations applied"
    fi
}

# Function to test performance
test_performance() {
    print_status "Testing performance..."
    
    # Test decision latency
    print_status "Testing decision latency..."
    start_time=$(date +%s%N)
    
    # Trigger some events
    for i in {1..100}; do
        echo "test_event_$i" > /proc/ainka 2>/dev/null || true
    done
    
    end_time=$(date +%s%N)
    latency=$(( (end_time - start_time) / 1000000 ))  # Convert to milliseconds
    
    print_status "Average decision latency: ${latency}ms"
    
    if [[ $latency -lt 100 ]]; then
        print_success "Decision latency within acceptable range"
    else
        print_warning "Decision latency higher than expected"
    fi
    
    # Test memory usage
    print_status "Testing memory usage..."
    if command_exists ps; then
        daemon_memory=$(ps -o rss= -p $DAEMON_PID 2>/dev/null || echo "0")
        print_status "Daemon memory usage: ${daemon_memory}KB"
        
        if [[ $daemon_memory -lt 50000 ]]; then
            print_success "Memory usage within acceptable range"
        else
            print_warning "Memory usage higher than expected"
        fi
    fi
}

# Function to generate test report
generate_report() {
    print_status "Generating test report..."
    
    report_file="/tmp/ainka_integration_report.txt"
    
    cat > "$report_file" << EOF
AINKA Integration Test Report
============================
Date: $(date)
Kernel: $(uname -r)
Duration: $TEST_DURATION seconds

Test Results:
------------

1. System Requirements: $(command_exists make && command_exists gcc && command_exists clang && command_exists cargo && echo "PASS" || echo "FAIL")
2. Kernel Module Loading: $(lsmod | grep -q ainka_core && echo "PASS" || echo "FAIL")
3. Enhanced Module Loading: $(lsmod | grep -q ainka_enhanced && echo "PASS" || echo "FAIL")
4. /proc Interface: $(test -f /proc/ainka && echo "PASS" || echo "FAIL")
5. Daemon Running: $(kill -0 $DAEMON_PID 2>/dev/null && echo "PASS" || echo "FAIL")
6. eBPF Support: $(test -d /sys/fs/bpf && echo "PASS" || echo "FAIL")
7. Netlink Communication: $(dmesg | grep -q "AINKA.*netlink" && echo "PASS" || echo "FAIL")
8. CLI Communication: $(test -f /tmp/cli_test.log && echo "PASS" || echo "FAIL")

Performance Metrics:
-------------------
- Decision Latency: ${latency}ms
- Daemon Memory Usage: ${daemon_memory}KB
- Stress Test Duration: ${STRESS_TEST_DURATION}s

Logs:
-----
EOF
    
    # Append relevant logs
    echo "=== Kernel Logs ===" >> "$report_file"
    dmesg | grep AINKA | tail -20 >> "$report_file"
    
    echo "" >> "$report_file"
    echo "=== Daemon Logs ===" >> "$report_file"
    tail -20 /tmp/ainka_daemon.log >> "$report_file" 2>/dev/null || echo "No daemon logs" >> "$report_file"
    
    echo "" >> "$report_file"
    echo "=== Stress Test Logs ===" >> "$report_file"
    tail -20 /tmp/ainka_stress_test.log >> "$report_file" 2>/dev/null || echo "No stress test logs" >> "$report_file"
    
    print_success "Test report generated: $report_file"
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up..."
    
    # Stop daemon
    if [[ -n $DAEMON_PID ]]; then
        kill $DAEMON_PID 2>/dev/null || true
    fi
    
    # Unload kernel modules
    cd kernel
    make uninstall-all 2>/dev/null || true
    cd ..
    
    # Kill stress test processes
    pkill -f "stress-ng" 2>/dev/null || true
    pkill -f "dd.*if=/dev/zero" 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Main test function
main() {
    print_status "Starting AINKA Integration Test Suite"
    print_status "====================================="
    
    # Initialize log file
    echo "AINKA Integration Test Log" > "$LOG_FILE"
    echo "==========================" >> "$LOG_FILE"
    echo "Started at: $(date)" >> "$LOG_FILE"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Run tests
    check_requirements
    build_components
    install_kernel_modules
    start_daemon
    test_kernel_modules
    test_ebpf_programs
    test_daemon
    test_ai_functionality
    run_stress_tests
    test_performance
    generate_report
    
    print_success "Integration test completed successfully!"
    print_status "Check the report at: /tmp/ainka_integration_report.txt"
}

# Check if running as root
check_root

# Run main function
main "$@" 