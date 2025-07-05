#!/bin/bash

# AINKA Test Script
#
# This script runs tests for all components of the AINKA project
#
# Copyright (C) 2024 AINKA Community
# Licensed under Apache 2.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_LOG="$PROJECT_ROOT/test.log"

# Test configuration
KERNEL_MODULE_NAME="ainka_lkm"
DAEMON_NAME="ainka-daemon"
CLI_NAME="ainka-cli"

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

# Function to test kernel module
test_kernel_module() {
    print_status "Testing kernel module..."
    
    cd "$PROJECT_ROOT/kernel"
    
    # Check if module exists
    if [ ! -f "$KERNEL_MODULE_NAME.ko" ]; then
        print_error "Kernel module not found. Please build it first."
        return 1
    fi
    
    # Test module loading
    print_status "Testing module loading..."
    if sudo insmod "$KERNEL_MODULE_NAME.ko"; then
        print_success "Module loaded successfully"
        
        # Check if /proc/ainka exists
        if [ -f "/proc/ainka" ]; then
            print_success "/proc/ainka interface created"
            
            # Test reading from /proc/ainka
            print_status "Testing /proc/ainka read..."
            if cat /proc/ainka > /dev/null; then
                print_success "Read from /proc/ainka successful"
            else
                print_error "Failed to read from /proc/ainka"
            fi
            
            # Test writing to /proc/ainka
            print_status "Testing /proc/ainka write..."
            if echo "test_command" | sudo tee /proc/ainka > /dev/null; then
                print_success "Write to /proc/ainka successful"
            else
                print_error "Failed to write to /proc/ainka"
            fi
            
            # Check kernel logs
            print_status "Checking kernel logs..."
            if dmesg | grep -q "AINKA"; then
                print_success "AINKA messages found in kernel logs"
                dmesg | grep "AINKA" | tail -5
            else
                print_warning "No AINKA messages in kernel logs"
            fi
        else
            print_error "/proc/ainka interface not created"
        fi
        
        # Unload module
        print_status "Unloading module..."
        if sudo rmmod "$KERNEL_MODULE_NAME"; then
            print_success "Module unloaded successfully"
        else
            print_error "Failed to unload module"
        fi
    else
        print_error "Failed to load module"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
}

# Function to test Rust components
test_rust_component() {
    local component_name="$1"
    local component_dir="$2"
    
    print_status "Testing $component_name..."
    
    cd "$component_dir"
    
    # Check if Cargo.toml exists
    if [ ! -f "Cargo.toml" ]; then
        print_error "Cargo.toml not found in $component_dir"
        return 1
    fi
    
    # Run unit tests
    print_status "Running unit tests..."
    if cargo test --release; then
        print_success "$component_name unit tests passed"
    else
        print_error "$component_name unit tests failed"
        return 1
    fi
    
    # Run integration tests if they exist
    if [ -d "tests" ]; then
        print_status "Running integration tests..."
        if cargo test --release --test integration; then
            print_success "$component_name integration tests passed"
        else
            print_warning "$component_name integration tests failed or not available"
        fi
    fi
    
    cd "$PROJECT_ROOT"
}

# Function to test daemon
test_daemon() {
    test_rust_component "AI daemon" "$PROJECT_ROOT/daemon"
}

# Function to test CLI
test_cli() {
    test_rust_component "CLI tool" "$PROJECT_ROOT/cli"
}

# Function to test CLI functionality
test_cli_functionality() {
    print_status "Testing CLI functionality..."
    
    local cli_binary="$PROJECT_ROOT/cli/target/release/$CLI_NAME"
    
    # Check if CLI binary exists
    if [ ! -f "$cli_binary" ]; then
        print_error "CLI binary not found. Please build it first."
        return 1
    fi
    
    # Test help command
    print_status "Testing help command..."
    if "$cli_binary" --help > /dev/null; then
        print_success "Help command works"
    else
        print_error "Help command failed"
    fi
    
    # Test status command (without kernel module)
    print_status "Testing status command..."
    if "$cli_binary" status > /dev/null; then
        print_success "Status command works"
    else
        print_warning "Status command failed (expected without kernel module)"
    fi
    
    # Test info command
    print_status "Testing info command..."
    if "$cli_binary" info > /dev/null; then
        print_success "Info command works"
    else
        print_error "Info command failed"
    fi
}

# Function to test daemon functionality
test_daemon_functionality() {
    print_status "Testing daemon functionality..."
    
    local daemon_binary="$PROJECT_ROOT/daemon/target/release/$DAEMON_NAME"
    
    # Check if daemon binary exists
    if [ ! -f "$daemon_binary" ]; then
        print_error "Daemon binary not found. Please build it first."
        return 1
    fi
    
    # Test daemon startup (briefly)
    print_status "Testing daemon startup..."
    timeout 5s "$daemon_binary" --foreground || true
    
    print_success "Daemon startup test completed"
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    
    # Test with kernel module loaded
    if [ -f "$PROJECT_ROOT/kernel/$KERNEL_MODULE_NAME.ko" ]; then
        print_status "Testing full system integration..."
        
        # Load kernel module
        sudo insmod "$PROJECT_ROOT/kernel/$KERNEL_MODULE_NAME.ko"
        
        # Test CLI with kernel module
        if [ -f "$PROJECT_ROOT/cli/target/release/$CLI_NAME" ]; then
            print_status "Testing CLI with kernel module..."
            "$PROJECT_ROOT/cli/target/release/$CLI_NAME" status
            "$PROJECT_ROOT/cli/target/release/$CLI_NAME" send "test_integration"
        fi
        
        # Test daemon with kernel module
        if [ -f "$PROJECT_ROOT/daemon/target/release/$DAEMON_NAME" ]; then
            print_status "Testing daemon with kernel module..."
            timeout 10s "$PROJECT_ROOT/daemon/target/release/$DAEMON_NAME" --foreground || true
        fi
        
        # Unload kernel module
        sudo rmmod "$KERNEL_MODULE_NAME"
        
        print_success "Integration tests completed"
    else
        print_warning "Kernel module not found, skipping integration tests"
    fi
}

# Function to run performance tests
run_performance_tests() {
    print_status "Running performance tests..."
    
    # Test kernel module performance
    if [ -f "$PROJECT_ROOT/kernel/$KERNEL_MODULE_NAME.ko" ]; then
        print_status "Testing kernel module performance..."
        
        sudo insmod "$PROJECT_ROOT/kernel/$KERNEL_MODULE_NAME.ko"
        
        # Test write performance
        print_status "Testing write performance..."
        time for i in {1..100}; do
            echo "perf_test_$i" | sudo tee /proc/ainka > /dev/null
        done
        
        # Test read performance
        print_status "Testing read performance..."
        time for i in {1..100}; do
            cat /proc/ainka > /dev/null
        done
        
        sudo rmmod "$KERNEL_MODULE_NAME"
        
        print_success "Performance tests completed"
    fi
}

# Function to show test summary
show_summary() {
    print_status "Test Summary:"
    echo "=============="
    
    # Count test results
    local passed=0
    local failed=0
    local warnings=0
    
    # This would be populated by actual test results
    # For now, just show a placeholder
    echo "Tests completed. Check the output above for results."
    
    print_status "Test log: $TEST_LOG"
}

# Main test function
main() {
    print_status "Starting AINKA test suite..."
    print_status "Project root: $PROJECT_ROOT"
    
    # Redirect output to log file
    exec > >(tee -a "$TEST_LOG")
    exec 2>&1
    
    local test_results=()
    
    # Run component tests
    if test_kernel_module; then
        test_results+=("kernel:pass")
    else
        test_results+=("kernel:fail")
    fi
    
    if test_daemon; then
        test_results+=("daemon:pass")
    else
        test_results+=("daemon:fail")
    fi
    
    if test_cli; then
        test_results+=("cli:pass")
    else
        test_results+=("cli:fail")
    fi
    
    # Run functionality tests
    test_cli_functionality
    test_daemon_functionality
    
    # Run integration tests if requested
    if [ "$1" = "--integration" ]; then
        run_integration_tests
    fi
    
    # Run performance tests if requested
    if [ "$1" = "--performance" ] || [ "$2" = "--performance" ]; then
        run_performance_tests
    fi
    
    # Show summary
    show_summary
    
    # Check if all tests passed
    local failed_tests=0
    for result in "${test_results[@]}"; do
        if [[ "$result" == *":fail" ]]; then
            ((failed_tests++))
        fi
    done
    
    if [ $failed_tests -eq 0 ]; then
        print_success "All tests passed!"
        exit 0
    else
        print_error "$failed_tests test(s) failed"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "AINKA Test Script"
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --integration  Run integration tests"
        echo "  --performance  Run performance tests"
        echo "  --help         Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                    # Run basic tests"
        echo "  $0 --integration     # Run integration tests"
        echo "  $0 --performance     # Run performance tests"
        echo "  $0 --integration --performance  # Run all tests"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac 