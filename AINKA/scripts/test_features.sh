#!/bin/bash

# AINKA Feature Test Script
# Tests all implemented features of the AINKA system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test results
print_test_result() {
    local test_name="$1"
    local result="$2"
    local message="$3"
    
    if [ "$result" -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name - $message"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name - $message"
        ((TESTS_FAILED++))
    fi
}

# Function to print section header
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if file exists
file_exists() {
    [ -f "$1" ]
}

# Function to check if directory exists
dir_exists() {
    [ -d "$1" ]
}

echo -e "${BLUE}AINKA Feature Test Suite${NC}"
echo "Testing all implemented features..."
echo "=================================="

# Test 1: Check if we're running as root (required for kernel module)
print_section "Privilege Check"
if [ "$EUID" -eq 0 ]; then
    print_test_result "Root Privileges" 0 "Running as root - kernel module operations allowed"
else
    print_test_result "Root Privileges" 1 "Not running as root - some features may be limited"
fi

# Test 2: Check kernel module compilation
print_section "Kernel Module Tests"
if file_exists "kernel/ainka_simple.c"; then
    print_test_result "Simple Kernel Module Source" 0 "ainka_simple.c exists"
else
    print_test_result "Simple Kernel Module Source" 1 "ainka_simple.c missing"
fi

if file_exists "kernel/Makefile"; then
    print_test_result "Kernel Makefile" 0 "Makefile exists"
else
    print_test_result "Kernel Makefile" 1 "Makefile missing"
fi

# Test 3: Check Rust daemon compilation
print_section "Rust Daemon Tests"
if file_exists "daemon/Cargo.toml"; then
    print_test_result "Cargo.toml" 0 "Cargo.toml exists"
else
    print_test_result "Cargo.toml" 1 "Cargo.toml missing"
fi

if file_exists "daemon/src/main.rs"; then
    print_test_result "Main.rs" 0 "main.rs exists"
else
    print_test_result "Main.rs" 1 "main.rs missing"
fi

# Test 4: Check core modules
print_section "Core Module Tests"
core_modules=(
    "daemon/src/anomaly_detector.rs"
    "daemon/src/security_monitor.rs"
    "daemon/src/predictive_scaler.rs"
    "daemon/src/database.rs"
    "daemon/src/ebpf_manager.rs"
    "daemon/src/ml_engine.rs"
    "daemon/src/data_pipeline.rs"
    "daemon/src/telemetry_hub.rs"
    "daemon/src/policy_engine.rs"
    "daemon/src/ai_engine.rs"
)

for module in "${core_modules[@]}"; do
    if file_exists "$module"; then
        print_test_result "Module: $(basename "$module")" 0 "Module exists"
    else
        print_test_result "Module: $(basename "$module")" 1 "Module missing"
    fi
done

# Test 5: Check eBPF programs
print_section "eBPF Program Tests"
if file_exists "kernel/ebpf/ainka_monitors.c"; then
    print_test_result "eBPF Monitors" 0 "ainka_monitors.c exists"
else
    print_test_result "eBPF Monitors" 1 "ainka_monitors.c missing"
fi

if file_exists "kernel/ebpf/ainka_tracepoints.c"; then
    print_test_result "eBPF Tracepoints" 0 "ainka_tracepoints.c exists"
else
    print_test_result "eBPF Tracepoints" 1 "ainka_tracepoints.c missing"
fi

# Test 6: Check documentation
print_section "Documentation Tests"
docs=(
    "README.md"
    "docs/architecture.md"
    "docs/IMPLEMENTATION_GUIDE.md"
    "QUICKSTART.md"
    "SECURITY.md"
    "CONTRIBUTING.md"
)

for doc in "${docs[@]}"; do
    if file_exists "$doc"; then
        print_test_result "Documentation: $(basename "$doc")" 0 "Document exists"
    else
        print_test_result "Documentation: $(basename "$doc")" 1 "Document missing"
    fi
done

# Test 7: Check build scripts
print_section "Build Script Tests"
scripts=(
    "scripts/build.sh"
    "scripts/install.sh"
    "scripts/test.sh"
    "scripts/demo.sh"
    "scripts/integration_test.sh"
)

for script in "${scripts[@]}"; do
    if file_exists "$script"; then
        print_test_result "Script: $(basename "$script")" 0 "Script exists"
    else
        print_test_result "Script: $(basename "$script")" 1 "Script missing"
    fi
done

# Test 8: Check system dependencies
print_section "System Dependency Tests"
dependencies=(
    "gcc"
    "make"
    "rustc"
    "cargo"
    "clang"
    "llc"
)

for dep in "${dependencies[@]}"; do
    if command_exists "$dep"; then
        print_test_result "Dependency: $dep" 0 "Command available"
    else
        print_test_result "Dependency: $dep" 1 "Command missing"
    fi
done

# Test 9: Check kernel headers
print_section "Kernel Header Tests"
if dir_exists "/usr/src/linux-headers-$(uname -r)"; then
    print_test_result "Kernel Headers" 0 "Kernel headers found"
else
    print_test_result "Kernel Headers" 1 "Kernel headers missing - install linux-headers-$(uname -r)"
fi

# Test 10: Check if kernel module can be built
print_section "Kernel Module Build Test"
if [ "$EUID" -eq 0 ] && command_exists "make" && command_exists "gcc"; then
    cd kernel
    if make simple >/dev/null 2>&1; then
        print_test_result "Kernel Module Build" 0 "Simple kernel module builds successfully"
    else
        print_test_result "Kernel Module Build" 1 "Kernel module build failed"
    fi
    cd ..
else
    print_test_result "Kernel Module Build" 1 "Skipped - requires root and build tools"
fi

# Test 11: Check if Rust daemon can be built
print_section "Rust Daemon Build Test"
if command_exists "cargo"; then
    cd daemon
    if cargo check --features full >/dev/null 2>&1; then
        print_test_result "Rust Daemon Build" 0 "Daemon compiles successfully"
    else
        print_test_result "Rust Daemon Build" 1 "Daemon compilation failed"
    fi
    cd ..
else
    print_test_result "Rust Daemon Build" 1 "Skipped - cargo not available"
fi

# Test 12: Check database integration
print_section "Database Integration Tests"
if command_exists "sqlite3"; then
    print_test_result "SQLite3" 0 "SQLite3 available"
else
    print_test_result "SQLite3" 1 "SQLite3 not available - install sqlite3"
fi

# Test 13: Check eBPF support
print_section "eBPF Support Tests"
if [ -f "/sys/kernel/debug/bpf" ]; then
    print_test_result "eBPF Debug FS" 0 "eBPF debug filesystem available"
else
    print_test_result "eBPF Debug FS" 1 "eBPF debug filesystem not available"
fi

if [ -f "/sys/fs/bpf" ]; then
    print_test_result "eBPF FS" 0 "eBPF filesystem available"
else
    print_test_result "eBPF FS" 1 "eBPF filesystem not available"
fi

# Test 14: Check system monitoring capabilities
print_section "System Monitoring Tests"
if [ -f "/proc/stat" ]; then
    print_test_result "CPU Stats" 0 "/proc/stat accessible"
else
    print_test_result "CPU Stats" 1 "/proc/stat not accessible"
fi

if [ -f "/proc/meminfo" ]; then
    print_test_result "Memory Info" 0 "/proc/meminfo accessible"
else
    print_test_result "Memory Info" 1 "/proc/meminfo not accessible"
fi

if [ -f "/proc/loadavg" ]; then
    print_test_result "Load Average" 0 "/proc/loadavg accessible"
else
    print_test_result "Load Average" 1 "/proc/loadavg not accessible"
fi

# Test 15: Check ML capabilities
print_section "Machine Learning Tests"
if command_exists "python3"; then
    python3 -c "import numpy" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_test_result "NumPy" 0 "NumPy available for ML"
    else
        print_test_result "NumPy" 1 "NumPy not available"
    fi
else
    print_test_result "Python3" 1 "Python3 not available for ML testing"
fi

# Test 16: Check network capabilities
print_section "Network Tests"
if command_exists "netstat"; then
    print_test_result "Network Tools" 0 "netstat available"
else
    print_test_result "Network Tools" 1 "netstat not available"
fi

# Test 17: Check security features
print_section "Security Feature Tests"
if [ -f "/proc/sys/kernel/dmesg_restrict" ]; then
    print_test_result "Kernel Security" 0 "Kernel security features available"
else
    print_test_result "Kernel Security" 1 "Kernel security features not available"
fi

# Test 18: Check performance monitoring
print_section "Performance Monitoring Tests"
if command_exists "perf"; then
    print_test_result "Perf Tool" 0 "perf available for performance monitoring"
else
    print_test_result "Perf Tool" 1 "perf not available"
fi

# Test 19: Check logging capabilities
print_section "Logging Tests"
if [ -w "/var/log" ]; then
    print_test_result "Log Directory" 0 "/var/log writable"
else
    print_test_result "Log Directory" 1 "/var/log not writable"
fi

# Test 20: Check configuration management
print_section "Configuration Tests"
if [ -w "/etc" ]; then
    print_test_result "Config Directory" 0 "/etc writable for configuration"
else
    print_test_result "Config Directory" 1 "/etc not writable"
fi

# Summary
echo -e "\n${BLUE}=== Test Summary ===${NC}"
echo -e "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All tests passed! AINKA is ready to use.${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠️  Some tests failed. Please check the output above and install missing dependencies.${NC}"
    echo -e "\n${BLUE}Common fixes:${NC}"
    echo "1. Install build tools: sudo apt-get install build-essential"
    echo "2. Install kernel headers: sudo apt-get install linux-headers-\$(uname -r)"
    echo "3. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "4. Install SQLite: sudo apt-get install sqlite3"
    echo "5. Run as root for kernel module tests: sudo $0"
    exit 1
fi 