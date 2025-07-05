# AINKA Kernel Module Makefile
#
# This Makefile builds the AINKA kernel modules for Linux
# Supports both enhanced core module and original LKM
#
# Copyright (C) 2024 AINKA Community
# Licensed under GPLv3

# Module names
CORE_MODULE_NAME = ainka_core
LKM_MODULE_NAME = ainka_lkm
ENHANCED_MODULE_NAME = ainka_enhanced
SIMPLE_MODULE_NAME = ainka_simple

# Source files
obj-m += $(CORE_MODULE_NAME).o
obj-m += $(LKM_MODULE_NAME).o
obj-m += $(ENHANCED_MODULE_NAME).o
obj-m += $(SIMPLE_MODULE_NAME).o

# Kernel source directory (can be overridden)
KERNEL_SRC ?= /lib/modules/$(shell uname -r)/build

# Build directory
BUILD_DIR = $(PWD)

# Compiler flags
ccflags-y += -Wall -Wextra -Werror
ccflags-y += -DDEBUG
ccflags-y += -DAINKA_ENHANCED

# eBPF compiler
CLANG ?= clang
LLC ?= llc

# eBPF source files
EBPF_SOURCES = ebpf/ainka_tracepoints.c
EBPF_OBJECTS = $(EBPF_SOURCES:.c=.o)

# Default target - build all modules
all: simple core enhanced lkm ebpf

# Build simple module only (recommended)
simple: $(SIMPLE_MODULE_NAME).ko

# Build core module only
core: $(CORE_MODULE_NAME).ko

# Build enhanced module only
enhanced: $(ENHANCED_MODULE_NAME).ko

# Build original LKM only
lkm: $(LKM_MODULE_NAME).ko

# Build eBPF programs
ebpf: $(EBPF_OBJECTS)

# Build the core kernel module
$(CORE_MODULE_NAME).ko: $(CORE_MODULE_NAME).c
	$(MAKE) -C $(KERNEL_SRC) M=$(BUILD_DIR) $(CORE_MODULE_NAME).ko

# Build the enhanced kernel module
$(ENHANCED_MODULE_NAME).ko: $(ENHANCED_MODULE_NAME).c
	$(MAKE) -C $(KERNEL_SRC) M=$(BUILD_DIR) $(ENHANCED_MODULE_NAME).ko

# Build the original LKM
$(LKM_MODULE_NAME).ko: $(LKM_MODULE_NAME).c
	$(MAKE) -C $(KERNEL_SRC) M=$(BUILD_DIR) $(LKM_MODULE_NAME).ko

# Build the simple kernel module
$(SIMPLE_MODULE_NAME).ko: $(SIMPLE_MODULE_NAME).c
	$(MAKE) -C $(KERNEL_SRC) M=$(BUILD_DIR) $(SIMPLE_MODULE_NAME).ko

# Build eBPF programs
%.o: %.c
	$(CLANG) -target bpf -c $< -o $@ \
		-I$(KERNEL_SRC)/arch/x86/include \
		-I$(KERNEL_SRC)/arch/x86/include/generated \
		-I$(KERNEL_SRC)/include \
		-I$(KERNEL_SRC)/arch/x86/include/uapi \
		-I$(KERNEL_SRC)/include/uapi \
		-I$(KERNEL_SRC)/include/generated/uapi \
		-I$(KERNEL_SRC)/tools/include \
		-I$(KERNEL_SRC)/tools/include/uapi \
		-I$(KERNEL_SRC)/tools/testing/selftests/bpf \
		-fno-stack-protector \
		-O2 \
		-g

# Clean build artifacts
clean:
	$(MAKE) -C $(KERNEL_SRC) M=$(BUILD_DIR) clean
	rm -f *.ko *.mod.c *.mod.o *.o modules.order Module.symvers
	rm -f $(EBPF_OBJECTS)
	rm -f ebpf/*.o

# Install modules
install: install-simple

# Install simple module (recommended)
install-simple: $(SIMPLE_MODULE_NAME).ko
	sudo insmod $(SIMPLE_MODULE_NAME).ko

# Install core module
install-core: $(CORE_MODULE_NAME).ko
	sudo insmod $(CORE_MODULE_NAME).ko

# Install enhanced module
install-enhanced: $(ENHANCED_MODULE_NAME).ko
	sudo insmod $(ENHANCED_MODULE_NAME).ko

# Install original LKM
install-lkm: $(LKM_MODULE_NAME).ko
	sudo insmod $(LKM_MODULE_NAME).ko

# Remove modules
uninstall: uninstall-all

# Remove all modules
uninstall-all:
	-sudo rmmod $(CORE_MODULE_NAME) 2>/dev/null || true
	-sudo rmmod $(ENHANCED_MODULE_NAME) 2>/dev/null || true
	-sudo rmmod $(LKM_MODULE_NAME) 2>/dev/null || true

# Remove specific modules
uninstall-core:
	-sudo rmmod $(CORE_MODULE_NAME) 2>/dev/null || true

uninstall-enhanced:
	-sudo rmmod $(ENHANCED_MODULE_NAME) 2>/dev/null || true

uninstall-lkm:
	-sudo rmmod $(LKM_MODULE_NAME) 2>/dev/null || true

# Load modules
load: install

# Load specific modules
load-core: install-core
load-enhanced: install-enhanced
load-lkm: install-lkm

# Unload modules
unload: uninstall

# Unload specific modules
unload-core: uninstall-core
unload-enhanced: uninstall-enhanced
unload-lkm: uninstall-lkm

# Check if modules are loaded
status:
	@echo "AINKA Kernel Modules Status:"
	@echo "============================"
	@if lsmod | grep -q $(CORE_MODULE_NAME); then \
		echo "✓ $(CORE_MODULE_NAME) is loaded"; \
		lsmod | grep $(CORE_MODULE_NAME); \
	else \
		echo "✗ $(CORE_MODULE_NAME) is not loaded"; \
	fi
	@if lsmod | grep -q $(ENHANCED_MODULE_NAME); then \
		echo "✓ $(ENHANCED_MODULE_NAME) is loaded"; \
		lsmod | grep $(ENHANCED_MODULE_NAME); \
	else \
		echo "✗ $(ENHANCED_MODULE_NAME) is not loaded"; \
	fi
	@if lsmod | grep -q $(LKM_MODULE_NAME); then \
		echo "✓ $(LKM_MODULE_NAME) is loaded"; \
		lsmod | grep $(LKM_MODULE_NAME); \
	else \
		echo "✗ $(LKM_MODULE_NAME) is not loaded"; \
	fi

# Show module information
info: $(CORE_MODULE_NAME).ko $(ENHANCED_MODULE_NAME).ko $(LKM_MODULE_NAME).ko
	@echo "Core Module Information:"
	@modinfo $(CORE_MODULE_NAME).ko
	@echo ""
	@echo "Enhanced Module Information:"
	@modinfo $(ENHANCED_MODULE_NAME).ko
	@echo ""
	@echo "LKM Module Information:"
	@modinfo $(LKM_MODULE_NAME).ko

# Test the modules
test: test-core test-enhanced test-lkm

# Test core module
test-core: install-core
	@echo "Testing AINKA Core Module..."
	@echo "Checking if /proc/ainka exists..."
	@if [ -f /proc/ainka ]; then \
		echo "✓ /proc/ainka interface created successfully"; \
		echo "Reading from /proc/ainka:"; \
		cat /proc/ainka; \
		echo ""; \
		echo "✓ Core module test passed"; \
	else \
		echo "✗ /proc/ainka interface not found"; \
		exit 1; \
	fi

# Test enhanced module
test-enhanced: install-enhanced
	@echo "Testing AINKA Enhanced Module..."
	@echo "Checking enhanced functionality..."
	@if [ -f /proc/ainka ]; then \
		echo "✓ Enhanced module loaded successfully"; \
		echo "Reading enhanced status:"; \
		cat /proc/ainka; \
		echo ""; \
		echo "✓ Enhanced module test passed"; \
	else \
		echo "✗ Enhanced module not working properly"; \
		exit 1; \
	fi

# Test original LKM
test-lkm: install-lkm
	@echo "Testing AINKA LKM..."
	@echo "Checking LKM functionality..."
	@if [ -f /proc/ainka ]; then \
		echo "✓ LKM loaded successfully"; \
		echo "Reading LKM status:"; \
		cat /proc/ainka; \
		echo ""; \
		echo "✓ LKM test passed"; \
	else \
		echo "✗ LKM not working properly"; \
		exit 1; \
	fi

# Check kernel logs
logs:
	@echo "AINKA Kernel Module Logs:"
	@echo "========================="
	dmesg | grep AINKA

# Format code (requires clang-format)
format:
	clang-format -i *.c ebpf/*.c

# Static analysis (requires sparse)
sparse:
	sparse *.c

# Security audit
audit:
	@echo "Running security audit..."
	@echo "Checking for common kernel module security issues..."
	@if grep -q "copy_from_user\|copy_to_user" *.c; then \
		echo "✓ User space interaction functions found"; \
	else \
		echo "⚠ No user space interaction functions found"; \
	fi
	@if grep -q "EXPORT_SYMBOL" *.c; then \
		echo "✓ Exported symbols found"; \
	else \
		echo "⚠ No exported symbols found"; \
	fi

# Performance profiling
profile:
	@echo "AINKA Performance Profile:"
	@echo "=========================="
	@if [ -f /proc/ainka ]; then \
		echo "Module loaded, checking performance metrics..."; \
		cat /proc/ainka | grep -E "(time|optimization|decision)"; \
	else \
		echo "Module not loaded, cannot profile"; \
	fi

# Development setup
dev-setup:
	@echo "Setting up AINKA development environment..."
	@echo "Installing required packages..."
	@sudo apt-get update
	@sudo apt-get install -y build-essential linux-headers-$(shell uname -r) \
		clang llvm libbpf-dev libelf-dev zlib1g-dev
	@echo "✓ Development environment setup complete"

# Documentation generation
docs:
	@echo "Generating AINKA documentation..."
	@echo "Kernel Module Documentation:" > docs/kernel_api.md
	@echo "===========================" >> docs/kernel_api.md
	@echo "" >> docs/kernel_api.md
	@echo "## Exported Functions" >> docs/kernel_api.md
	@echo "" >> docs/kernel_api.md
	@grep -n "EXPORT_SYMBOL" *.c | while read line; do \
		echo "- \`$$line\`" >> docs/kernel_api.md; \
	done
	@echo "✓ Documentation generated in docs/kernel_api.md"

# Help target
help:
	@echo "AINKA Kernel Module Makefile"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Build all modules and eBPF programs (default)"
	@echo "  core        - Build core module only"
	@echo "  enhanced    - Build enhanced module only"
	@echo "  lkm         - Build original LKM only"
	@echo "  ebpf        - Build eBPF programs only"
	@echo "  clean       - Clean build artifacts"
	@echo "  install     - Install core module"
	@echo "  install-core, install-enhanced, install-lkm - Install specific modules"
	@echo "  uninstall   - Remove all modules"
	@echo "  uninstall-core, uninstall-enhanced, uninstall-lkm - Remove specific modules"
	@echo "  load        - Load core module (alias for install)"
	@echo "  load-core, load-enhanced, load-lkm - Load specific modules"
	@echo "  unload      - Unload all modules (alias for uninstall)"
	@echo "  unload-core, unload-enhanced, unload-lkm - Unload specific modules"
	@echo "  status      - Check if modules are loaded"
	@echo "  info        - Show module information"
	@echo "  test        - Test all modules"
	@echo "  test-core, test-enhanced, test-lkm - Test specific modules"
	@echo "  logs        - Show kernel logs for AINKA"
	@echo "  format      - Format source code"
	@echo "  sparse      - Run static analysis"
	@echo "  audit       - Run security audit"
	@echo "  profile     - Performance profiling"
	@echo "  dev-setup   - Setup development environment"
	@echo "  docs        - Generate documentation"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  KERNEL_SRC  - Kernel source directory (default: /lib/modules/\$(shell uname -r)/build)"
	@echo "  CLANG       - Clang compiler for eBPF (default: clang)"
	@echo "  LLC         - LLVM linker for eBPF (default: llc)"

# Phony targets
.PHONY: all core enhanced lkm ebpf clean install install-core install-enhanced install-lkm \
        uninstall uninstall-all uninstall-core uninstall-enhanced uninstall-lkm \
        load load-core load-enhanced load-lkm unload unload-core unload-enhanced unload-lkm \
        status info test test-core test-enhanced test-lkm logs format sparse audit profile \
        dev-setup docs help 