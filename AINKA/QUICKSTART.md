# AINKA Quick Start Guide

This guide will help you get AINKA up and running on Ubuntu in under 10 minutes.

## Prerequisites

- Ubuntu 20.04 LTS or later
- sudo privileges
- Internet connection

## Step 1: Install Dependencies

```bash
# Update package list
sudo apt update

# Install build dependencies
sudo apt install -y build-essential linux-headers-$(uname -r) rustc cargo git

# Install additional tools (optional but recommended)
sudo apt install -y clang-format rustfmt cargo-audit
```

## Step 2: Clone and Build

```bash
# Clone the repository
git clone https://github.com/yourusername/ainka.git
cd ainka

# Build everything
./scripts/build.sh
```

## Step 3: Install and Test

```bash
# Load the kernel module
sudo insmod kernel/ainka_lkm.ko

# Test the CLI
./cli/target/release/ainka-cli status

# Start the daemon (in background)
./daemon/target/release/ainka-daemon &
```

## Step 4: Verify Installation

```bash
# Check if kernel module is loaded
lsmod | grep ainka

# Check /proc interface
cat /proc/ainka

# Test CLI commands
./cli/target/release/ainka-cli info
./cli/target/release/ainka-cli metrics --interval 2 --samples 5
```

## Step 5: Basic Usage

### Check System Status
```bash
./cli/target/release/ainka-cli status
```

### Monitor System Metrics
```bash
./cli/target/release/ainka-cli metrics --interval 1 --samples 10
```

### Send Commands to Kernel
```bash
./cli/target/release/ainka-cli send "ping"
./cli/target/release/ainka-cli send "status"
```

### View Kernel Logs
```bash
./cli/target/release/ainka-cli logs --lines 20
```

### Get AI Suggestions
```bash
./cli/target/release/ainka-cli suggestions
```

## Troubleshooting

### Kernel Module Issues

**Problem**: Module fails to load
```bash
# Check kernel logs
dmesg | tail -20

# Verify kernel headers
ls /lib/modules/$(uname -r)/build

# Rebuild module
cd kernel && make clean && make
```

**Problem**: /proc/ainka not found
```bash
# Check if module is loaded
lsmod | grep ainka

# Check /proc filesystem
ls -la /proc/ | grep ainka
```

### Daemon Issues

**Problem**: Daemon fails to start
```bash
# Check daemon logs
./daemon/target/release/ainka-daemon --foreground

# Verify kernel module is loaded
lsmod | grep ainka
```

**Problem**: Permission denied
```bash
# Check file permissions
ls -la /proc/ainka

# Fix permissions if needed
sudo chmod 666 /proc/ainka
```

### CLI Issues

**Problem**: CLI commands fail
```bash
# Check if binaries exist
ls -la cli/target/release/ainka-cli

# Rebuild CLI
cd cli && cargo build --release
```

## Advanced Configuration

### Daemon Configuration

Create a configuration file:
```bash
mkdir -p ~/.config/ainka
cat > ~/.config/ainka/config.toml << EOF
kernel_path = "/proc/ainka"
log_level = "info"
monitor_interval = 5
enable_ai = true
enable_monitoring = true
EOF
```

### Systemd Service (Optional)

Create a systemd service for the daemon:
```bash
sudo tee /etc/systemd/system/ainka-daemon.service << EOF
[Unit]
Description=AINKA AI Daemon
After=network.target

[Service]
Type=simple
User=root
ExecStart=/path/to/ainka/daemon/target/release/ainka-daemon
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable ainka-daemon
sudo systemctl start ainka-daemon
```

## Performance Tuning

### Kernel Module Tuning
```bash
# Increase buffer size (requires module rebuild)
# Edit kernel/ainka_lkm.c and change AINKA_BUFFER_SIZE

# Monitor module performance
dmesg | grep AINKA
```

### Daemon Tuning
```bash
# Adjust monitoring interval
./daemon/target/release/ainka-daemon --interval 10

# Enable verbose logging
RUST_LOG=debug ./daemon/target/release/ainka-daemon
```

## Security Considerations

### File Permissions
```bash
# Secure /proc interface
sudo chmod 600 /proc/ainka
sudo chown root:root /proc/ainka
```

### Network Security
```bash
# If using network features, configure firewall
sudo ufw allow from 127.0.0.1
```

## Uninstallation

```bash
# Stop daemon
pkill ainka-daemon

# Unload kernel module
sudo rmmod ainka_lkm

# Remove binaries
sudo rm -f /usr/local/bin/ainka-*

# Remove configuration
rm -rf ~/.config/ainka
```

## Next Steps

1. **Read the Documentation**: Check out the `docs/` directory
2. **Join the Community**: Visit our GitHub discussions
3. **Contribute**: See `CONTRIBUTING.md` for guidelines
4. **Report Issues**: Use GitHub issues for bugs and feature requests

## Support

- **Documentation**: `docs/` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Security**: See `SECURITY.md`

---

**Congratulations!** You now have AINKA running on your system. The AI assistant is monitoring your system and ready to provide intelligent suggestions for optimization and troubleshooting. 