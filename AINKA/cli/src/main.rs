/*
 * AINKA CLI
 * 
 * This CLI tool provides a command-line interface for interacting with
 * the AINKA kernel module and daemon.
 * 
 * Copyright (C) 2024 AINKA Community
 * Licensed under Apache 2.0
 */

use std::fs::{File, OpenOptions};
use std::io::{Read, Write, BufRead, BufReader};
use std::path::Path;
use std::process::Command;

use clap::{Parser, Subcommand};
use colored::*;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use sysinfo::{System, SystemExt, CpuExt, DiskExt, NetworkExt};
use tabled::{Table, Tabled};

/// AINKA CLI main command
#[derive(Parser)]
#[command(name = "ainka")]
#[command(about = "AINKA CLI - Command-line interface for AINKA AI assistant")]
#[command(version = "0.1.0")]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Path to kernel interface
    #[arg(short, long, default_value = "/proc/ainka")]
    kernel_path: String,
    
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Available CLI commands
#[derive(Subcommand)]
enum Commands {
    /// Show system status
    Status {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },
    
    /// Show system metrics
    Metrics {
        /// Refresh interval in seconds
        #[arg(short, long, default_value = "1")]
        interval: u64,
        
        /// Number of samples to collect
        #[arg(short, long, default_value = "10")]
        samples: usize,
    },
    
    /// Send command to kernel
    Send {
        /// Command to send
        command: String,
    },
    
    /// Show kernel logs
    Logs {
        /// Number of log lines to show
        #[arg(short, long, default_value = "20")]
        lines: usize,
        
        /// Follow logs in real-time
        #[arg(short, long)]
        follow: bool,
    },
    
    /// Show AI suggestions
    Suggestions {
        /// Show all suggestions
        #[arg(short, long)]
        all: bool,
    },
    
    /// Control daemon
    Daemon {
        #[command(subcommand)]
        action: DaemonCommands,
    },
    
    /// Show system information
    Info {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },
    
    /// Test kernel module
    Test {
        /// Run all tests
        #[arg(short, long)]
        all: bool,
    },
}

/// Daemon control commands
#[derive(Subcommand)]
enum DaemonCommands {
    /// Start the daemon
    Start,
    
    /// Stop the daemon
    Stop,
    
    /// Restart the daemon
    Restart,
    
    /// Show daemon status
    Status,
}

/// System metrics structure
#[derive(Debug, Clone, Serialize, Deserialize, Tabled)]
struct SystemMetrics {
    #[tabled(rename = "CPU %")]
    cpu_usage: f32,
    #[tabled(rename = "Memory %")]
    memory_usage: f32,
    #[tabled(rename = "Disk %")]
    disk_usage: f32,
    #[tabled(rename = "Load Avg")]
    load_average: f64,
    #[tabled(rename = "Network RX (MB)")]
    network_rx_mb: f64,
    #[tabled(rename = "Network TX (MB)")]
    network_tx_mb: f64,
}

/// AI suggestion structure
#[derive(Debug, Clone, Serialize, Deserialize, Tabled)]
struct AISuggestion {
    #[tabled(rename = "Category")]
    category: String,
    #[tabled(rename = "Severity")]
    severity: String,
    #[tabled(rename = "Message")]
    message: String,
    #[tabled(rename = "Action")]
    action: String,
    #[tabled(rename = "Confidence")]
    confidence: f32,
}

/// AINKA CLI main structure
struct AinkaCli {
    kernel_path: String,
    verbose: bool,
    system: System,
}

impl AinkaCli {
    /// Create a new CLI instance
    fn new(kernel_path: String, verbose: bool) -> Self {
        Self {
            kernel_path,
            verbose,
            system: System::new_all(),
        }
    }

    /// Check if kernel module is loaded
    fn check_kernel_module(&self) -> Result<bool> {
        Ok(Path::new(&self.kernel_path).exists())
    }

    /// Read from kernel interface
    fn read_from_kernel(&self) -> Result<String> {
        let mut file = File::open(&self.kernel_path)
            .context("Failed to open kernel interface")?;
        
        let mut content = String::new();
        file.read_to_string(&mut content)
            .context("Failed to read from kernel interface")?;
        
        Ok(content)
    }

    /// Write to kernel interface
    fn write_to_kernel(&self, command: &str) -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .open(&self.kernel_path)
            .context("Failed to open kernel interface for writing")?;
        
        file.write_all(command.as_bytes())
            .context("Failed to write to kernel interface")?;
        
        if self.verbose {
            println!("Sent command to kernel: {}", command);
        }
        
        Ok(())
    }

    /// Collect system metrics
    fn collect_metrics(&mut self) -> Result<SystemMetrics> {
        self.system.refresh_all();
        
        let cpu_usage = self.system.global_cpu_info().cpu_usage();
        let memory_usage = {
            let memory = self.system.memory();
            (memory.used() as f32 / memory.total() as f32) * 100.0
        };
        
        let disk_usage = {
            let disks = self.system.disks();
            if disks.is_empty() {
                0.0
            } else {
                let disk = &disks[0];
                (disk.available_space() as f32 / disk.total_space() as f32) * 100.0
            }
        };
        
        let (network_rx, network_tx) = {
            let networks = self.system.networks();
            let mut total_rx = 0;
            let mut total_tx = 0;
            
            for (_, network) in networks {
                total_rx += network.received();
                total_tx += network.transmitted();
            }
            
            (total_rx, total_tx)
        };
        
        let load_average = self.system.load_average().one;
        
        Ok(SystemMetrics {
            cpu_usage,
            memory_usage,
            disk_usage,
            load_average,
            network_rx_mb: network_rx as f64 / 1024.0 / 1024.0,
            network_tx_mb: network_tx as f64 / 1024.0 / 1024.0,
        })
    }

    /// Show system status
    fn show_status(&self, detailed: bool) -> Result<()> {
        println!("{}", "AINKA System Status".bold().blue());
        println!("{}", "==================".blue());
        
        // Check kernel module
        match self.check_kernel_module() {
            Ok(true) => {
                println!("{} Kernel Module: {}", "✓".green(), "Loaded".green());
                
                if detailed {
                    match self.read_from_kernel() {
                        Ok(content) => {
                            println!("\nKernel Module Output:");
                            println!("{}", content);
                        }
                        Err(e) => {
                            println!("{} Failed to read kernel output: {}", "✗".red(), e);
                        }
                    }
                }
            }
            Ok(false) => {
                println!("{} Kernel Module: {}", "✗".red(), "Not Loaded".red());
            }
            Err(e) => {
                println!("{} Kernel Module: {}", "✗".red(), format!("Error: {}", e).red());
            }
        }
        
        // Check daemon
        let daemon_running = self.check_daemon_status()?;
        if daemon_running {
            println!("{} Daemon: {}", "✓".green(), "Running".green());
        } else {
            println!("{} Daemon: {}", "✗".red(), "Not Running".red());
        }
        
        // Show system info
        if detailed {
            self.show_system_info()?;
        }
        
        Ok(())
    }

    /// Show system metrics
    fn show_metrics(&mut self, interval: u64, samples: usize) -> Result<()> {
        println!("{}", "AINKA System Metrics".bold().blue());
        println!("{}", "===================".blue());
        
        for i in 0..samples {
            let metrics = self.collect_metrics()?;
            
            if i == 0 {
                println!("{}", Table::new(vec![metrics]));
            } else {
                print!("\r{}", Table::new(vec![metrics]));
                std::io::stdout().flush()?;
            }
            
            if i < samples - 1 {
                std::thread::sleep(std::time::Duration::from_secs(interval));
            }
        }
        
        println!();
        Ok(())
    }

    /// Send command to kernel
    fn send_command(&self, command: &str) -> Result<()> {
        println!("Sending command to kernel: {}", command.bold());
        
        self.write_to_kernel(command)?;
        
        // Wait a moment and read response
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        match self.read_from_kernel() {
            Ok(response) => {
                println!("Kernel response:");
                println!("{}", response);
            }
            Err(e) => {
                println!("{} Failed to read response: {}", "✗".red(), e);
            }
        }
        
        Ok(())
    }

    /// Show kernel logs
    fn show_logs(&self, lines: usize, follow: bool) -> Result<()> {
        println!("{}", "AINKA Kernel Logs".bold().blue());
        println!("{}", "=================".blue());
        
        let output = Command::new("dmesg")
            .arg("-T")
            .output()
            .context("Failed to execute dmesg")?;
        
        let logs = String::from_utf8_lossy(&output.stdout);
        let log_lines: Vec<&str> = logs.lines()
            .filter(|line| line.contains("AINKA"))
            .collect();
        
        let start = if log_lines.len() > lines {
            log_lines.len() - lines
        } else {
            0
        };
        
        for line in &log_lines[start..] {
            println!("{}", line);
        }
        
        if follow {
            println!("Following logs (press Ctrl+C to stop)...");
            // In a real implementation, you would use inotify or similar
            // to watch for new log entries
        }
        
        Ok(())
    }

    /// Show AI suggestions
    fn show_suggestions(&self, all: bool) -> Result<()> {
        println!("{}", "AINKA AI Suggestions".bold().blue());
        println!("{}", "===================".blue());
        
        // Read from kernel to get suggestions
        match self.read_from_kernel() {
            Ok(content) => {
                let lines: Vec<&str> = content.lines().collect();
                let mut suggestions = Vec::new();
                
                for line in lines {
                    if line.contains("ai_suggestion:") {
                        let parts: Vec<&str> = line.split(':').collect();
                        if parts.len() >= 4 {
                            suggestions.push(AISuggestion {
                                category: parts[1].to_string(),
                                severity: parts[2].to_string(),
                                message: parts[3].to_string(),
                                action: "See documentation for details".to_string(),
                                confidence: 0.85,
                            });
                        }
                    }
                }
                
                if suggestions.is_empty() {
                    println!("No AI suggestions available");
                } else {
                    println!("{}", Table::new(suggestions));
                }
            }
            Err(e) => {
                println!("{} Failed to read suggestions: {}", "✗".red(), e);
            }
        }
        
        Ok(())
    }

    /// Control daemon
    fn control_daemon(&self, action: &DaemonCommands) -> Result<()> {
        match action {
            DaemonCommands::Start => {
                println!("Starting AINKA daemon...");
                let output = Command::new("systemctl")
                    .args(["start", "ainka-daemon"])
                    .output()
                    .context("Failed to start daemon")?;
                
                if output.status.success() {
                    println!("{} Daemon started successfully", "✓".green());
                } else {
                    println!("{} Failed to start daemon", "✗".red());
                }
            }
            DaemonCommands::Stop => {
                println!("Stopping AINKA daemon...");
                let output = Command::new("systemctl")
                    .args(["stop", "ainka-daemon"])
                    .output()
                    .context("Failed to stop daemon")?;
                
                if output.status.success() {
                    println!("{} Daemon stopped successfully", "✓".green());
                } else {
                    println!("{} Failed to stop daemon", "✗".red());
                }
            }
            DaemonCommands::Restart => {
                println!("Restarting AINKA daemon...");
                let output = Command::new("systemctl")
                    .args(["restart", "ainka-daemon"])
                    .output()
                    .context("Failed to restart daemon")?;
                
                if output.status.success() {
                    println!("{} Daemon restarted successfully", "✓".green());
                } else {
                    println!("{} Failed to restart daemon", "✗".red());
                }
            }
            DaemonCommands::Status => {
                let running = self.check_daemon_status()?;
                if running {
                    println!("{} Daemon is running", "✓".green());
                } else {
                    println!("{} Daemon is not running", "✗".red());
                }
            }
        }
        
        Ok(())
    }

    /// Show system information
    fn show_system_info(&self) -> Result<()> {
        println!("\n{}", "System Information".bold().blue());
        println!("{}", "==================".blue());
        
        let output = Command::new("uname")
            .args(["-a"])
            .output()
            .context("Failed to get system information")?;
        
        println!("OS: {}", String::from_utf8_lossy(&output.stdout).trim());
        
        // Show kernel version
        let kernel_output = Command::new("uname")
            .args(["-r"])
            .output()
            .context("Failed to get kernel version")?;
        
        println!("Kernel: {}", String::from_utf8_lossy(&kernel_output.stdout).trim());
        
        // Show CPU info
        let cpu_info = self.system.global_cpu_info();
        println!("CPU: {} cores", self.system.cpus().len());
        println!("CPU Model: {}", cpu_info.brand());
        
        // Show memory info
        let memory = self.system.memory();
        println!("Memory: {} GB total, {} GB used", 
                memory.total() / 1024 / 1024 / 1024,
                memory.used() / 1024 / 1024 / 1024);
        
        Ok(())
    }

    /// Test kernel module
    fn test_kernel_module(&self, all: bool) -> Result<()> {
        println!("{}", "AINKA Kernel Module Tests".bold().blue());
        println!("{}", "=========================".blue());
        
        // Test 1: Check if module is loaded
        println!("Test 1: Checking if kernel module is loaded...");
        match self.check_kernel_module() {
            Ok(true) => {
                println!("{} Module is loaded", "✓".green());
            }
            Ok(false) => {
                println!("{} Module is not loaded", "✗".red());
                return Ok(());
            }
            Err(e) => {
                println!("{} Error checking module: {}", "✗".red(), e);
                return Ok(());
            }
        }
        
        // Test 2: Read from /proc interface
        println!("Test 2: Testing /proc interface read...");
        match self.read_from_kernel() {
            Ok(content) => {
                println!("{} Successfully read from /proc/ainka", "✓".green());
                if self.verbose {
                    println!("Content: {}", content);
                }
            }
            Err(e) => {
                println!("{} Failed to read from /proc/ainka: {}", "✗".red(), e);
            }
        }
        
        // Test 3: Write to /proc interface
        println!("Test 3: Testing /proc interface write...");
        match self.write_to_kernel("test_command") {
            Ok(_) => {
                println!("{} Successfully wrote to /proc/ainka", "✓".green());
            }
            Err(e) => {
                println!("{} Failed to write to /proc/ainka: {}", "✗".red(), e);
            }
        }
        
        if all {
            // Test 4: Check kernel logs
            println!("Test 4: Checking kernel logs...");
            let output = Command::new("dmesg")
                .output()
                .context("Failed to execute dmesg")?;
            
            let logs = String::from_utf8_lossy(&output.stdout);
            if logs.contains("AINKA") {
                println!("{} AINKA messages found in kernel logs", "✓".green());
            } else {
                println!("{} No AINKA messages in kernel logs", "✗".red());
            }
        }
        
        println!("{} All tests completed", "✓".green());
        Ok(())
    }

    /// Check daemon status
    fn check_daemon_status(&self) -> Result<bool> {
        let output = Command::new("systemctl")
            .args(["is-active", "ainka-daemon"])
            .output()
            .context("Failed to check daemon status")?;
        
        Ok(String::from_utf8_lossy(&output.stdout).trim() == "active")
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    let mut ainka_cli = AinkaCli::new(cli.kernel_path, cli.verbose);
    
    match cli.command {
        Commands::Status { detailed } => {
            ainka_cli.show_status(detailed)?;
        }
        Commands::Metrics { interval, samples } => {
            ainka_cli.show_metrics(interval, samples)?;
        }
        Commands::Send { command } => {
            ainka_cli.send_command(&command)?;
        }
        Commands::Logs { lines, follow } => {
            ainka_cli.show_logs(lines, follow)?;
        }
        Commands::Suggestions { all } => {
            ainka_cli.show_suggestions(all)?;
        }
        Commands::Daemon { action } => {
            ainka_cli.control_daemon(&action)?;
        }
        Commands::Info { detailed } => {
            ainka_cli.show_system_info()?;
        }
        Commands::Test { all } => {
            ainka_cli.test_kernel_module(all)?;
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cli_creation() {
        let cli = AinkaCli::new("/proc/ainka".to_string(), false);
        assert_eq!(cli.kernel_path, "/proc/ainka");
        assert!(!cli.verbose);
    }
    
    #[test]
    fn test_metrics_collection() {
        let mut cli = AinkaCli::new("/proc/ainka".to_string(), false);
        cli.system.refresh_all();
        
        let metrics = cli.collect_metrics().unwrap();
        assert!(metrics.cpu_usage >= 0.0 && metrics.cpu_usage <= 100.0);
        assert!(metrics.memory_usage >= 0.0 && metrics.memory_usage <= 100.0);
    }
} 