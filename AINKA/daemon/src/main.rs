/*
 * AINKA AI Daemon - Userspace AI Learning and Decision Engine
 * 
 * This daemon communicates with the AINKA kernel module to provide
 * intelligent system optimization and autonomous management.
 */

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::interval;
use anyhow::{Result, Context};
use clap::{Parser, Subcommand};
use log::{info, warn, error, debug};
use serde_json::json;

use ainka_daemon::{
    system_monitor::SystemMonitor,
    performance_optimizer::PerformanceOptimizer,
    ml_engine::RegressionPipeline,
    data_pipeline::{DataPipeline, PreprocessingConfig, FeatureConfig},
    telemetry_hub::{TelemetryEvent, EventType},
    config::Config,
    utils::setup_logging,
};

#[derive(Parser)]
#[command(name = "ainka-daemon")]
#[command(about = "AINKA - Intelligent Linux System Optimizer")]
#[command(version = "0.2.0")]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    #[arg(short, long, default_value = "info")]
    log_level: String,
    
    #[arg(short, long, default_value = "/etc/ainka/config.toml")]
    config: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the AINKA daemon for continuous system optimization
    Start {
        #[arg(short, long)]
        daemon: bool,
        
        #[arg(short, long, default_value = "300")]
        interval: u64,
    },
    
    /// Optimize system performance immediately
    Optimize {
        #[arg(short, long)]
        aggressive: bool,
        
        #[arg(short, long)]
        target: Option<String>,
    },
    
    /// Monitor system performance and show insights
    Monitor {
        #[arg(short, long, default_value = "60")]
        duration: u64,
        
        #[arg(short, long)]
        json: bool,
    },
    
    /// Analyze system performance and provide recommendations
    Analyze {
        #[arg(short, long)]
        detailed: bool,
        
        #[arg(short, long)]
        output: Option<String>,
    },
    
    /// Install AINKA as a system service
    Install,
    
    /// Uninstall AINKA system service
    Uninstall,
    
    /// Show system status and AINKA health
    Status,
    
    /// Show configuration and settings
    Config {
        #[arg(short, long)]
        show: bool,
        
        #[arg(short, long)]
        edit: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Setup logging
    setup_logging(&cli.log_level)?;
    
    info!("AINKA Intelligent Linux System Optimizer v0.2.0");
    info!("================================================");
    
    // Load configuration
    let config = Config::load(&cli.config).context("Failed to load configuration")?;
    
    match cli.command {
        Commands::Start { daemon, interval } => {
            if daemon {
                run_daemon_mode(config, interval).await?;
            } else {
                run_interactive_mode(config, interval).await?;
            }
        }
        Commands::Optimize { aggressive, target } => {
            run_optimization(config, aggressive, target).await?;
        }
        Commands::Monitor { duration, json } => {
            run_monitoring(config, duration, json).await?;
        }
        Commands::Analyze { detailed, output } => {
            run_analysis(config, detailed, output).await?;
        }
        Commands::Install => {
            install_service().await?;
        }
        Commands::Uninstall => {
            uninstall_service().await?;
        }
        Commands::Status => {
            show_status(config).await?;
        }
        Commands::Config { show, edit } => {
            if show {
                show_config(&config)?;
            } else if edit {
                edit_config(&cli.config).await?;
            } else {
                show_config(&config)?;
            }
        }
    }
    
    Ok(())
}

/// Run AINKA in daemon mode for continuous optimization
async fn run_daemon_mode(config: Config, interval_secs: u64) -> Result<()> {
    info!("Starting AINKA daemon mode...");
    
    // Create event channel
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<TelemetryEvent>();
    
    // Initialize components
    let system_monitor = Arc::new(SystemMonitor::new(config.clone()));
    let performance_optimizer = Arc::new(PerformanceOptimizer::new(config.clone()));
    
    // Initialize data pipeline
    let data_pipeline = DataPipeline::new(
        event_tx.clone(),
        PreprocessingConfig::default(),
        FeatureConfig::default(),
    );
    
    // Initialize ML pipeline for system load prediction
    let mut ml_pipeline = RegressionPipeline::new(30, "system_load".to_string());
    
    info!("AINKA daemon started successfully");
    info!("Optimization interval: {} seconds", interval_secs);
    info!("Press Ctrl+C to stop");
    
    let mut interval_timer = interval(Duration::from_secs(interval_secs));
    let mut iteration = 0;
    
    loop {
        interval_timer.tick().await;
        iteration += 1;
        
        info!("=== Optimization Cycle {} ===", iteration);
        
        // Collect system metrics
        let metrics = system_monitor.collect_metrics().await?;
        
        // Generate telemetry events
        let events = system_monitor.generate_events(metrics).await?;
        
        // Process events through data pipeline
        data_pipeline.process_events(events.clone()).await?;
        
        // Update ML model
        ml_pipeline.process_events(events).await?;
        
        // Get ML predictions
        let predictions = ml_pipeline.predict(&vec![])?;
        
        // Perform optimizations
        let optimizations = performance_optimizer.optimize_system(metrics, predictions).await?;
        
        // Apply optimizations
        for optimization in optimizations {
            info!("Applied optimization: {}", optimization.description);
            if let Err(e) = optimization.apply().await {
                warn!("Failed to apply optimization: {}", e);
            }
        }
        
        // Show performance metrics
        let ml_metrics = ml_pipeline.get_performance_metrics();
        info!("ML Model Performance - MSE: {:.6}, R²: {:.6}", ml_metrics.mse, ml_metrics.r2);
        
        // Check for urgent issues
        if let Some(issue) = system_monitor.detect_urgent_issues().await? {
            warn!("Urgent issue detected: {}", issue.description);
            if let Err(e) = issue.resolve().await {
                error!("Failed to resolve urgent issue: {}", e);
            }
        }
    }
}

/// Run AINKA in interactive mode
async fn run_interactive_mode(config: Config, interval_secs: u64) -> Result<()> {
    info!("Starting AINKA interactive mode...");
    info!("Press 'q' to quit, 's' for status, 'o' for immediate optimization");
    
    let system_monitor = Arc::new(SystemMonitor::new(config.clone()));
    let performance_optimizer = Arc::new(PerformanceOptimizer::new(config));
    
    let mut interval_timer = interval(Duration::from_secs(interval_secs));
    
    loop {
        interval_timer.tick().await;
        
        // Collect and display metrics
        let metrics = system_monitor.collect_metrics().await?;
        display_system_status(&metrics)?;
        
        // Check for user input (simplified - in real implementation, use async input)
        // For now, just show status every interval
    }
}

/// Run immediate system optimization
async fn run_optimization(config: Config, aggressive: bool, target: Option<String>) -> Result<()> {
    info!("Running immediate system optimization...");
    
    let system_monitor = Arc::new(SystemMonitor::new(config.clone()));
    let performance_optimizer = Arc::new(PerformanceOptimizer::new(config));
    
    // Collect current system state
    let metrics = system_monitor.collect_metrics().await?;
    
    info!("Current system state:");
    display_system_status(&metrics)?;
    
    // Perform targeted or general optimization
    let optimizations = if let Some(target) = target {
        performance_optimizer.optimize_target(&target, &metrics, aggressive).await?
    } else {
        performance_optimizer.optimize_system(metrics, vec![]).await?
    };
    
    // Apply optimizations
    info!("Applying optimizations...");
    for optimization in optimizations {
        info!("Applying: {}", optimization.description);
        match optimization.apply().await {
            Ok(_) => info!("✓ Success: {}", optimization.description),
            Err(e) => warn!("✗ Failed: {} - {}", optimization.description, e),
        }
    }
    
    // Show results
    let new_metrics = system_monitor.collect_metrics().await?;
    info!("Optimization complete!");
    info!("Performance improvement: {:.2}%", calculate_improvement(&metrics, &new_metrics));
    
    Ok(())
}

/// Run system monitoring
async fn run_monitoring(config: Config, duration_secs: u64, json_output: bool) -> Result<()> {
    info!("Monitoring system for {} seconds...", duration_secs);
    
    let system_monitor = Arc::new(SystemMonitor::new(config));
    let start_time = Instant::now();
    let mut interval_timer = interval(Duration::from_secs(1));
    
    while start_time.elapsed().as_secs() < duration_secs {
        interval_timer.tick().await;
        
        let metrics = system_monitor.collect_metrics().await?;
        
        if json_output {
            println!("{}", serde_json::to_string_pretty(&metrics)?);
        } else {
            display_system_status(&metrics)?;
        }
    }
    
    info!("Monitoring complete");
    Ok(())
}

/// Run system analysis
async fn run_analysis(config: Config, detailed: bool, output_file: Option<String>) -> Result<()> {
    info!("Analyzing system performance...");
    
    let system_monitor = Arc::new(SystemMonitor::new(config.clone()));
    let performance_optimizer = Arc::new(PerformanceOptimizer::new(config));
    
    // Collect comprehensive metrics
    let metrics = system_monitor.collect_detailed_metrics().await?;
    
    // Generate analysis report
    let analysis = performance_optimizer.analyze_system(&metrics, detailed).await?;
    
    // Display or save report
    if let Some(output_path) = output_file {
        std::fs::write(&output_path, serde_json::to_string_pretty(&analysis)?)?;
        info!("Analysis saved to: {}", output_path);
    } else {
        display_analysis(&analysis)?;
    }
    
    Ok(())
}

/// Install AINKA as a system service
async fn install_service() -> Result<()> {
    info!("Installing AINKA system service...");
    
    // Create systemd service file
    let service_content = r#"[Unit]
Description=AINKA Intelligent Linux System Optimizer
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/ainka-daemon start --daemon
Restart=always
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target"#;
    
    std::fs::write("/etc/systemd/system/ainka.service", service_content)?;
    
    // Reload systemd and enable service
    std::process::Command::new("systemctl")
        .args(["daemon-reload"])
        .status()?;
    
    std::process::Command::new("systemctl")
        .args(["enable", "ainka.service"])
        .status()?;
    
    info!("AINKA service installed successfully");
    info!("Start with: sudo systemctl start ainka");
    info!("Check status with: sudo systemctl status ainka");
    
    Ok(())
}

/// Uninstall AINKA system service
async fn uninstall_service() -> Result<()> {
    info!("Uninstalling AINKA system service...");
    
    // Stop and disable service
    let _ = std::process::Command::new("systemctl")
        .args(["stop", "ainka.service"])
        .status();
    
    let _ = std::process::Command::new("systemctl")
        .args(["disable", "ainka.service"])
        .status();
    
    // Remove service file
    let _ = std::fs::remove_file("/etc/systemd/system/ainka.service");
    
    // Reload systemd
    let _ = std::process::Command::new("systemctl")
        .args(["daemon-reload"])
        .status();
    
    info!("AINKA service uninstalled successfully");
    Ok(())
}

/// Show system status and AINKA health
async fn show_status(config: Config) -> Result<()> {
    info!("AINKA System Status");
    info!("==================");
    
    let system_monitor = Arc::new(SystemMonitor::new(config));
    let metrics = system_monitor.collect_metrics().await?;
    
    // System status
    info!("System Status:");
    info!("  CPU Usage: {:.1}%", metrics.cpu_usage);
    info!("  Memory Usage: {:.1}%", metrics.memory_usage);
    info!("  Load Average: {:.2}, {:.2}, {:.2}", 
        metrics.load_average.0, metrics.load_average.1, metrics.load_average.2);
    info!("  Disk I/O: {:.1} MB/s read, {:.1} MB/s write", 
        metrics.disk_io.read_mbps, metrics.disk_io.write_mbps);
    info!("  Network: {:.1} MB/s in, {:.1} MB/s out", 
        metrics.network.in_mbps, metrics.network.out_mbps);
    
    // Service status
    let service_status = std::process::Command::new("systemctl")
        .args(["is-active", "ainka.service"])
        .output();
    
    match service_status {
        Ok(output) => {
            let status = String::from_utf8_lossy(&output.stdout).trim();
            info!("AINKA Service: {}", if status == "active" { "✓ Running" } else { "✗ Stopped" });
        }
        Err(_) => {
            info!("AINKA Service: Not installed");
        }
    }
    
    Ok(())
}

/// Show configuration
fn show_config(config: &Config) -> Result<()> {
    info!("AINKA Configuration");
    info!("===================");
    println!("{}", serde_json::to_string_pretty(config)?);
    Ok(())
}

/// Edit configuration
async fn edit_config(config_path: &str) -> Result<()> {
    info!("Opening configuration file for editing: {}", config_path);
    
    // Try to open with default editor
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nano".to_string());
    
    std::process::Command::new(editor)
        .arg(config_path)
        .status()?;
    
    info!("Configuration file edited");
    Ok(())
}

/// Display system status in a user-friendly format
fn display_system_status(metrics: &ainka_daemon::system_monitor::SystemMetrics) -> Result<()> {
    println!("\n=== System Status ===");
    println!("CPU:    {:>6.1}% | Memory: {:>6.1}% | Load: {:>5.2}", 
        metrics.cpu_usage, metrics.memory_usage, metrics.load_average.0);
    println!("Disk I/O: {:>6.1} MB/s | Network: {:>6.1} MB/s", 
        metrics.disk_io.read_mbps + metrics.disk_io.write_mbps,
        metrics.network.in_mbps + metrics.network.out_mbps);
    println!("Uptime: {} | Processes: {}", 
        format_duration(metrics.uptime), metrics.process_count);
    println!("========================\n");
    Ok(())
}

/// Display analysis results
fn display_analysis(analysis: &ainka_daemon::performance_optimizer::SystemAnalysis) -> Result<()> {
    println!("\n=== System Analysis ===");
    println!("Overall Health Score: {:.1}/100", analysis.health_score);
    println!("\nIssues Found:");
    for issue in &analysis.issues {
        println!("  • {} (Severity: {})", issue.description, issue.severity);
    }
    println!("\nRecommendations:");
    for rec in &analysis.recommendations {
        println!("  • {}", rec.description);
    }
    println!("\nPerformance Metrics:");
    println!("  CPU Efficiency: {:.1}%", analysis.cpu_efficiency);
    println!("  Memory Efficiency: {:.1}%", analysis.memory_efficiency);
    println!("  I/O Efficiency: {:.1}%", analysis.io_efficiency);
    println!("  Network Efficiency: {:.1}%", analysis.network_efficiency);
    println!("=====================\n");
    Ok(())
}

/// Calculate performance improvement percentage
fn calculate_improvement(old: &ainka_daemon::system_monitor::SystemMetrics, 
                        new: &ainka_daemon::system_monitor::SystemMetrics) -> f64 {
    let old_score = (100.0 - old.cpu_usage) + (100.0 - old.memory_usage);
    let new_score = (100.0 - new.cpu_usage) + (100.0 - new.memory_usage);
    ((new_score - old_score) / old_score) * 100.0
}

/// Format duration in a human-readable way
fn format_duration(seconds: u64) -> String {
    let days = seconds / 86400;
    let hours = (seconds % 86400) / 3600;
    let minutes = (seconds % 3600) / 60;
    
    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
} 