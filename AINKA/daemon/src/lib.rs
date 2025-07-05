//! AINKA Daemon Library
//! 
//! This library provides the core functionality for the AINKA Intelligent Linux System Optimizer.
//! It includes system monitoring, performance optimization, machine learning, and telemetry collection.

pub mod config;
pub mod system_monitor;
pub mod performance_optimizer;
pub mod ml_engine;
pub mod data_pipeline;
pub mod telemetry_hub;
pub mod utils;
pub mod ebpf_manager;
pub mod ai_engine;
pub mod policy_engine;
pub mod ipc_layer;
pub mod anomaly_detector;
pub mod security_monitor;
pub mod predictive_scaler;
pub mod database;

// Re-export main types for convenience
pub use config::Config;
pub use system_monitor::{SystemMonitor, SystemMetrics};
pub use performance_optimizer::{PerformanceOptimizer, SystemAnalysis, Optimization};
pub use ml_engine::RegressionPipeline;
pub use data_pipeline::{DataPipeline, PreprocessingConfig, FeatureConfig};
pub use telemetry_hub::{TelemetryEvent, EventType};
pub use utils::setup_logging;

/// AINKA Daemon version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration path
pub const DEFAULT_CONFIG_PATH: &str = "/etc/ainka/config.toml";

/// Default log level
pub const DEFAULT_LOG_LEVEL: &str = "info";

/// Default optimization interval in seconds
pub const DEFAULT_OPTIMIZATION_INTERVAL: u64 = 300;

/// Default monitoring duration in seconds
pub const DEFAULT_MONITORING_DURATION: u64 = 60;

/// System service name
pub const SERVICE_NAME: &str = "ainka.service";

/// Binary name
pub const BINARY_NAME: &str = "ainka-daemon";

/// Installation directory
pub const INSTALL_DIR: &str = "/usr/local/bin";

/// Configuration directory
pub const CONFIG_DIR: &str = "/etc/ainka";

/// Data directory
pub const DATA_DIR: &str = "/var/lib/ainka";

/// Log directory
pub const LOG_DIR: &str = "/var/log/ainka";

/// Error types
#[derive(thiserror::Error, Debug)]
pub enum AinkaError {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("System monitoring error: {0}")]
    SystemMonitor(String),
    
    #[error("Performance optimization error: {0}")]
    PerformanceOptimizer(String),
    
    #[error("Machine learning error: {0}")]
    MachineLearning(String),
    
    #[error("Data pipeline error: {0}")]
    DataPipeline(String),
    
    #[error("Telemetry error: {0}")]
    Telemetry(String),
    
    #[error("eBPF error: {0}")]
    Ebpf(String),
    
    #[error("IPC error: {0}")]
    Ipc(String),
    
    #[error("Service error: {0}")]
    Service(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("System command error: {0}")]
    SystemCommand(String),
}

/// Result type for AINKA operations
pub type AinkaResult<T> = Result<T, AinkaError>;

/// System health status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum HealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Performance metrics summary
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceSummary {
    pub cpu_efficiency: f64,
    pub memory_efficiency: f64,
    pub io_efficiency: f64,
    pub network_efficiency: f64,
    pub overall_score: f64,
    pub health_status: HealthStatus,
    pub recommendations: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Optimization result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizationResult {
    pub applied: Vec<String>,
    pub skipped: Vec<String>,
    pub failed: Vec<String>,
    pub performance_improvement: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// System event
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemEvent {
    pub event_type: String,
    pub severity: String,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: serde_json::Value,
}

/// AINKA daemon statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DaemonStats {
    pub uptime: std::time::Duration,
    pub optimization_cycles: u64,
    pub events_processed: u64,
    pub ml_predictions: u64,
    pub optimizations_applied: u64,
    pub errors_count: u64,
    pub last_optimization: Option<chrono::DateTime<chrono::Utc>>,
    pub average_cycle_time: std::time::Duration,
}

impl Default for DaemonStats {
    fn default() -> Self {
        Self {
            uptime: std::time::Duration::from_secs(0),
            optimization_cycles: 0,
            events_processed: 0,
            ml_predictions: 0,
            optimizations_applied: 0,
            errors_count: 0,
            last_optimization: None,
            average_cycle_time: std::time::Duration::from_secs(0),
        }
    }
}

/// Main AINKA daemon struct
pub struct AinkaDaemon {
    config: Config,
    system_monitor: SystemMonitor,
    performance_optimizer: PerformanceOptimizer,
    ml_pipeline: RegressionPipeline,
    data_pipeline: DataPipeline,
    stats: DaemonStats,
    start_time: std::time::Instant,
}

impl AinkaDaemon {
    /// Create a new AINKA daemon instance
    pub fn new(config: Config) -> AinkaResult<Self> {
        let system_monitor = SystemMonitor::new(config.clone());
        let performance_optimizer = PerformanceOptimizer::new(config.clone());
        let ml_pipeline = RegressionPipeline::new(30, "system_load".to_string());
        let data_pipeline = DataPipeline::new(
            tokio::sync::mpsc::unbounded_channel().0,
            PreprocessingConfig::default(),
            FeatureConfig::default(),
        );
        
        Ok(Self {
            config,
            system_monitor,
            performance_optimizer,
            ml_pipeline,
            data_pipeline,
            stats: DaemonStats::default(),
            start_time: std::time::Instant::now(),
        })
    }
    
    /// Start the daemon
    pub async fn start(&mut self) -> AinkaResult<()> {
        log::info!("Starting AINKA daemon v{}", VERSION);
        
        // Initialize components
        self.system_monitor.initialize().await?;
        self.performance_optimizer.initialize().await?;
        
        log::info!("AINKA daemon started successfully");
        Ok(())
    }
    
    /// Stop the daemon
    pub async fn stop(&mut self) -> AinkaResult<()> {
        log::info!("Stopping AINKA daemon");
        
        // Cleanup
        self.system_monitor.cleanup().await?;
        self.performance_optimizer.cleanup().await?;
        
        log::info!("AINKA daemon stopped");
        Ok(())
    }
    
    /// Get daemon statistics
    pub fn get_stats(&self) -> DaemonStats {
        let mut stats = self.stats.clone();
        stats.uptime = self.start_time.elapsed();
        stats
    }
    
    /// Run a single optimization cycle
    pub async fn run_optimization_cycle(&mut self) -> AinkaResult<OptimizationResult> {
        let cycle_start = std::time::Instant::now();
        
        // Collect metrics
        let metrics = self.system_monitor.collect_metrics().await?;
        
        // Generate events
        let events = self.system_monitor.generate_events(metrics.clone()).await?;
        self.stats.events_processed += events.len() as u64;
        
        // Process through ML pipeline
        self.ml_pipeline.process_events(events.clone()).await?;
        let predictions = self.ml_pipeline.predict(&vec![])?;
        self.stats.ml_predictions += 1;
        
        // Apply optimizations
        let optimizations = self.performance_optimizer.optimize_system(metrics, predictions).await?;
        
        let mut result = OptimizationResult {
            applied: Vec::new(),
            skipped: Vec::new(),
            failed: Vec::new(),
            performance_improvement: 0.0,
            timestamp: chrono::Utc::now(),
        };
        
        for optimization in optimizations {
            match optimization.apply().await {
                Ok(_) => {
                    result.applied.push(optimization.description);
                    self.stats.optimizations_applied += 1;
                }
                Err(e) => {
                    result.failed.push(format!("{}: {}", optimization.description, e));
                    self.stats.errors_count += 1;
                }
            }
        }
        
        self.stats.optimization_cycles += 1;
        self.stats.last_optimization = Some(chrono::Utc::now());
        self.stats.average_cycle_time = cycle_start.elapsed();
        
        Ok(result)
    }
} 