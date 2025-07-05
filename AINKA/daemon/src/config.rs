use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use crate::AinkaError;

/// Main configuration structure for AINKA daemon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// General daemon settings
    pub daemon: DaemonConfig,
    
    /// System monitoring settings
    pub monitoring: MonitoringConfig,
    
    /// Performance optimization settings
    pub optimization: OptimizationConfig,
    
    /// Machine learning settings
    pub ml: MLConfig,
    
    /// Logging settings
    pub logging: LoggingConfig,
    
    /// eBPF settings
    pub ebpf: EbpfConfig,
    
    /// IPC settings
    pub ipc: IpcConfig,
}

/// Daemon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Optimization interval in seconds
    pub optimization_interval: u64,
    
    /// Maximum number of optimization cycles before restart
    pub max_cycles: u64,
    
    /// Enable daemon mode
    pub daemon_mode: bool,
    
    /// Enable interactive mode
    pub interactive_mode: bool,
    
    /// Data directory for storing metrics and models
    pub data_dir: String,
    
    /// Log directory
    pub log_dir: String,
    
    /// PID file path
    pub pid_file: String,
}

/// System monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// CPU monitoring interval in seconds
    pub cpu_interval: u64,
    
    /// Memory monitoring interval in seconds
    pub memory_interval: u64,
    
    /// Disk I/O monitoring interval in seconds
    pub disk_interval: u64,
    
    /// Network monitoring interval in seconds
    pub network_interval: u64,
    
    /// Process monitoring interval in seconds
    pub process_interval: u64,
    
    /// Enable detailed monitoring
    pub detailed_monitoring: bool,
    
    /// Maximum number of processes to monitor
    pub max_processes: usize,
    
    /// CPU usage threshold for alerts
    pub cpu_threshold: f64,
    
    /// Memory usage threshold for alerts
    pub memory_threshold: f64,
    
    /// Disk usage threshold for alerts
    pub disk_threshold: f64,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable CPU optimization
    pub enable_cpu_optimization: bool,
    
    /// Enable memory optimization
    pub enable_memory_optimization: bool,
    
    /// Enable I/O optimization
    pub enable_io_optimization: bool,
    
    /// Enable network optimization
    pub enable_network_optimization: bool,
    
    /// Aggressive optimization mode
    pub aggressive_mode: bool,
    
    /// Maximum optimization attempts per cycle
    pub max_optimizations_per_cycle: usize,
    
    /// Optimization timeout in seconds
    pub optimization_timeout: u64,
    
    /// Enable automatic service restart
    pub enable_service_restart: bool,
    
    /// Enable configuration drift detection
    pub enable_config_drift_detection: bool,
    
    /// CPU governor preferences
    pub cpu_governors: Vec<String>,
    
    /// I/O scheduler preferences
    pub io_schedulers: Vec<String>,
    
    /// Network buffer sizes
    pub network_buffer_sizes: NetworkBufferSizes,
}

/// Network buffer sizes configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkBufferSizes {
    /// TCP receive buffer size
    pub tcp_rmem: (u32, u32, u32),
    
    /// TCP send buffer size
    pub tcp_wmem: (u32, u32, u32),
    
    /// UDP receive buffer size
    pub udp_rmem: u32,
    
    /// UDP send buffer size
    pub udp_wmem: u32,
}

/// Machine learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    /// Enable machine learning predictions
    pub enable_ml: bool,
    
    /// Learning rate for regression models
    pub learning_rate: f64,
    
    /// Number of features to use
    pub feature_count: usize,
    
    /// Model update interval in seconds
    pub model_update_interval: u64,
    
    /// Minimum data points for training
    pub min_training_points: usize,
    
    /// Maximum model age in seconds
    pub max_model_age: u64,
    
    /// Enable feature importance tracking
    pub enable_feature_importance: bool,
    
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    
    /// Anomaly detection threshold
    pub anomaly_threshold: f64,
    
    /// Model storage directory
    pub model_dir: String,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (debug, info, warn, error)
    pub level: String,
    
    /// Log format (text, json)
    pub format: String,
    
    /// Enable file logging
    pub enable_file_logging: bool,
    
    /// Enable console logging
    pub enable_console_logging: bool,
    
    /// Log file path
    pub log_file: String,
    
    /// Maximum log file size in MB
    pub max_log_size: u64,
    
    /// Number of log files to keep
    pub log_rotation_count: u32,
    
    /// Enable structured logging
    pub enable_structured_logging: bool,
}

/// eBPF configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EbpfConfig {
    /// Enable eBPF monitoring
    pub enable_ebpf: bool,
    
    /// eBPF program path
    pub program_path: String,
    
    /// Enable tracepoint monitoring
    pub enable_tracepoints: bool,
    
    /// Enable kprobe monitoring
    pub enable_kprobes: bool,
    
    /// Enable network monitoring
    pub enable_network_monitoring: bool,
    
    /// Maximum events per second
    pub max_events_per_sec: u64,
    
    /// Ring buffer size
    pub ring_buffer_size: u32,
}

/// IPC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcConfig {
    /// Enable netlink communication
    pub enable_netlink: bool,
    
    /// Netlink socket path
    pub netlink_socket: String,
    
    /// Enable shared memory
    pub enable_shared_memory: bool,
    
    /// Shared memory size
    pub shared_memory_size: usize,
    
    /// Enable custom syscalls
    pub enable_custom_syscalls: bool,
    
    /// IPC timeout in seconds
    pub ipc_timeout: u64,
}

impl Config {
    /// Load configuration from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        
        if !path.exists() {
            log::warn!("Configuration file not found: {:?}, using defaults", path);
            return Ok(Self::default());
        }
        
        let content = std::fs::read_to_string(path)
            .context("Failed to read configuration file")?;
        
        let config: Config = toml::from_str(&content)
            .context("Failed to parse configuration file")?;
        
        log::info!("Configuration loaded from: {:?}", path);
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        
        // Create directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create configuration directory")?;
        }
        
        let content = toml::to_string_pretty(self)
            .context("Failed to serialize configuration")?;
        
        std::fs::write(path, content)
            .context("Failed to write configuration file")?;
        
        log::info!("Configuration saved to: {:?}", path);
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate intervals
        if self.daemon.optimization_interval == 0 {
            return Err(AinkaError::Config("Optimization interval cannot be zero".to_string()).into());
        }
        
        if self.monitoring.cpu_interval == 0 {
            return Err(AinkaError::Config("CPU monitoring interval cannot be zero".to_string()).into());
        }
        
        // Validate thresholds
        if self.monitoring.cpu_threshold <= 0.0 || self.monitoring.cpu_threshold > 100.0 {
            return Err(AinkaError::Config("CPU threshold must be between 0 and 100".to_string()).into());
        }
        
        if self.monitoring.memory_threshold <= 0.0 || self.monitoring.memory_threshold > 100.0 {
            return Err(AinkaError::Config("Memory threshold must be between 0 and 100".to_string()).into());
        }
        
        // Validate ML settings
        if self.ml.learning_rate <= 0.0 || self.ml.learning_rate > 1.0 {
            return Err(AinkaError::Config("Learning rate must be between 0 and 1".to_string()).into());
        }
        
        Ok(())
    }
    
    /// Get configuration as JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .context("Failed to serialize configuration to JSON")
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            daemon: DaemonConfig {
                optimization_interval: 300,
                max_cycles: 1000,
                daemon_mode: true,
                interactive_mode: false,
                data_dir: "/var/lib/ainka".to_string(),
                log_dir: "/var/log/ainka".to_string(),
                pid_file: "/var/run/ainka.pid".to_string(),
            },
            monitoring: MonitoringConfig {
                cpu_interval: 5,
                memory_interval: 10,
                disk_interval: 15,
                network_interval: 10,
                process_interval: 30,
                detailed_monitoring: false,
                max_processes: 100,
                cpu_threshold: 80.0,
                memory_threshold: 85.0,
                disk_threshold: 90.0,
            },
            optimization: OptimizationConfig {
                enable_cpu_optimization: true,
                enable_memory_optimization: true,
                enable_io_optimization: true,
                enable_network_optimization: true,
                aggressive_mode: false,
                max_optimizations_per_cycle: 5,
                optimization_timeout: 30,
                enable_service_restart: false,
                enable_config_drift_detection: true,
                cpu_governors: vec!["performance".to_string(), "ondemand".to_string(), "powersave".to_string()],
                io_schedulers: vec!["bfq".to_string(), "kyber".to_string(), "mq-deadline".to_string()],
                network_buffer_sizes: NetworkBufferSizes {
                    tcp_rmem: (4096, 87380, 16777216),
                    tcp_wmem: (4096, 65536, 16777216),
                    udp_rmem: 262144,
                    udp_wmem: 262144,
                },
            },
            ml: MLConfig {
                enable_ml: true,
                learning_rate: 0.01,
                feature_count: 10,
                model_update_interval: 60,
                min_training_points: 100,
                max_model_age: 3600,
                enable_feature_importance: true,
                enable_anomaly_detection: true,
                anomaly_threshold: 2.0,
                model_dir: "/var/lib/ainka/models".to_string(),
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "text".to_string(),
                enable_file_logging: true,
                enable_console_logging: true,
                log_file: "/var/log/ainka/ainka.log".to_string(),
                max_log_size: 100,
                log_rotation_count: 5,
                enable_structured_logging: false,
            },
            ebpf: EbpfConfig {
                enable_ebpf: false,
                program_path: "/usr/lib/ainka/ebpf".to_string(),
                enable_tracepoints: true,
                enable_kprobes: false,
                enable_network_monitoring: false,
                max_events_per_sec: 10000,
                ring_buffer_size: 1024 * 1024,
            },
            ipc: IpcConfig {
                enable_netlink: true,
                netlink_socket: "/proc/net/ainka".to_string(),
                enable_shared_memory: false,
                shared_memory_size: 1024 * 1024,
                enable_custom_syscalls: false,
                ipc_timeout: 5,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.daemon.optimization_interval, 300);
        assert_eq!(config.monitoring.cpu_threshold, 80.0);
        assert!(config.ml.enable_ml);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());
        
        config.daemon.optimization_interval = 0;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_save_load() {
        let config = Config::default();
        let temp_file = NamedTempFile::new().unwrap();
        
        config.save(&temp_file.path()).unwrap();
        let loaded_config = Config::load(&temp_file.path()).unwrap();
        
        assert_eq!(config.daemon.optimization_interval, loaded_config.daemon.optimization_interval);
    }
} 