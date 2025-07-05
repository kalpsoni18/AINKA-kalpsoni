use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use sysinfo::{System, SystemExt, CpuExt, DiskExt, NetworkExt, ProcessExt};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use crate::{Config, TelemetryEvent, EventType, AinkaError};

/// System metrics collected by the monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    
    /// Memory usage percentage
    pub memory_usage: f64,
    
    /// Load average (1min, 5min, 15min)
    pub load_average: (f64, f64, f64),
    
    /// Disk I/O metrics
    pub disk_io: DiskIOMetrics,
    
    /// Network metrics
    pub network: NetworkMetrics,
    
    /// Process count
    pub process_count: usize,
    
    /// System uptime in seconds
    pub uptime: u64,
    
    /// Top processes by CPU usage
    pub top_processes: Vec<ProcessInfo>,
    
    /// System temperature (if available)
    pub temperature: Option<f64>,
    
    /// Power consumption (if available)
    pub power_consumption: Option<f64>,
    
    /// Timestamp of collection
    pub timestamp: DateTime<Utc>,
}

/// Disk I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOMetrics {
    /// Read operations per second
    pub read_ops: u64,
    
    /// Write operations per second
    pub write_ops: u64,
    
    /// Read bandwidth in MB/s
    pub read_mbps: f64,
    
    /// Write bandwidth in MB/s
    pub write_mbps: f64,
    
    /// Average read latency in milliseconds
    pub read_latency_ms: f64,
    
    /// Average write latency in milliseconds
    pub write_latency_ms: f64,
    
    /// Disk utilization percentage
    pub utilization: f64,
}

/// Network metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Incoming bandwidth in MB/s
    pub in_mbps: f64,
    
    /// Outgoing bandwidth in MB/s
    pub out_mbps: f64,
    
    /// Incoming packets per second
    pub in_packets: u64,
    
    /// Outgoing packets per second
    pub out_packets: u64,
    
    /// Network errors
    pub errors: u64,
    
    /// Network drops
    pub drops: u64,
}

/// Process information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    /// Process ID
    pub pid: u32,
    
    /// Process name
    pub name: String,
    
    /// CPU usage percentage
    pub cpu_usage: f64,
    
    /// Memory usage in MB
    pub memory_mb: u64,
    
    /// Process state
    pub state: String,
    
    /// Command line
    pub command: String,
}

/// System monitor for collecting metrics
pub struct SystemMonitor {
    config: Config,
    sys: System,
    last_cpu_usage: f64,
    last_disk_io: Option<DiskIOMetrics>,
    last_network: Option<NetworkMetrics>,
    last_update: Instant,
}

impl SystemMonitor {
    /// Create a new system monitor
    pub fn new(config: Config) -> Self {
        Self {
            config,
            sys: System::new_all(),
            last_cpu_usage: 0.0,
            last_disk_io: None,
            last_network: None,
            last_update: Instant::now(),
        }
    }
    
    /// Initialize the system monitor
    pub async fn initialize(&mut self) -> Result<()> {
        log::info!("Initializing system monitor");
        
        // Refresh system information
        self.sys.refresh_all();
        
        // Get initial metrics
        self.last_cpu_usage = self.sys.global_cpu_info().cpu_usage();
        
        log::info!("System monitor initialized");
        Ok(())
    }
    
    /// Cleanup the system monitor
    pub async fn cleanup(&mut self) -> Result<()> {
        log::info!("Cleaning up system monitor");
        Ok(())
    }
    
    /// Collect current system metrics
    pub async fn collect_metrics(&mut self) -> Result<SystemMetrics> {
        let start_time = Instant::now();
        
        // Refresh system information
        self.sys.refresh_all();
        
        // Collect CPU metrics
        let cpu_usage = self.collect_cpu_metrics().await?;
        
        // Collect memory metrics
        let memory_usage = self.collect_memory_metrics().await?;
        
        // Collect load average
        let load_average = self.collect_load_average().await?;
        
        // Collect disk I/O metrics
        let disk_io = self.collect_disk_metrics().await?;
        
        // Collect network metrics
        let network = self.collect_network_metrics().await?;
        
        // Collect process information
        let (process_count, top_processes) = self.collect_process_metrics().await?;
        
        // Collect system uptime
        let uptime = self.sys.uptime();
        
        // Collect temperature and power (if available)
        let temperature = self.collect_temperature().await?;
        let power_consumption = self.collect_power_consumption().await?;
        
        let metrics = SystemMetrics {
            cpu_usage,
            memory_usage,
            load_average,
            disk_io,
            network,
            process_count,
            uptime,
            top_processes,
            temperature,
            power_consumption,
            timestamp: Utc::now(),
        };
        
        // Update last values
        self.last_cpu_usage = cpu_usage;
        self.last_disk_io = Some(disk_io.clone());
        self.last_network = Some(network.clone());
        self.last_update = start_time;
        
        log::debug!("Collected system metrics in {:?}", start_time.elapsed());
        Ok(metrics)
    }
    
    /// Collect detailed system metrics
    pub async fn collect_detailed_metrics(&mut self) -> Result<SystemMetrics> {
        let mut metrics = self.collect_metrics().await?;
        
        // Add detailed process information
        if self.config.monitoring.detailed_monitoring {
            // This would include more detailed process analysis
            log::debug!("Detailed monitoring enabled");
        }
        
        Ok(metrics)
    }
    
    /// Generate telemetry events from metrics
    pub async fn generate_events(&self, metrics: SystemMetrics) -> Result<Vec<TelemetryEvent>> {
        let mut events = Vec::new();
        
        // CPU events
        if metrics.cpu_usage > self.config.monitoring.cpu_threshold {
            events.push(TelemetryEvent {
                event_type: EventType::HighCpuUsage,
                timestamp: metrics.timestamp,
                data: serde_json::json!({
                    "cpu_usage": metrics.cpu_usage,
                    "threshold": self.config.monitoring.cpu_threshold
                }),
            });
        }
        
        // Memory events
        if metrics.memory_usage > self.config.monitoring.memory_threshold {
            events.push(TelemetryEvent {
                event_type: EventType::HighMemoryUsage,
                timestamp: metrics.timestamp,
                data: serde_json::json!({
                    "memory_usage": metrics.memory_usage,
                    "threshold": self.config.monitoring.memory_threshold
                }),
            });
        }
        
        // Load average events
        if metrics.load_average.0 > 1.0 {
            events.push(TelemetryEvent {
                event_type: EventType::HighLoadAverage,
                timestamp: metrics.timestamp,
                data: serde_json::json!({
                    "load_average": metrics.load_average.0
                }),
            });
        }
        
        // Disk I/O events
        if metrics.disk_io.utilization > 80.0 {
            events.push(TelemetryEvent {
                event_type: EventType::HighDiskIO,
                timestamp: metrics.timestamp,
                data: serde_json::json!({
                    "disk_utilization": metrics.disk_io.utilization
                }),
            });
        }
        
        // Network events
        if metrics.network.errors > 0 {
            events.push(TelemetryEvent {
                event_type: EventType::NetworkErrors,
                timestamp: metrics.timestamp,
                data: serde_json::json!({
                    "network_errors": metrics.network.errors
                }),
            });
        }
        
        // Temperature events
        if let Some(temp) = metrics.temperature {
            if temp > 80.0 {
                events.push(TelemetryEvent {
                    event_type: EventType::HighTemperature,
                    timestamp: metrics.timestamp,
                    data: serde_json::json!({
                        "temperature": temp
                    }),
                });
            }
        }
        
        Ok(events)
    }
    
    /// Detect urgent system issues
    pub async fn detect_urgent_issues(&self) -> Result<Option<SystemIssue>> {
        // This would implement urgent issue detection
        // For now, return None
        Ok(None)
    }
    
    /// Collect CPU metrics
    async fn collect_cpu_metrics(&self) -> Result<f64> {
        let cpu_usage = self.sys.global_cpu_info().cpu_usage();
        Ok(cpu_usage)
    }
    
    /// Collect memory metrics
    async fn collect_memory_metrics(&self) -> Result<f64> {
        let total_memory = self.sys.total_memory();
        let used_memory = self.sys.used_memory();
        let memory_usage = (used_memory as f64 / total_memory as f64) * 100.0;
        Ok(memory_usage)
    }
    
    /// Collect load average
    async fn collect_load_average(&self) -> Result<(f64, f64, f64)> {
        let load_avg = self.sys.load_average();
        Ok((load_avg.one, load_avg.five, load_avg.fifteen))
    }
    
    /// Collect disk I/O metrics
    async fn collect_disk_metrics(&self) -> Result<DiskIOMetrics> {
        let mut total_read_ops = 0;
        let mut total_write_ops = 0;
        let mut total_read_bytes = 0;
        let mut total_write_bytes = 0;
        let mut disk_count = 0;
        
        for disk in self.sys.disks() {
            total_read_ops += disk.read_operations();
            total_write_ops += disk.write_operations();
            total_read_bytes += disk.read_bytes();
            total_write_bytes += disk.write_bytes();
            disk_count += 1;
        }
        
        let time_diff = self.last_update.elapsed().as_secs_f64();
        let read_mbps = (total_read_bytes as f64 / 1024.0 / 1024.0) / time_diff;
        let write_mbps = (total_write_bytes as f64 / 1024.0 / 1024.0) / time_diff;
        
        Ok(DiskIOMetrics {
            read_ops: total_read_ops,
            write_ops: total_write_ops,
            read_mbps,
            write_mbps,
            read_latency_ms: 0.0, // Would need more sophisticated tracking
            write_latency_ms: 0.0,
            utilization: 0.0, // Would need more sophisticated tracking
        })
    }
    
    /// Collect network metrics
    async fn collect_network_metrics(&self) -> Result<NetworkMetrics> {
        let mut total_in_bytes = 0;
        let mut total_out_bytes = 0;
        let mut total_in_packets = 0;
        let mut total_out_packets = 0;
        let mut total_errors = 0;
        let mut total_drops = 0;
        
        for (_, network) in self.sys.networks() {
            total_in_bytes += network.received();
            total_out_bytes += network.transmitted();
            total_in_packets += network.packets_received();
            total_out_packets += network.packets_transmitted();
            total_errors += network.errors_on_received();
            total_drops += network.dropped_on_received();
        }
        
        let time_diff = self.last_update.elapsed().as_secs_f64();
        let in_mbps = (total_in_bytes as f64 / 1024.0 / 1024.0) / time_diff;
        let out_mbps = (total_out_bytes as f64 / 1024.0 / 1024.0) / time_diff;
        
        Ok(NetworkMetrics {
            in_mbps,
            out_mbps,
            in_packets: total_in_packets,
            out_packets: total_out_packets,
            errors: total_errors,
            drops: total_drops,
        })
    }
    
    /// Collect process metrics
    async fn collect_process_metrics(&self) -> Result<(usize, Vec<ProcessInfo>)> {
        let mut processes = Vec::new();
        
        for (pid, process) in self.sys.processes() {
            if processes.len() >= self.config.monitoring.max_processes {
                break;
            }
            
            processes.push(ProcessInfo {
                pid: *pid,
                name: process.name().to_string(),
                cpu_usage: process.cpu_usage(),
                memory_mb: process.memory() / 1024,
                state: format!("{:?}", process.status()),
                command: process.cmd().join(" "),
            });
        }
        
        // Sort by CPU usage
        processes.sort_by(|a, b| b.cpu_usage.partial_cmp(&a.cpu_usage).unwrap());
        
        // Take top 10
        let top_processes = processes.into_iter().take(10).collect();
        
        Ok((self.sys.processes().len(), top_processes))
    }
    
    /// Collect temperature (if available)
    async fn collect_temperature(&self) -> Result<Option<f64>> {
        // This would read from /sys/class/thermal/thermal_zone*/temp
        // For now, return None
        Ok(None)
    }
    
    /// Collect power consumption (if available)
    async fn collect_power_consumption(&self) -> Result<Option<f64>> {
        // This would read from power management interfaces
        // For now, return None
        Ok(None)
    }
}

/// System issue that needs attention
#[derive(Debug, Clone)]
pub struct SystemIssue {
    pub severity: IssueSeverity,
    pub description: String,
    pub resolution: String,
}

impl SystemIssue {
    /// Resolve the issue
    pub async fn resolve(&self) -> Result<()> {
        log::info!("Resolving issue: {}", self.description);
        // Implementation would depend on the specific issue
        Ok(())
    }
}

/// Issue severity levels
#[derive(Debug, Clone)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_system_monitor_creation() {
        let config = Config::default();
        let monitor = SystemMonitor::new(config);
        assert_eq!(monitor.last_cpu_usage, 0.0);
    }
    
    #[tokio::test]
    async fn test_metrics_collection() {
        let config = Config::default();
        let mut monitor = SystemMonitor::new(config);
        monitor.initialize().await.unwrap();
        
        let metrics = monitor.collect_metrics().await.unwrap();
        assert!(metrics.cpu_usage >= 0.0 && metrics.cpu_usage <= 100.0);
        assert!(metrics.memory_usage >= 0.0 && metrics.memory_usage <= 100.0);
    }
} 