use std::path::Path;
use anyhow::{Result, Context};
use log::{LevelFilter, info, warn, error};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

/// Setup logging with the specified level
pub fn setup_logging(level: &str) -> Result<()> {
    let level_filter = match level.to_lowercase().as_str() {
        "trace" => LevelFilter::Trace,
        "debug" => LevelFilter::Debug,
        "info" => LevelFilter::Info,
        "warn" => LevelFilter::Warn,
        "error" => LevelFilter::Error,
        _ => LevelFilter::Info,
    };
    
    // Initialize tracing subscriber
    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env()
            .add_directive(format!("ainka_daemon={}", level_filter).parse()?)
            .add_directive("info".parse()?))
        .with(fmt::layer()
            .with_target(false)
            .with_thread_ids(false)
            .with_thread_names(false))
        .init();
    
    info!("Logging initialized with level: {}", level);
    Ok(())
}

/// Check if running as root
pub fn is_root() -> bool {
    nix::unistd::Uid::effective().is_root()
}

/// Check if a file exists and is readable
pub fn file_exists_and_readable<P: AsRef<Path>>(path: P) -> bool {
    let path = path.as_ref();
    path.exists() && path.is_file() && std::fs::metadata(path).is_ok()
}

/// Check if a directory exists and is writable
pub fn directory_exists_and_writable<P: AsRef<Path>>(path: P) -> bool {
    let path = path.as_ref();
    path.exists() && path.is_dir() && {
        let metadata = std::fs::metadata(path);
        metadata.is_ok() && metadata.unwrap().permissions().readonly() == false
    }
}

/// Create directory if it doesn't exist
pub fn ensure_directory<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        std::fs::create_dir_all(path)
            .context(format!("Failed to create directory: {:?}", path))?;
        info!("Created directory: {:?}", path);
    }
    Ok(())
}

/// Write PID to file
pub fn write_pid_file<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    let pid = std::process::id();
    std::fs::write(path, pid.to_string())
        .context(format!("Failed to write PID file: {:?}", path))?;
    info!("Wrote PID {} to file: {:?}", pid, path);
    Ok(())
}

/// Remove PID file
pub fn remove_pid_file<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if path.exists() {
        std::fs::remove_file(path)
            .context(format!("Failed to remove PID file: {:?}", path))?;
        info!("Removed PID file: {:?}", path);
    }
    Ok(())
}

/// Get system information
pub fn get_system_info() -> Result<SystemInfo> {
    let uname = nix::sys::utsname::uname()?;
    
    Ok(SystemInfo {
        kernel_version: uname.release().to_string(),
        hostname: uname.nodename().to_string(),
        architecture: uname.machine().to_string(),
        os_name: uname.sysname().to_string(),
    })
}

/// System information
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub kernel_version: String,
    pub hostname: String,
    pub architecture: String,
    pub os_name: String,
}

/// Format bytes to human readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:.1} {}", size, UNITS[unit_index])
}

/// Format duration to human readable format
pub fn format_duration(duration: std::time::Duration) -> String {
    let seconds = duration.as_secs();
    let minutes = seconds / 60;
    let hours = minutes / 60;
    let days = hours / 24;
    
    if days > 0 {
        format!("{}d {}h {}m", days, hours % 24, minutes % 60)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes % 60)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds % 60)
    } else {
        format!("{}s", seconds)
    }
}

/// Execute a command and return output
pub fn execute_command(cmd: &str, args: &[&str]) -> Result<CommandOutput> {
    let output = std::process::Command::new(cmd)
        .args(args)
        .output()
        .context(format!("Failed to execute command: {} {:?}", cmd, args))?;
    
    Ok(CommandOutput {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        status: output.status,
    })
}

/// Command output
#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub status: std::process::ExitStatus,
}

impl CommandOutput {
    /// Check if command was successful
    pub fn is_success(&self) -> bool {
        self.status.success()
    }
    
    /// Get exit code
    pub fn exit_code(&self) -> Option<i32> {
        self.status.code()
    }
}

/// Check if a service is running
pub fn is_service_running(service_name: &str) -> Result<bool> {
    let output = execute_command("systemctl", &["is-active", service_name])?;
    Ok(output.stdout.trim() == "active")
}

/// Start a service
pub fn start_service(service_name: &str) -> Result<()> {
    let output = execute_command("systemctl", &["start", service_name])?;
    if !output.is_success() {
        return Err(anyhow::anyhow!("Failed to start service {}: {}", service_name, output.stderr));
    }
    info!("Started service: {}", service_name);
    Ok(())
}

/// Stop a service
pub fn stop_service(service_name: &str) -> Result<()> {
    let output = execute_command("systemctl", &["stop", service_name])?;
    if !output.is_success() {
        return Err(anyhow::anyhow!("Failed to stop service {}: {}", service_name, output.stderr));
    }
    info!("Stopped service: {}", service_name);
    Ok(())
}

/// Restart a service
pub fn restart_service(service_name: &str) -> Result<()> {
    let output = execute_command("systemctl", &["restart", service_name])?;
    if !output.is_success() {
        return Err(anyhow::anyhow!("Failed to restart service {}: {}", service_name, output.stderr));
    }
    info!("Restarted service: {}", service_name);
    Ok(())
}

/// Get CPU information
pub fn get_cpu_info() -> Result<CpuInfo> {
    let output = execute_command("lscpu", &[])?;
    if !output.is_success() {
        return Err(anyhow::anyhow!("Failed to get CPU info: {}", output.stderr));
    }
    
    let mut cpu_info = CpuInfo::default();
    
    for line in output.stdout.lines() {
        if let Some((key, value)) = line.split_once(':') {
            match key.trim() {
                "Model name" => cpu_info.model = value.trim().to_string(),
                "CPU(s)" => {
                    if let Ok(count) = value.trim().parse() {
                        cpu_info.core_count = count;
                    }
                }
                "Thread(s) per core" => {
                    if let Ok(threads) = value.trim().parse() {
                        cpu_info.threads_per_core = threads;
                    }
                }
                "CPU max MHz" => {
                    if let Ok(mhz) = value.trim().parse() {
                        cpu_info.max_frequency_mhz = mhz;
                    }
                }
                _ => {}
            }
        }
    }
    
    Ok(cpu_info)
}

/// CPU information
#[derive(Debug, Clone, Default)]
pub struct CpuInfo {
    pub model: String,
    pub core_count: u32,
    pub threads_per_core: u32,
    pub max_frequency_mhz: f64,
}

/// Get memory information
pub fn get_memory_info() -> Result<MemoryInfo> {
    let output = execute_command("free", &["-h"])?;
    if !output.is_success() {
        return Err(anyhow::anyhow!("Failed to get memory info: {}", output.stderr));
    }
    
    let lines: Vec<&str> = output.stdout.lines().collect();
    if lines.len() < 2 {
        return Err(anyhow::anyhow!("Unexpected memory info format"));
    }
    
    let mem_line: Vec<&str> = lines[1].split_whitespace().collect();
    if mem_line.len() < 3 {
        return Err(anyhow::anyhow!("Unexpected memory line format"));
    }
    
    Ok(MemoryInfo {
        total: mem_line[1].to_string(),
        used: mem_line[2].to_string(),
        available: mem_line[6].to_string(),
    })
}

/// Memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total: String,
    pub used: String,
    pub available: String,
}

/// Check if a kernel module is loaded
pub fn is_kernel_module_loaded(module_name: &str) -> bool {
    let output = execute_command("lsmod", &[]);
    if let Ok(output) = output {
        output.stdout.lines().any(|line| line.starts_with(module_name))
    } else {
        false
    }
}

/// Load a kernel module
pub fn load_kernel_module(module_name: &str) -> Result<()> {
    let output = execute_command("modprobe", &[module_name])?;
    if !output.is_success() {
        return Err(anyhow::anyhow!("Failed to load kernel module {}: {}", module_name, output.stderr));
    }
    info!("Loaded kernel module: {}", module_name);
    Ok(())
}

/// Unload a kernel module
pub fn unload_kernel_module(module_name: &str) -> Result<()> {
    let output = execute_command("modprobe", &["-r", module_name])?;
    if !output.is_success() {
        return Err(anyhow::anyhow!("Failed to unload kernel module {}: {}", module_name, output.stderr));
    }
    info!("Unloaded kernel module: {}", module_name);
    Ok(())
}

/// Get disk usage information
pub fn get_disk_usage(path: &str) -> Result<DiskUsage> {
    let output = execute_command("df", &["-h", path])?;
    if !output.is_success() {
        return Err(anyhow::anyhow!("Failed to get disk usage for {}: {}", path, output.stderr));
    }
    
    let lines: Vec<&str> = output.stdout.lines().collect();
    if lines.len() < 2 {
        return Err(anyhow::anyhow!("Unexpected disk usage format"));
    }
    
    let parts: Vec<&str> = lines[1].split_whitespace().collect();
    if parts.len() < 5 {
        return Err(anyhow::anyhow!("Unexpected disk usage line format"));
    }
    
    Ok(DiskUsage {
        filesystem: parts[0].to_string(),
        total: parts[1].to_string(),
        used: parts[2].to_string(),
        available: parts[3].to_string(),
        usage_percent: parts[4].to_string(),
        mount_point: parts[5].to_string(),
    })
}

/// Disk usage information
#[derive(Debug, Clone)]
pub struct DiskUsage {
    pub filesystem: String,
    pub total: String,
    pub used: String,
    pub available: String,
    pub usage_percent: String,
    pub mount_point: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }
    
    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(std::time::Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(std::time::Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(std::time::Duration::from_secs(3661)), "1h 1m");
    }
    
    #[test]
    fn test_file_exists_and_readable() {
        // Test with a file that should exist
        assert!(file_exists_and_readable("/proc/cpuinfo"));
        
        // Test with a file that shouldn't exist
        assert!(!file_exists_and_readable("/nonexistent/file"));
    }
} 