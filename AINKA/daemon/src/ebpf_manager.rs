use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use serde::{Deserialize, Serialize};
use libbpf_rs::{
    Object, Program, Link, RingBuffer, RingBufferBuilder,
    Map, MapType, PerfEvent, PerfEventBuilder
};

/// eBPF event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EbpfEventType {
    Syscall,
    Memory,
    Performance,
    Security,
    Network,
    Io,
    Scheduler,
    Custom(String),
}

/// eBPF event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EbpfEvent {
    pub event_type: EbpfEventType,
    pub timestamp: u64,
    pub pid: u32,
    pub tid: u32,
    pub cpu: u32,
    pub data: HashMap<String, f64>,
    pub metadata: HashMap<String, String>,
}

/// eBPF program configuration
#[derive(Debug, Clone)]
pub struct EbpfConfig {
    pub enabled_programs: Vec<String>,
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub max_events_per_second: u32,
}

impl Default for EbpfConfig {
    fn default() -> Self {
        Self {
            enabled_programs: vec![
                "syscall_monitor".to_string(),
                "memory_monitor".to_string(),
                "performance_monitor".to_string(),
                "network_monitor".to_string(),
                "io_monitor".to_string(),
            ],
            sample_rate: 1000, // 1 in 1000 events
            buffer_size: 1024 * 1024, // 1MB
            max_events_per_second: 10000,
        }
    }
}

/// eBPF program statistics
#[derive(Debug, Clone)]
pub struct EbpfStats {
    pub events_processed: u64,
    pub events_dropped: u64,
    pub programs_loaded: u32,
    pub programs_active: u32,
    pub last_event_time: Option<Instant>,
    pub uptime: Duration,
}

/// eBPF Manager for loading and managing eBPF programs
pub struct EbpfManager {
    config: EbpfConfig,
    stats: Arc<Mutex<EbpfStats>>,
    programs: HashMap<String, Program>,
    links: HashMap<String, Link>,
    ring_buffers: HashMap<String, RingBuffer>,
    event_sender: mpsc::UnboundedSender<EbpfEvent>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

/// Rate limiter for eBPF events
struct RateLimiter {
    last_event: Instant,
    event_count: u32,
    max_events_per_second: u32,
}

impl RateLimiter {
    fn new(max_events_per_second: u32) -> Self {
        Self {
            last_event: Instant::now(),
            event_count: 0,
            max_events_per_second,
        }
    }

    fn should_process_event(&mut self) -> bool {
        let now = Instant::now();
        
        if now.duration_since(self.last_event) >= Duration::from_secs(1) {
            self.event_count = 0;
            self.last_event = now;
        }
        
        if self.event_count < self.max_events_per_second {
            self.event_count += 1;
            true
        } else {
            false
        }
    }
}

impl EbpfManager {
    /// Create a new eBPF manager
    pub fn new(config: EbpfConfig, event_sender: mpsc::UnboundedSender<EbpfEvent>) -> Result<Self, Box<dyn std::error::Error>> {
        let stats = Arc::new(Mutex::new(EbpfStats {
            events_processed: 0,
            events_dropped: 0,
            programs_loaded: 0,
            programs_active: 0,
            last_event_time: None,
            uptime: Duration::from_secs(0),
        }));

        let rate_limiter = Arc::new(Mutex::new(RateLimiter::new(config.max_events_per_second)));

        Ok(Self {
            config,
            stats,
            programs: HashMap::new(),
            links: HashMap::new(),
            ring_buffers: HashMap::new(),
            event_sender,
            rate_limiter,
        })
    }

    /// Load eBPF programs from object file
    pub fn load_programs(&mut self, object_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        log::info!("Loading eBPF programs from: {}", object_path);
        
        let obj = Object::open_file(object_path)?;
        obj.load()?;

        // Load individual programs
        for program_name in &self.config.enabled_programs {
            if let Some(program) = obj.program(program_name) {
                match self.load_program(program_name, program) {
                    Ok(_) => {
                        log::info!("Successfully loaded eBPF program: {}", program_name);
                        self.stats.lock().unwrap().programs_loaded += 1;
                    }
                    Err(e) => {
                        log::warn!("Failed to load eBPF program {}: {}", program_name, e);
                    }
                }
            } else {
                log::warn!("eBPF program not found: {}", program_name);
            }
        }

        Ok(())
    }

    /// Load a specific eBPF program
    fn load_program(&mut self, name: &str, program: Program) -> Result<(), Box<dyn std::error::Error>> {
        let program = program.load()?;
        
        // Attach program based on type
        let link = match name {
            "syscall_monitor" => self.attach_syscall_monitor(program)?,
            "memory_monitor" => self.attach_memory_monitor(program)?,
            "performance_monitor" => self.attach_performance_monitor(program)?,
            "network_monitor" => self.attach_network_monitor(program)?,
            "io_monitor" => self.attach_io_monitor(program)?,
            "scheduler_monitor" => self.attach_scheduler_monitor(program)?,
            _ => {
                log::warn!("Unknown eBPF program type: {}", name);
                return Ok(());
            }
        };

        self.programs.insert(name.to_string(), program);
        self.links.insert(name.to_string(), link);
        self.stats.lock().unwrap().programs_active += 1;

        Ok(())
    }

    /// Attach syscall monitor
    fn attach_syscall_monitor(&self, program: Program) -> Result<Link, Box<dyn std::error::Error>> {
        let link = program.attach_tracepoint("syscalls", "sys_enter_execve")?;
        Ok(link)
    }

    /// Attach memory monitor
    fn attach_memory_monitor(&self, program: Program) -> Result<Link, Box<dyn std::error::Error>> {
        let link = program.attach_tracepoint("kmem", "kmalloc")?;
        Ok(link)
    }

    /// Attach performance monitor
    fn attach_performance_monitor(&self, program: Program) -> Result<Link, Box<dyn std::error::Error>> {
        let link = program.attach_tracepoint("sched", "sched_switch")?;
        Ok(link)
    }

    /// Attach network monitor
    fn attach_network_monitor(&self, program: Program) -> Result<Link, Box<dyn std::error::Error>> {
        let link = program.attach_tracepoint("net", "netif_receive_skb")?;
        Ok(link)
    }

    /// Attach I/O monitor
    fn attach_io_monitor(&self, program: Program) -> Result<Link, Box<dyn std::error::Error>> {
        let link = program.attach_tracepoint("block", "block_rq_issue")?;
        Ok(link)
    }

    /// Attach scheduler monitor
    fn attach_scheduler_monitor(&self, program: Program) -> Result<Link, Box<dyn std::error::Error>> {
        let link = program.attach_tracepoint("sched", "sched_wakeup")?;
        Ok(link)
    }

    /// Setup ring buffers for event collection
    pub fn setup_ring_buffers(&mut self, obj: &Object) -> Result<(), Box<dyn std::error::Error>> {
        let mut builder = RingBufferBuilder::new();
        
        // Add ring buffers for different event types
        if let Some(map) = obj.map("syscall_events") {
            builder.add(map, Box::new(|data| {
                self.handle_syscall_event(data);
            }))?;
        }

        if let Some(map) = obj.map("memory_events") {
            builder.add(map, Box::new(|data| {
                self.handle_memory_event(data);
            }))?;
        }

        if let Some(map) = obj.map("performance_events") {
            builder.add(map, Box::new(|data| {
                self.handle_performance_event(data);
            }))?;
        }

        if let Some(map) = obj.map("network_events") {
            builder.add(map, Box::new(|data| {
                self.handle_network_event(data);
            }))?;
        }

        if let Some(map) = obj.map("io_events") {
            builder.add(map, Box::new(|data| {
                self.handle_io_event(data);
            }))?;
        }

        let ring_buffer = builder.build()?;
        self.ring_buffers.insert("main".to_string(), ring_buffer);

        Ok(())
    }

    /// Handle syscall events
    fn handle_syscall_event(&self, data: &[u8]) {
        if !self.rate_limiter.lock().unwrap().should_process_event() {
            self.stats.lock().unwrap().events_dropped += 1;
            return;
        }

        // Parse syscall event data
        if data.len() >= 16 {
            let pid = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
            let syscall = u32::from_ne_bytes([data[4], data[5], data[6], data[7]]);
            let timestamp = u64::from_ne_bytes([
                data[8], data[9], data[10], data[11],
                data[12], data[13], data[14], data[15]
            ]);

            let mut event_data = HashMap::new();
            event_data.insert("syscall_number".to_string(), syscall as f64);
            
            let mut metadata = HashMap::new();
            metadata.insert("syscall_name".to_string(), self.get_syscall_name(syscall));

            let event = EbpfEvent {
                event_type: EbpfEventType::Syscall,
                timestamp,
                pid,
                tid: pid,
                cpu: 0,
                data: event_data,
                metadata,
            };

            if let Err(e) = self.event_sender.send(event) {
                log::error!("Failed to send syscall event: {}", e);
            } else {
                self.stats.lock().unwrap().events_processed += 1;
                self.stats.lock().unwrap().last_event_time = Some(Instant::now());
            }
        }
    }

    /// Handle memory events
    fn handle_memory_event(&self, data: &[u8]) {
        if !self.rate_limiter.lock().unwrap().should_process_event() {
            self.stats.lock().unwrap().events_dropped += 1;
            return;
        }

        if data.len() >= 20 {
            let pid = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
            let size = u64::from_ne_bytes([
                data[4], data[5], data[6], data[7],
                data[8], data[9], data[10], data[11]
            ]);
            let timestamp = u64::from_ne_bytes([
                data[12], data[13], data[14], data[15],
                data[16], data[17], data[18], data[19]
            ]);

            let mut event_data = HashMap::new();
            event_data.insert("allocation_size".to_string(), size as f64);

            let event = EbpfEvent {
                event_type: EbpfEventType::Memory,
                timestamp,
                pid,
                tid: pid,
                cpu: 0,
                data: event_data,
                metadata: HashMap::new(),
            };

            if let Err(e) = self.event_sender.send(event) {
                log::error!("Failed to send memory event: {}", e);
            } else {
                self.stats.lock().unwrap().events_processed += 1;
                self.stats.lock().unwrap().last_event_time = Some(Instant::now());
            }
        }
    }

    /// Handle performance events
    fn handle_performance_event(&self, data: &[u8]) {
        if !self.rate_limiter.lock().unwrap().should_process_event() {
            self.stats.lock().unwrap().events_dropped += 1;
            return;
        }

        if data.len() >= 24 {
            let prev_pid = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
            let next_pid = u32::from_ne_bytes([data[4], data[5], data[6], data[7]]);
            let cpu = u32::from_ne_bytes([data[8], data[9], data[10], data[11]]);
            let timestamp = u64::from_ne_bytes([
                data[12], data[13], data[14], data[15],
                data[16], data[17], data[18], data[19],
                data[20], data[21], data[22], data[23]
            ]);

            let mut event_data = HashMap::new();
            event_data.insert("prev_pid".to_string(), prev_pid as f64);
            event_data.insert("next_pid".to_string(), next_pid as f64);
            event_data.insert("cpu".to_string(), cpu as f64);

            let event = EbpfEvent {
                event_type: EbpfEventType::Performance,
                timestamp,
                pid: next_pid,
                tid: next_pid,
                cpu,
                data: event_data,
                metadata: HashMap::new(),
            };

            if let Err(e) = self.event_sender.send(event) {
                log::error!("Failed to send performance event: {}", e);
            } else {
                self.stats.lock().unwrap().events_processed += 1;
                self.stats.lock().unwrap().last_event_time = Some(Instant::now());
            }
        }
    }

    /// Handle network events
    fn handle_network_event(&self, data: &[u8]) {
        if !self.rate_limiter.lock().unwrap().should_process_event() {
            self.stats.lock().unwrap().events_dropped += 1;
            return;
        }

        if data.len() >= 16 {
            let pid = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
            let packet_size = u32::from_ne_bytes([data[4], data[5], data[6], data[7]]);
            let timestamp = u64::from_ne_bytes([
                data[8], data[9], data[10], data[11],
                data[12], data[13], data[14], data[15]
            ]);

            let mut event_data = HashMap::new();
            event_data.insert("packet_size".to_string(), packet_size as f64);

            let event = EbpfEvent {
                event_type: EbpfEventType::Network,
                timestamp,
                pid,
                tid: pid,
                cpu: 0,
                data: event_data,
                metadata: HashMap::new(),
            };

            if let Err(e) = self.event_sender.send(event) {
                log::error!("Failed to send network event: {}", e);
            } else {
                self.stats.lock().unwrap().events_processed += 1;
                self.stats.lock().unwrap().last_event_time = Some(Instant::now());
            }
        }
    }

    /// Handle I/O events
    fn handle_io_event(&self, data: &[u8]) {
        if !self.rate_limiter.lock().unwrap().should_process_event() {
            self.stats.lock().unwrap().events_dropped += 1;
            return;
        }

        if data.len() >= 20 {
            let pid = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
            let sector = u64::from_ne_bytes([
                data[4], data[5], data[6], data[7],
                data[8], data[9], data[10], data[11]
            ]);
            let nr_sectors = u32::from_ne_bytes([data[12], data[13], data[14], data[15]]);
            let timestamp = u64::from_ne_bytes([
                data[16], data[17], data[18], data[19]
            ]);

            let mut event_data = HashMap::new();
            event_data.insert("sector".to_string(), sector as f64);
            event_data.insert("nr_sectors".to_string(), nr_sectors as f64);

            let event = EbpfEvent {
                event_type: EbpfEventType::Io,
                timestamp,
                pid,
                tid: pid,
                cpu: 0,
                data: event_data,
                metadata: HashMap::new(),
            };

            if let Err(e) = self.event_sender.send(event) {
                log::error!("Failed to send I/O event: {}", e);
            } else {
                self.stats.lock().unwrap().events_processed += 1;
                self.stats.lock().unwrap().last_event_time = Some(Instant::now());
            }
        }
    }

    /// Get syscall name from number
    fn get_syscall_name(&self, syscall: u32) -> String {
        match syscall {
            1 => "exit".to_string(),
            2 => "fork".to_string(),
            3 => "read".to_string(),
            4 => "write".to_string(),
            5 => "open".to_string(),
            6 => "close".to_string(),
            7 => "waitpid".to_string(),
            8 => "creat".to_string(),
            9 => "link".to_string(),
            10 => "unlink".to_string(),
            11 => "execve".to_string(),
            12 => "chdir".to_string(),
            13 => "time".to_string(),
            14 => "mknod".to_string(),
            15 => "chmod".to_string(),
            16 => "lchown".to_string(),
            17 => "break".to_string(),
            18 => "oldstat".to_string(),
            19 => "lseek".to_string(),
            20 => "getpid".to_string(),
            _ => format!("syscall_{}", syscall),
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> EbpfStats {
        let mut stats = self.stats.lock().unwrap().clone();
        stats.uptime = stats.last_event_time
            .map(|t| t.elapsed())
            .unwrap_or(Duration::from_secs(0));
        stats
    }

    /// Update configuration
    pub fn update_config(&mut self, config: EbpfConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.config = config;
        
        // Update rate limiter
        *self.rate_limiter.lock().unwrap() = RateLimiter::new(self.config.max_events_per_second);
        
        Ok(())
    }

    /// Poll ring buffers
    pub fn poll_ring_buffers(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for (name, ring_buffer) in &mut self.ring_buffers {
            if let Err(e) = ring_buffer.poll(Duration::from_millis(100)) {
                log::warn!("Error polling ring buffer {}: {}", name, e);
            }
        }
        Ok(())
    }

    /// Cleanup resources
    pub fn cleanup(&mut self) {
        log::info!("Cleaning up eBPF manager");
        
        // Remove links
        self.links.clear();
        
        // Remove programs
        self.programs.clear();
        
        // Remove ring buffers
        self.ring_buffers.clear();
        
        log::info!("eBPF manager cleanup completed");
    }
}

impl Drop for EbpfManager {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn test_ebpf_manager_creation() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let config = EbpfConfig::default();
        
        let manager = EbpfManager::new(config, tx);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(10);
        
        // Should allow first 10 events
        for _ in 0..10 {
            assert!(limiter.should_process_event());
        }
        
        // Should block 11th event
        assert!(!limiter.should_process_event());
    }
} 