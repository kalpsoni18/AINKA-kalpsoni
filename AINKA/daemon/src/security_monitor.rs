use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use crate::ebpf_manager::EbpfEvent;

/// Security threat types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SecurityThreatType {
    UnauthorizedAccess,
    PrivilegeEscalation,
    SuspiciousProcess,
    NetworkIntrusion,
    FileSystemTampering,
    MemoryCorruption,
    BruteForceAttack,
    DataExfiltration,
    MalwareActivity,
    Custom(String),
}

/// Security threat severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum SecuritySeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// Security event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub threat_type: SecurityThreatType,
    pub severity: SecuritySeverity,
    pub timestamp: Instant,
    pub source_ip: Option<String>,
    pub source_pid: Option<u32>,
    pub target_pid: Option<u32>,
    pub description: String,
    pub evidence: HashMap<String, String>,
    pub confidence: f64,
    pub action_taken: Option<String>,
}

/// Security rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub threat_type: SecurityThreatType,
    pub conditions: Vec<SecurityCondition>,
    pub actions: Vec<SecurityAction>,
    pub enabled: bool,
    pub priority: u32,
}

/// Security condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityCondition {
    pub field: String,
    pub operator: String,
    pub value: String,
}

/// Security action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityAction {
    Log,
    Alert,
    Block,
    KillProcess,
    Quarantine,
    Custom(String),
}

/// Security monitor configuration
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    pub enabled_monitors: Vec<String>,
    pub alert_threshold: u32,
    pub block_threshold: u32,
    pub quarantine_threshold: u32,
    pub max_events_per_minute: u32,
    pub whitelist_ips: HashSet<String>,
    pub whitelist_processes: HashSet<String>,
    pub blacklist_ips: HashSet<String>,
    pub blacklist_processes: HashSet<String>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        let mut whitelist_ips = HashSet::new();
        whitelist_ips.insert("127.0.0.1".to_string());
        whitelist_ips.insert("::1".to_string());

        let mut whitelist_processes = HashSet::new();
        whitelist_processes.insert("systemd".to_string());
        whitelist_processes.insert("init".to_string());

        Self {
            enabled_monitors: vec![
                "process_monitor".to_string(),
                "network_monitor".to_string(),
                "file_monitor".to_string(),
                "syscall_monitor".to_string(),
                "privilege_monitor".to_string(),
            ],
            alert_threshold: 5,
            block_threshold: 10,
            quarantine_threshold: 15,
            max_events_per_minute: 1000,
            whitelist_ips,
            whitelist_processes,
            blacklist_ips: HashSet::new(),
            blacklist_processes: HashSet::new(),
        }
    }
}

/// Process monitor
struct ProcessMonitor {
    suspicious_processes: HashMap<u32, ProcessInfo>,
    process_history: VecDeque<ProcessEvent>,
    max_history: usize,
}

#[derive(Debug, Clone)]
struct ProcessInfo {
    pid: u32,
    name: String,
    parent_pid: u32,
    uid: u32,
    gid: u32,
    start_time: Instant,
    suspicious_activities: Vec<String>,
    threat_score: f64,
}

#[derive(Debug, Clone)]
struct ProcessEvent {
    pid: u32,
    event_type: String,
    timestamp: Instant,
    details: HashMap<String, String>,
}

impl ProcessMonitor {
    fn new(max_history: usize) -> Self {
        Self {
            suspicious_processes: HashMap::new(),
            process_history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    fn add_event(&mut self, event: ProcessEvent) {
        if self.process_history.len() >= self.max_history {
            self.process_history.pop_front();
        }
        self.process_history.push_back(event.clone());

        // Update process info
        let process_info = self.suspicious_processes
            .entry(event.pid)
            .or_insert_with(|| ProcessInfo {
                pid: event.pid,
                name: event.details.get("name").unwrap_or(&"unknown".to_string()).clone(),
                parent_pid: event.details.get("parent_pid").unwrap_or(&"0".to_string()).parse().unwrap_or(0),
                uid: event.details.get("uid").unwrap_or(&"0".to_string()).parse().unwrap_or(0),
                gid: event.details.get("gid").unwrap_or(&"0".to_string()).parse().unwrap_or(0),
                start_time: Instant::now(),
                suspicious_activities: Vec::new(),
                threat_score: 0.0,
            });

        // Update threat score
        process_info.threat_score += self.calculate_threat_score(&event);
        process_info.suspicious_activities.push(event.event_type.clone());
    }

    fn calculate_threat_score(&self, event: &ProcessEvent) -> f64 {
        match event.event_type.as_str() {
            "privilege_escalation" => 10.0,
            "suspicious_syscall" => 5.0,
            "file_access" => 2.0,
            "network_connection" => 3.0,
            "memory_access" => 4.0,
            _ => 1.0,
        }
    }

    fn get_suspicious_processes(&self) -> Vec<&ProcessInfo> {
        self.suspicious_processes.values()
            .filter(|p| p.threat_score > 5.0)
            .collect()
    }
}

/// Network monitor
struct NetworkMonitor {
    connections: HashMap<String, ConnectionInfo>,
    connection_history: VecDeque<NetworkEvent>,
    max_history: usize,
}

#[derive(Debug, Clone)]
struct ConnectionInfo {
    source_ip: String,
    dest_ip: String,
    source_port: u16,
    dest_port: u16,
    protocol: String,
    start_time: Instant,
    bytes_sent: u64,
    bytes_received: u64,
    threat_score: f64,
}

#[derive(Debug, Clone)]
struct NetworkEvent {
    source_ip: String,
    dest_ip: String,
    source_port: u16,
    dest_port: u16,
    protocol: String,
    event_type: String,
    timestamp: Instant,
    details: HashMap<String, String>,
}

impl NetworkMonitor {
    fn new(max_history: usize) -> Self {
        Self {
            connections: HashMap::new(),
            connection_history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    fn add_event(&mut self, event: NetworkEvent) {
        if self.connection_history.len() >= self.max_history {
            self.connection_history.pop_front();
        }
        self.connection_history.push_back(event.clone());

        let key = format!("{}:{}->{}:{}", event.source_ip, event.source_port, event.dest_ip, event.dest_port);
        
        let connection_info = self.connections
            .entry(key.clone())
            .or_insert_with(|| ConnectionInfo {
                source_ip: event.source_ip.clone(),
                dest_ip: event.dest_ip.clone(),
                source_port: event.source_port,
                dest_port: event.dest_port,
                protocol: event.protocol.clone(),
                start_time: Instant::now(),
                bytes_sent: 0,
                bytes_received: 0,
                threat_score: 0.0,
            });

        // Update connection info
        if let Some(bytes) = event.details.get("bytes_sent") {
            connection_info.bytes_sent += bytes.parse().unwrap_or(0);
        }
        if let Some(bytes) = event.details.get("bytes_received") {
            connection_info.bytes_received += bytes.parse().unwrap_or(0);
        }

        // Update threat score
        connection_info.threat_score += self.calculate_threat_score(&event);
    }

    fn calculate_threat_score(&self, event: &NetworkEvent) -> f64 {
        match event.event_type.as_str() {
            "connection_attempt" => 1.0,
            "data_transfer" => 2.0,
            "suspicious_payload" => 10.0,
            "port_scan" => 8.0,
            "brute_force" => 15.0,
            _ => 1.0,
        }
    }

    fn get_suspicious_connections(&self) -> Vec<&ConnectionInfo> {
        self.connections.values()
            .filter(|c| c.threat_score > 5.0)
            .collect()
    }
}

/// File system monitor
struct FileSystemMonitor {
    file_events: VecDeque<FileEvent>,
    suspicious_files: HashMap<String, FileInfo>,
    max_history: usize,
}

#[derive(Debug, Clone)]
struct FileEvent {
    path: String,
    operation: String,
    pid: u32,
    timestamp: Instant,
    details: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct FileInfo {
    path: String,
    size: u64,
    permissions: String,
    owner: String,
    group: String,
    last_modified: Instant,
    threat_score: f64,
    suspicious_operations: Vec<String>,
}

impl FileSystemMonitor {
    fn new(max_history: usize) -> Self {
        Self {
            file_events: VecDeque::with_capacity(max_history),
            suspicious_files: HashMap::new(),
            max_history,
        }
    }

    fn add_event(&mut self, event: FileEvent) {
        if self.file_events.len() >= self.max_history {
            self.file_events.pop_front();
        }
        self.file_events.push_back(event.clone());

        let file_info = self.suspicious_files
            .entry(event.path.clone())
            .or_insert_with(|| FileInfo {
                path: event.path.clone(),
                size: event.details.get("size").unwrap_or(&"0".to_string()).parse().unwrap_or(0),
                permissions: event.details.get("permissions").unwrap_or(&"".to_string()).clone(),
                owner: event.details.get("owner").unwrap_or(&"".to_string()).clone(),
                group: event.details.get("group").unwrap_or(&"".to_string()).clone(),
                last_modified: Instant::now(),
                threat_score: 0.0,
                suspicious_operations: Vec::new(),
            });

        // Update threat score
        file_info.threat_score += self.calculate_threat_score(&event);
        file_info.suspicious_operations.push(event.operation.clone());
    }

    fn calculate_threat_score(&self, event: &FileEvent) -> f64 {
        match event.operation.as_str() {
            "delete" => 3.0,
            "modify" => 2.0,
            "create" => 1.0,
            "execute" => 5.0,
            "privileged_access" => 8.0,
            _ => 1.0,
        }
    }

    fn get_suspicious_files(&self) -> Vec<&FileInfo> {
        self.suspicious_files.values()
            .filter(|f| f.threat_score > 3.0)
            .collect()
    }
}

/// Main security monitor
pub struct SecurityMonitor {
    config: SecurityConfig,
    process_monitor: ProcessMonitor,
    network_monitor: NetworkMonitor,
    file_monitor: FileSystemMonitor,
    rules: Vec<SecurityRule>,
    event_sender: mpsc::UnboundedSender<SecurityEvent>,
    threat_counts: HashMap<SecurityThreatType, u32>,
    last_reset: Instant,
}

impl SecurityMonitor {
    /// Create a new security monitor
    pub fn new(config: SecurityConfig, event_sender: mpsc::UnboundedSender<SecurityEvent>) -> Self {
        let process_monitor = ProcessMonitor::new(1000);
        let network_monitor = NetworkMonitor::new(1000);
        let file_monitor = FileSystemMonitor::new(1000);
        let rules = Self::load_default_rules();
        let threat_counts = HashMap::new();

        Self {
            config,
            process_monitor,
            network_monitor,
            file_monitor,
            rules,
            event_sender,
            threat_counts,
            last_reset: Instant::now(),
        }
    }

    /// Process eBPF events
    pub fn process_ebpf_event(&mut self, event: &EbpfEvent) -> Result<(), Box<dyn std::error::Error>> {
        // Reset counters if needed
        if self.last_reset.elapsed() >= Duration::from_secs(60) {
            self.threat_counts.clear();
            self.last_reset = Instant::now();
        }

        match event.event_type {
            crate::ebpf_manager::EbpfEventType::Syscall => {
                self.handle_syscall_event(event);
            }
            crate::ebpf_manager::EbpfEventType::Network => {
                self.handle_network_event(event);
            }
            crate::ebpf_manager::EbpfEventType::Security => {
                self.handle_security_event(event);
            }
            _ => {
                // Handle other event types
            }
        }

        Ok(())
    }

    /// Handle syscall events
    fn handle_syscall_event(&mut self, event: &EbpfEvent) {
        let syscall_name = event.metadata.get("syscall_name").unwrap_or(&"unknown".to_string());
        
        // Check for suspicious syscalls
        let suspicious_syscalls = [
            "execve", "fork", "clone", "ptrace", "setuid", "setgid",
            "chmod", "chown", "unlink", "rmdir", "rename"
        ];

        if suspicious_syscalls.contains(&syscall_name.as_str()) {
            let process_event = ProcessEvent {
                pid: event.pid,
                event_type: "suspicious_syscall".to_string(),
                timestamp: Instant::now(),
                details: {
                    let mut details = HashMap::new();
                    details.insert("syscall".to_string(), syscall_name.clone());
                    details.insert("name".to_string(), event.metadata.get("process_name").unwrap_or(&"unknown".to_string()).clone());
                    details
                },
            };

            self.process_monitor.add_event(process_event);

            // Check if this triggers any security rules
            self.check_security_rules(event);
        }
    }

    /// Handle network events
    fn handle_network_event(&mut self, event: &EbpfEvent) {
        if let Some(source_ip) = event.metadata.get("source_ip") {
            if let Some(dest_ip) = event.metadata.get("dest_ip") {
                let network_event = NetworkEvent {
                    source_ip: source_ip.clone(),
                    dest_ip: dest_ip.clone(),
                    source_port: event.metadata.get("source_port").unwrap_or(&"0".to_string()).parse().unwrap_or(0),
                    dest_port: event.metadata.get("dest_port").unwrap_or(&"0".to_string()).parse().unwrap_or(0),
                    protocol: event.metadata.get("protocol").unwrap_or(&"unknown".to_string()).clone(),
                    event_type: "connection_attempt".to_string(),
                    timestamp: Instant::now(),
                    details: event.metadata.clone(),
                };

                self.network_monitor.add_event(network_event);

                // Check for blacklisted IPs
                if self.config.blacklist_ips.contains(source_ip) {
                    self.create_security_event(
                        SecurityThreatType::NetworkIntrusion,
                        SecuritySeverity::High,
                        format!("Connection attempt from blacklisted IP: {}", source_ip),
                        Some(source_ip.clone()),
                        Some(event.pid),
                        None,
                        event.metadata.clone(),
                        0.9,
                        Some("block".to_string()),
                    );
                }
            }
        }
    }

    /// Handle security events
    fn handle_security_event(&mut self, event: &EbpfEvent) {
        // Process security-specific events
        if let Some(threat_type) = event.metadata.get("threat_type") {
            match threat_type.as_str() {
                "privilege_escalation" => {
                    self.create_security_event(
                        SecurityThreatType::PrivilegeEscalation,
                        SecuritySeverity::Critical,
                        "Privilege escalation detected".to_string(),
                        None,
                        Some(event.pid),
                        None,
                        event.metadata.clone(),
                        0.8,
                        Some("alert".to_string()),
                    );
                }
                "suspicious_process" => {
                    self.create_security_event(
                        SecurityThreatType::SuspiciousProcess,
                        SecuritySeverity::Medium,
                        "Suspicious process activity detected".to_string(),
                        None,
                        Some(event.pid),
                        None,
                        event.metadata.clone(),
                        0.7,
                        Some("log".to_string()),
                    );
                }
                _ => {}
            }
        }
    }

    /// Check security rules
    fn check_security_rules(&mut self, event: &EbpfEvent) {
        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }

            if self.evaluate_rule(rule, event) {
                self.execute_rule_actions(rule, event);
            }
        }
    }

    /// Evaluate a security rule
    fn evaluate_rule(&self, rule: &SecurityRule, event: &EbpfEvent) -> bool {
        for condition in &rule.conditions {
            if !self.evaluate_condition(condition, event) {
                return false;
            }
        }
        true
    }

    /// Evaluate a security condition
    fn evaluate_condition(&self, condition: &SecurityCondition, event: &EbpfEvent) -> bool {
        let value = match condition.field.as_str() {
            "syscall" => event.metadata.get("syscall_name"),
            "pid" => Some(&event.pid.to_string()),
            "uid" => event.metadata.get("uid"),
            "gid" => event.metadata.get("gid"),
            _ => None,
        };

        if let Some(actual_value) = value {
            match condition.operator.as_str() {
                "equals" => actual_value == &condition.value,
                "contains" => actual_value.contains(&condition.value),
                "starts_with" => actual_value.starts_with(&condition.value),
                "ends_with" => actual_value.ends_with(&condition.value),
                _ => false,
            }
        } else {
            false
        }
    }

    /// Execute rule actions
    fn execute_rule_actions(&mut self, rule: &SecurityRule, event: &EbpfEvent) {
        for action in &rule.actions {
            match action {
                SecurityAction::Log => {
                    log::info!("Security rule triggered: {} - {}", rule.name, rule.description);
                }
                SecurityAction::Alert => {
                    self.create_security_event(
                        rule.threat_type.clone(),
                        SecuritySeverity::Medium,
                        format!("Security rule triggered: {}", rule.name),
                        None,
                        Some(event.pid),
                        None,
                        event.metadata.clone(),
                        0.8,
                        Some("alert".to_string()),
                    );
                }
                SecurityAction::Block => {
                    // Implement blocking logic
                    log::warn!("Blocking action triggered for rule: {}", rule.name);
                }
                SecurityAction::KillProcess => {
                    // Implement process killing logic
                    log::warn!("Kill process action triggered for PID: {}", event.pid);
                }
                SecurityAction::Quarantine => {
                    // Implement quarantine logic
                    log::warn!("Quarantine action triggered for PID: {}", event.pid);
                }
                SecurityAction::Custom(action_name) => {
                    log::info!("Custom action '{}' triggered for rule: {}", action_name, rule.name);
                }
            }
        }
    }

    /// Create a security event
    fn create_security_event(
        &mut self,
        threat_type: SecurityThreatType,
        severity: SecuritySeverity,
        description: String,
        source_ip: Option<String>,
        source_pid: Option<u32>,
        target_pid: Option<u32>,
        evidence: HashMap<String, String>,
        confidence: f64,
        action_taken: Option<String>,
    ) {
        // Update threat count
        *self.threat_counts.entry(threat_type.clone()).or_insert(0) += 1;

        let security_event = SecurityEvent {
            threat_type,
            severity,
            timestamp: Instant::now(),
            source_ip,
            source_pid,
            target_pid,
            description,
            evidence,
            confidence,
            action_taken,
        };

        if let Err(e) = self.event_sender.send(security_event) {
            log::error!("Failed to send security event: {}", e);
        }
    }

    /// Load default security rules
    fn load_default_rules() -> Vec<SecurityRule> {
        vec![
            SecurityRule {
                id: "rule_001".to_string(),
                name: "Privilege Escalation Detection".to_string(),
                description: "Detect attempts to escalate privileges".to_string(),
                threat_type: SecurityThreatType::PrivilegeEscalation,
                conditions: vec![
                    SecurityCondition {
                        field: "syscall".to_string(),
                        operator: "equals".to_string(),
                        value: "setuid".to_string(),
                    },
                ],
                actions: vec![SecurityAction::Alert, SecurityAction::Log],
                enabled: true,
                priority: 1,
            },
            SecurityRule {
                id: "rule_002".to_string(),
                name: "Suspicious Process Creation".to_string(),
                description: "Detect suspicious process creation".to_string(),
                threat_type: SecurityThreatType::SuspiciousProcess,
                conditions: vec![
                    SecurityCondition {
                        field: "syscall".to_string(),
                        operator: "equals".to_string(),
                        value: "execve".to_string(),
                    },
                ],
                actions: vec![SecurityAction::Log],
                enabled: true,
                priority: 2,
            },
        ]
    }

    /// Get security statistics
    pub fn get_stats(&self) -> HashMap<String, u32> {
        let mut stats = HashMap::new();
        
        for (threat_type, count) in &self.threat_counts {
            stats.insert(format!("{:?}", threat_type), *count);
        }
        
        stats.insert("suspicious_processes".to_string(), self.process_monitor.get_suspicious_processes().len() as u32);
        stats.insert("suspicious_connections".to_string(), self.network_monitor.get_suspicious_connections().len() as u32);
        stats.insert("suspicious_files".to_string(), self.file_monitor.get_suspicious_files().len() as u32);
        
        stats
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SecurityConfig) {
        self.config = config;
    }

    /// Add a custom security rule
    pub fn add_rule(&mut self, rule: SecurityRule) {
        self.rules.push(rule);
    }

    /// Get all security rules
    pub fn get_rules(&self) -> &Vec<SecurityRule> {
        &self.rules
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn test_security_monitor_creation() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let config = SecurityConfig::default();
        
        let monitor = SecurityMonitor::new(config, tx);
        assert!(!monitor.rules.is_empty());
        assert!(monitor.threat_counts.is_empty());
    }

    #[test]
    fn test_process_monitor() {
        let mut monitor = ProcessMonitor::new(100);
        
        let event = ProcessEvent {
            pid: 1234,
            event_type: "suspicious_syscall".to_string(),
            timestamp: Instant::now(),
            details: {
                let mut details = HashMap::new();
                details.insert("syscall".to_string(), "setuid".to_string());
                details.insert("name".to_string(), "test_process".to_string());
                details
            },
        };
        
        monitor.add_event(event);
        let suspicious = monitor.get_suspicious_processes();
        assert!(!suspicious.is_empty());
    }

    #[test]
    fn test_network_monitor() {
        let mut monitor = NetworkMonitor::new(100);
        
        let event = NetworkEvent {
            source_ip: "192.168.1.1".to_string(),
            dest_ip: "10.0.0.1".to_string(),
            source_port: 12345,
            dest_port: 80,
            protocol: "tcp".to_string(),
            event_type: "connection_attempt".to_string(),
            timestamp: Instant::now(),
            details: HashMap::new(),
        };
        
        monitor.add_event(event);
        let suspicious = monitor.get_suspicious_connections();
        assert!(suspicious.is_empty()); // Should be empty for normal connection
    }
} 