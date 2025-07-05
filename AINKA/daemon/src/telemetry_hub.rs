/*
 * AINKA Telemetry Hub
 * 
 * This module implements the telemetry hub for the AI-Native kernel
 * assistant, providing metrics collection, log analysis, history
 * management, and alerting capabilities.
 * 
 * Copyright (C) 2024 AINKA Community
 * Licensed under Apache 2.0
 */

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use tokio::sync::mpsc;
use crate::AinkaError;

/// Telemetry event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    HighCpuUsage,
    HighMemoryUsage,
    HighLoadAverage,
    HighDiskIO,
    NetworkErrors,
    HighTemperature,
    ServiceFailure,
    PerformanceOptimization,
    SystemBoot,
    SystemShutdown,
    AnomalyDetected,
    SecurityAlert,
    Custom(String),
}

/// Telemetry event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Event type
    pub event_type: EventType,
    
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Event data (JSON)
    pub data: serde_json::Value,
    
    /// Event severity
    pub severity: Option<EventSeverity>,
    
    /// Source component
    pub source: Option<String>,
    
    /// Event ID
    pub id: Option<String>,
}

/// Event severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// Telemetry hub for collecting and processing events
pub struct TelemetryHub {
    /// Event queue
    events: Arc<Mutex<VecDeque<TelemetryEvent>>>,
    
    /// Maximum number of events to keep in memory
    max_events: usize,
    
    /// Event sender
    event_tx: mpsc::UnboundedSender<TelemetryEvent>,
    
    /// Event receiver
    event_rx: Option<mpsc::UnboundedReceiver<TelemetryEvent>>,
    
    /// Event processors
    processors: Vec<Box<dyn EventProcessor + Send + Sync>>,
    
    /// Statistics
    stats: TelemetryStats,
}

/// Event processor trait
pub trait EventProcessor {
    /// Process an event
    fn process(&self, event: &TelemetryEvent) -> Result<()>;
    
    /// Get processor name
    fn name(&self) -> &str;
}

/// Telemetry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryStats {
    /// Total events processed
    pub total_events: u64,
    
    /// Events by type
    pub events_by_type: std::collections::HashMap<String, u64>,
    
    /// Events by severity
    pub events_by_severity: std::collections::HashMap<String, u64>,
    
    /// Last event timestamp
    pub last_event: Option<DateTime<Utc>>,
    
    /// Average processing time
    pub avg_processing_time_ms: f64,
}

impl Default for TelemetryStats {
    fn default() -> Self {
        Self {
            total_events: 0,
            events_by_type: std::collections::HashMap::new(),
            events_by_severity: std::collections::HashMap::new(),
            last_event: None,
            avg_processing_time_ms: 0.0,
        }
    }
}

impl TelemetryHub {
    /// Create a new telemetry hub
    pub fn new(max_events: usize) -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        
        Self {
            events: Arc::new(Mutex::new(VecDeque::new())),
            max_events,
            event_tx,
            event_rx: Some(event_rx),
            processors: Vec::new(),
            stats: TelemetryStats::default(),
        }
    }
    
    /// Get event sender
    pub fn get_sender(&self) -> mpsc::UnboundedSender<TelemetryEvent> {
        self.event_tx.clone()
    }
    
    /// Add event processor
    pub fn add_processor(&mut self, processor: Box<dyn EventProcessor + Send + Sync>) {
        self.processors.push(processor);
    }
    
    /// Start processing events
    pub async fn start_processing(&mut self) -> Result<()> {
        log::info!("Starting telemetry hub event processing");
        
        if let Some(mut event_rx) = self.event_rx.take() {
            let events = Arc::clone(&self.events);
            let processors = self.processors.clone();
            let mut stats = self.stats.clone();
            
            tokio::spawn(async move {
                while let Some(event) = event_rx.recv().await {
                    let start_time = std::time::Instant::now();
                    
                    // Add to event queue
                    {
                        let mut events_guard = events.lock().unwrap();
                        events_guard.push_back(event.clone());
                        
                        // Remove old events if queue is full
                        while events_guard.len() > 1000 {
                            events_guard.pop_front();
                        }
                    }
                    
                    // Process event
                    for processor in &processors {
                        if let Err(e) = processor.process(&event) {
                            log::error!("Processor {} failed to process event: {}", processor.name(), e);
                        }
                    }
                    
                    // Update statistics
                    stats.total_events += 1;
                    stats.last_event = Some(event.timestamp);
                    
                    let event_type = match &event.event_type {
                        EventType::HighCpuUsage => "HighCpuUsage",
                        EventType::HighMemoryUsage => "HighMemoryUsage",
                        EventType::HighLoadAverage => "HighLoadAverage",
                        EventType::HighDiskIO => "HighDiskIO",
                        EventType::NetworkErrors => "NetworkErrors",
                        EventType::HighTemperature => "HighTemperature",
                        EventType::ServiceFailure => "ServiceFailure",
                        EventType::PerformanceOptimization => "PerformanceOptimization",
                        EventType::SystemBoot => "SystemBoot",
                        EventType::SystemShutdown => "SystemShutdown",
                        EventType::AnomalyDetected => "AnomalyDetected",
                        EventType::SecurityAlert => "SecurityAlert",
                        EventType::Custom(ref name) => name,
                    };
                    
                    *stats.events_by_type.entry(event_type.to_string()).or_insert(0) += 1;
                    
                    if let Some(severity) = &event.severity {
                        let severity_str = match severity {
                            EventSeverity::Debug => "Debug",
                            EventSeverity::Info => "Info",
                            EventSeverity::Warning => "Warning",
                            EventSeverity::Error => "Error",
                            EventSeverity::Critical => "Critical",
                        };
                        *stats.events_by_severity.entry(severity_str.to_string()).or_insert(0) += 1;
                    }
                    
                    let processing_time = start_time.elapsed().as_millis() as f64;
                    stats.avg_processing_time_ms = (stats.avg_processing_time_ms + processing_time) / 2.0;
                }
            });
        }
        
        log::info!("Telemetry hub event processing started");
        Ok(())
    }
    
    /// Get recent events
    pub fn get_recent_events(&self, count: usize) -> Vec<TelemetryEvent> {
        let events_guard = self.events.lock().unwrap();
        events_guard.iter().rev().take(count).cloned().collect()
    }
    
    /// Get events by type
    pub fn get_events_by_type(&self, event_type: &EventType) -> Vec<TelemetryEvent> {
        let events_guard = self.events.lock().unwrap();
        events_guard.iter()
            .filter(|event| std::mem::discriminant(&event.event_type) == std::mem::discriminant(event_type))
            .cloned()
            .collect()
    }
    
    /// Get events by severity
    pub fn get_events_by_severity(&self, severity: &EventSeverity) -> Vec<TelemetryEvent> {
        let events_guard = self.events.lock().unwrap();
        events_guard.iter()
            .filter(|event| event.severity.as_ref() == Some(severity))
            .cloned()
            .collect()
    }
    
    /// Get events in time range
    pub fn get_events_in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<TelemetryEvent> {
        let events_guard = self.events.lock().unwrap();
        events_guard.iter()
            .filter(|event| event.timestamp >= start && event.timestamp <= end)
            .cloned()
            .collect()
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> TelemetryStats {
        self.stats.clone()
    }
    
    /// Clear old events
    pub fn clear_old_events(&mut self, older_than: DateTime<Utc>) -> usize {
        let mut events_guard = self.events.lock().unwrap();
        let initial_len = events_guard.len();
        
        events_guard.retain(|event| event.timestamp >= older_than);
        
        initial_len - events_guard.len()
    }
    
    /// Export events to JSON
    pub fn export_events(&self, format: ExportFormat) -> Result<String> {
        let events_guard = self.events.lock().unwrap();
        let events: Vec<&TelemetryEvent> = events_guard.iter().collect();
        
        match format {
            ExportFormat::Json => {
                serde_json::to_string_pretty(&events)
                    .context("Failed to serialize events to JSON")
            }
            ExportFormat::Csv => {
                self.export_to_csv(&events)
            }
        }
    }
    
    /// Export events to CSV
    fn export_to_csv(&self, events: &[&TelemetryEvent]) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("timestamp,event_type,severity,source,data\n");
        
        for event in events {
            let event_type = match &event.event_type {
                EventType::HighCpuUsage => "HighCpuUsage",
                EventType::HighMemoryUsage => "HighMemoryUsage",
                EventType::HighLoadAverage => "HighLoadAverage",
                EventType::HighDiskIO => "HighDiskIO",
                EventType::NetworkErrors => "NetworkErrors",
                EventType::HighTemperature => "HighTemperature",
                EventType::ServiceFailure => "ServiceFailure",
                EventType::PerformanceOptimization => "PerformanceOptimization",
                EventType::SystemBoot => "SystemBoot",
                EventType::SystemShutdown => "SystemShutdown",
                EventType::AnomalyDetected => "AnomalyDetected",
                EventType::SecurityAlert => "SecurityAlert",
                EventType::Custom(ref name) => name,
            };
            
            let severity = event.severity.as_ref().map(|s| match s {
                EventSeverity::Debug => "Debug",
                EventSeverity::Info => "Info",
                EventSeverity::Warning => "Warning",
                EventSeverity::Error => "Error",
                EventSeverity::Critical => "Critical",
            }).unwrap_or("Unknown");
            
            let source = event.source.as_deref().unwrap_or("Unknown");
            let data = event.data.to_string().replace("\"", "\"\""); // Escape quotes
            
            csv.push_str(&format!("{},{},{},{},\"{}\"\n", 
                event.timestamp, event_type, severity, source, data));
        }
        
        Ok(csv)
    }
}

/// Export format
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
}

/// Log event processor
pub struct LogEventProcessor;

impl EventProcessor for LogEventProcessor {
    fn process(&self, event: &TelemetryEvent) -> Result<()> {
        let event_type = match &event.event_type {
            EventType::HighCpuUsage => "HighCpuUsage",
            EventType::HighMemoryUsage => "HighMemoryUsage",
            EventType::HighLoadAverage => "HighLoadAverage",
            EventType::HighDiskIO => "HighDiskIO",
            EventType::NetworkErrors => "NetworkErrors",
            EventType::HighTemperature => "HighTemperature",
            EventType::ServiceFailure => "ServiceFailure",
            EventType::PerformanceOptimization => "PerformanceOptimization",
            EventType::SystemBoot => "SystemBoot",
            EventType::SystemShutdown => "SystemShutdown",
            EventType::AnomalyDetected => "AnomalyDetected",
            EventType::SecurityAlert => "SecurityAlert",
            EventType::Custom(ref name) => name,
        };
        
        let severity = event.severity.as_ref().map(|s| match s {
            EventSeverity::Debug => log::Level::Debug,
            EventSeverity::Info => log::Level::Info,
            EventSeverity::Warning => log::Level::Warn,
            EventSeverity::Error => log::Level::Error,
            EventSeverity::Critical => log::Level::Error,
        }).unwrap_or(log::Level::Info);
        
        log::log!(severity, "Telemetry Event [{}]: {:?}", event_type, event.data);
        Ok(())
    }
    
    fn name(&self) -> &str {
        "LogEventProcessor"
    }
}

/// Alert event processor
pub struct AlertEventProcessor {
    alert_threshold: EventSeverity,
}

impl AlertEventProcessor {
    pub fn new(alert_threshold: EventSeverity) -> Self {
        Self { alert_threshold }
    }
    
    fn should_alert(&self, event: &TelemetryEvent) -> bool {
        if let Some(severity) = &event.severity {
            match (severity, &self.alert_threshold) {
                (EventSeverity::Critical, _) => true,
                (EventSeverity::Error, EventSeverity::Error | EventSeverity::Warning | EventSeverity::Info | EventSeverity::Debug) => true,
                (EventSeverity::Warning, EventSeverity::Warning | EventSeverity::Info | EventSeverity::Debug) => true,
                (EventSeverity::Info, EventSeverity::Info | EventSeverity::Debug) => true,
                (EventSeverity::Debug, EventSeverity::Debug) => true,
                _ => false,
            }
        } else {
            false
        }
    }
}

impl EventProcessor for AlertEventProcessor {
    fn process(&self, event: &TelemetryEvent) -> Result<()> {
        if self.should_alert(event) {
            log::warn!("ALERT: {:?} - {:?}", event.event_type, event.data);
            // Here you could send alerts via email, Slack, etc.
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "AlertEventProcessor"
    }
}

/// Metrics event processor
pub struct MetricsEventProcessor {
    metrics: Arc<Mutex<std::collections::HashMap<String, f64>>>,
}

impl MetricsEventProcessor {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
    
    pub fn get_metrics(&self) -> std::collections::HashMap<String, f64> {
        self.metrics.lock().unwrap().clone()
    }
}

impl EventProcessor for MetricsEventProcessor {
    fn process(&self, event: &TelemetryEvent) -> Result<()> {
        // Extract metrics from event data
        if let Some(value) = event.data.as_object() {
            for (key, val) in value {
                if let Some(num) = val.as_f64() {
                    let metric_key = format!("{}_{}", key, event.timestamp.timestamp());
                    self.metrics.lock().unwrap().insert(metric_key, num);
                }
            }
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "MetricsEventProcessor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_telemetry_hub_creation() {
        let hub = TelemetryHub::new(1000);
        assert_eq!(hub.get_stats().total_events, 0);
    }
    
    #[tokio::test]
    async fn test_event_processing() {
        let mut hub = TelemetryHub::new(1000);
        hub.add_processor(Box::new(LogEventProcessor));
        
        let sender = hub.get_sender();
        hub.start_processing().await.unwrap();
        
        let event = TelemetryEvent {
            event_type: EventType::HighCpuUsage,
            timestamp: Utc::now(),
            data: serde_json::json!({"cpu_usage": 95.5}),
            severity: Some(EventSeverity::Warning),
            source: Some("system_monitor".to_string()),
            id: Some("test-1".to_string()),
        };
        
        sender.send(event).unwrap();
        
        // Give some time for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        assert_eq!(hub.get_stats().total_events, 1);
    }
} 