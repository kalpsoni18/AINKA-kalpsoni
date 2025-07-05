use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use crate::ml_engine::LinearRegression;
use crate::data_pipeline::{DataPoint, FeatureSet};

/// Scaling target types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingTarget {
    CpuFrequency,
    MemoryAllocation,
    DiskIo,
    NetworkBandwidth,
    ProcessPriority,
    ThreadCount,
    Custom(String),
}

/// Scaling action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAction {
    pub target: ScalingTarget,
    pub action_type: String,
    pub value: f64,
    pub confidence: f64,
    pub timestamp: Instant,
    pub description: String,
}

/// Scaling prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPrediction {
    pub target: ScalingTarget,
    pub predicted_value: f64,
    pub current_value: f64,
    pub confidence: f64,
    pub time_horizon: Duration,
    pub timestamp: Instant,
    pub factors: HashMap<String, f64>,
}

/// Predictive scaler configuration
#[derive(Debug, Clone)]
pub struct PredictiveScalerConfig {
    pub enabled_targets: Vec<ScalingTarget>,
    pub prediction_horizon: Duration,
    pub min_confidence: f64,
    pub scaling_threshold: f64,
    pub max_scaling_factor: f64,
    pub history_window: Duration,
    pub update_interval: Duration,
    pub auto_scaling: bool,
}

impl Default for PredictiveScalerConfig {
    fn default() -> Self {
        Self {
            enabled_targets: vec![
                ScalingTarget::CpuFrequency,
                ScalingTarget::MemoryAllocation,
                ScalingTarget::ProcessPriority,
            ],
            prediction_horizon: Duration::from_secs(300), // 5 minutes
            min_confidence: 0.7,
            scaling_threshold: 0.1, // 10% change required
            max_scaling_factor: 2.0,
            history_window: Duration::from_secs(3600), // 1 hour
            update_interval: Duration::from_secs(60), // 1 minute
            auto_scaling: true,
        }
    }
}

/// CPU frequency scaler
struct CpuFrequencyScaler {
    model: LinearRegression,
    history: VecDeque<(Instant, f64)>,
    current_frequency: f64,
    max_frequency: f64,
    min_frequency: f64,
}

impl CpuFrequencyScaler {
    fn new() -> Self {
        Self {
            model: LinearRegression::new(0.01, 0.9, 0.001),
            history: VecDeque::new(),
            current_frequency: 2000.0, // Default 2GHz
            max_frequency: 4000.0, // 4GHz
            min_frequency: 800.0,  // 800MHz
        }
    }

    fn add_measurement(&mut self, timestamp: Instant, frequency: f64) {
        self.current_frequency = frequency;
        
        if self.history.len() >= 100 {
            self.history.pop_front();
        }
        self.history.push_back((timestamp, frequency));
    }

    fn predict_frequency(&mut self, features: &FeatureSet) -> Option<f64> {
        if self.history.len() < 10 {
            return None;
        }

        // Train model on historical data
        let mut training_data = Vec::new();
        for (i, (_, freq)) in self.history.iter().enumerate() {
            if i + 1 < self.history.len() {
                let next_freq = self.history[i + 1].1;
                training_data.push((features.clone(), next_freq));
            }
        }

        // Update model
        for (features, target) in training_data {
            self.model.update(&features, target);
        }

        // Make prediction
        let prediction = self.model.predict(features);
        Some(prediction.max(self.min_frequency).min(self.max_frequency))
    }

    fn get_scaling_action(&mut self, predicted_frequency: f64) -> Option<ScalingAction> {
        let change_ratio = (predicted_frequency - self.current_frequency).abs() / self.current_frequency;
        
        if change_ratio > 0.1 { // 10% change threshold
            let action_type = if predicted_frequency > self.current_frequency {
                "increase".to_string()
            } else {
                "decrease".to_string()
            };

            Some(ScalingAction {
                target: ScalingTarget::CpuFrequency,
                action_type,
                value: predicted_frequency,
                confidence: 0.8,
                timestamp: Instant::now(),
                description: format!("CPU frequency scaling: {} -> {:.0} MHz", 
                                   self.current_frequency as i32, predicted_frequency as i32),
            })
        } else {
            None
        }
    }
}

/// Memory allocation scaler
struct MemoryAllocationScaler {
    model: LinearRegression,
    history: VecDeque<(Instant, f64)>,
    current_usage: f64,
    total_memory: f64,
    swap_usage: f64,
}

impl MemoryAllocationScaler {
    fn new() -> Self {
        Self {
            model: LinearRegression::new(0.01, 0.9, 0.001),
            history: VecDeque::new(),
            current_usage: 0.0,
            total_memory: 8.0 * 1024.0 * 1024.0 * 1024.0, // 8GB
            swap_usage: 0.0,
        }
    }

    fn add_measurement(&mut self, timestamp: Instant, usage: f64, swap: f64) {
        self.current_usage = usage;
        self.swap_usage = swap;
        
        if self.history.len() >= 100 {
            self.history.pop_front();
        }
        self.history.push_back((timestamp, usage));
    }

    fn predict_memory_usage(&mut self, features: &FeatureSet) -> Option<f64> {
        if self.history.len() < 10 {
            return None;
        }

        // Train model on historical data
        let mut training_data = Vec::new();
        for (i, (_, usage)) in self.history.iter().enumerate() {
            if i + 1 < self.history.len() {
                let next_usage = self.history[i + 1].1;
                training_data.push((features.clone(), next_usage));
            }
        }

        // Update model
        for (features, target) in training_data {
            self.model.update(&features, target);
        }

        // Make prediction
        let prediction = self.model.predict(features);
        Some(prediction.max(0.0).min(1.0)) // Normalize to 0-1
    }

    fn get_scaling_action(&mut self, predicted_usage: f64) -> Option<ScalingAction> {
        let change_ratio = (predicted_usage - self.current_usage).abs();
        
        if change_ratio > 0.05 { // 5% change threshold
            let action_type = if predicted_usage > 0.9 {
                "emergency_cleanup".to_string()
            } else if predicted_usage > 0.8 {
                "aggressive_cleanup".to_string()
            } else if predicted_usage < 0.3 {
                "relax_cleanup".to_string()
            } else {
                "normal_cleanup".to_string()
            };

            Some(ScalingAction {
                target: ScalingTarget::MemoryAllocation,
                action_type,
                value: predicted_usage,
                confidence: 0.8,
                timestamp: Instant::now(),
                description: format!("Memory usage prediction: {:.1}% -> {:.1}%", 
                                   self.current_usage * 100.0, predicted_usage * 100.0),
            })
        } else {
            None
        }
    }
}

/// Process priority scaler
struct ProcessPriorityScaler {
    process_models: HashMap<u32, LinearRegression>,
    process_history: HashMap<u32, VecDeque<(Instant, f64)>>,
    current_priorities: HashMap<u32, i32>,
}

impl ProcessPriorityScaler {
    fn new() -> Self {
        Self {
            process_models: HashMap::new(),
            process_history: HashMap::new(),
            current_priorities: HashMap::new(),
        }
    }

    fn add_process_measurement(&mut self, pid: u32, timestamp: Instant, cpu_usage: f64, priority: i32) {
        self.current_priorities.insert(pid, priority);
        
        let history = self.process_history.entry(pid).or_insert_with(VecDeque::new);
        if history.len() >= 50 {
            history.pop_front();
        }
        history.push_back((timestamp, cpu_usage));
    }

    fn predict_process_priority(&mut self, pid: u32, features: &FeatureSet) -> Option<i32> {
        let history = self.process_history.get(&pid)?;
        if history.len() < 5 {
            return None;
        }

        let model = self.process_models.entry(pid).or_insert_with(|| 
            LinearRegression::new(0.01, 0.9, 0.001)
        );

        // Train model on historical data
        let mut training_data = Vec::new();
        for (i, (_, cpu_usage)) in history.iter().enumerate() {
            if i + 1 < history.len() {
                let next_cpu = history[i + 1].1;
                training_data.push((features.clone(), next_cpu));
            }
        }

        // Update model
        for (features, target) in training_data {
            model.update(&features, target);
        }

        // Make prediction and convert to priority
        let predicted_cpu = model.predict(features);
        let priority = self.cpu_to_priority(predicted_cpu);
        Some(priority)
    }

    fn cpu_to_priority(&self, cpu_usage: f64) -> i32 {
        // Convert CPU usage to process priority (lower number = higher priority)
        if cpu_usage > 0.8 {
            -10 // High priority for CPU-intensive processes
        } else if cpu_usage > 0.5 {
            -5  // Medium priority
        } else if cpu_usage > 0.2 {
            0   // Normal priority
        } else {
            10  // Low priority for idle processes
        }
    }

    fn get_scaling_action(&mut self, pid: u32, predicted_priority: i32) -> Option<ScalingAction> {
        let current_priority = self.current_priorities.get(&pid)?;
        
        if (predicted_priority - current_priority).abs() >= 5 {
            let action_type = if predicted_priority < *current_priority {
                "increase_priority".to_string()
            } else {
                "decrease_priority".to_string()
            };

            Some(ScalingAction {
                target: ScalingTarget::ProcessPriority,
                action_type,
                value: predicted_priority as f64,
                confidence: 0.7,
                timestamp: Instant::now(),
                description: format!("Process {} priority: {} -> {}", 
                                   pid, current_priority, predicted_priority),
            })
        } else {
            None
        }
    }
}

/// Main predictive scaler
pub struct PredictiveScaler {
    config: PredictiveScalerConfig,
    cpu_scaler: CpuFrequencyScaler,
    memory_scaler: MemoryAllocationScaler,
    process_scaler: ProcessPriorityScaler,
    action_sender: mpsc::UnboundedSender<ScalingAction>,
    prediction_sender: mpsc::UnboundedSender<ScalingPrediction>,
    last_update: Instant,
}

impl PredictiveScaler {
    /// Create a new predictive scaler
    pub fn new(
        config: PredictiveScalerConfig,
        action_sender: mpsc::UnboundedSender<ScalingAction>,
        prediction_sender: mpsc::UnboundedSender<ScalingPrediction>,
    ) -> Self {
        Self {
            config,
            cpu_scaler: CpuFrequencyScaler::new(),
            memory_scaler: MemoryAllocationScaler::new(),
            process_scaler: ProcessPriorityScaler::new(),
            action_sender,
            prediction_sender,
            last_update: Instant::now(),
        }
    }

    /// Process a new data point and generate predictions
    pub fn process_data_point(&mut self, data_point: &DataPoint) -> Result<(), Box<dyn std::error::Error>> {
        let now = Instant::now();
        
        // Update measurements
        if let Some(cpu_freq) = data_point.features.get("cpu_frequency") {
            self.cpu_scaler.add_measurement(now, *cpu_freq);
        }
        
        if let Some(memory_usage) = data_point.features.get("memory_usage") {
            let swap_usage = data_point.features.get("swap_usage").unwrap_or(&0.0);
            self.memory_scaler.add_measurement(now, *memory_usage, *swap_usage);
        }
        
        if let Some(process_cpu) = data_point.features.get("process_cpu") {
            let pid = data_point.features.get("pid").unwrap_or(&0.0) as u32;
            let priority = data_point.features.get("process_priority").unwrap_or(&0.0) as i32;
            self.process_scaler.add_process_measurement(pid, now, *process_cpu, priority);
        }

        // Generate predictions if enough time has passed
        if now.duration_since(self.last_update) >= self.config.update_interval {
            self.generate_predictions(data_point)?;
            self.last_update = now;
        }

        Ok(())
    }

    /// Generate predictions for all enabled targets
    fn generate_predictions(&mut self, data_point: &DataPoint) -> Result<(), Box<dyn std::error::Error>> {
        for target in &self.config.enabled_targets {
            match target {
                ScalingTarget::CpuFrequency => {
                    if let Some(predicted_freq) = self.cpu_scaler.predict_frequency(&data_point.features) {
                        let prediction = ScalingPrediction {
                            target: ScalingTarget::CpuFrequency,
                            predicted_value: predicted_freq,
                            current_value: self.cpu_scaler.current_frequency,
                            confidence: 0.8,
                            time_horizon: self.config.prediction_horizon,
                            timestamp: Instant::now(),
                            factors: {
                                let mut factors = HashMap::new();
                                factors.insert("load_average".to_string(), 
                                             data_point.features.get("load_average").unwrap_or(&0.0));
                                factors.insert("cpu_usage".to_string(), 
                                             data_point.features.get("cpu_usage").unwrap_or(&0.0));
                                factors
                            },
                        };

                        if let Err(e) = self.prediction_sender.send(prediction) {
                            log::error!("Failed to send CPU prediction: {}", e);
                        }

                        // Generate scaling action if auto-scaling is enabled
                        if self.config.auto_scaling {
                            if let Some(action) = self.cpu_scaler.get_scaling_action(predicted_freq) {
                                if let Err(e) = self.action_sender.send(action) {
                                    log::error!("Failed to send CPU scaling action: {}", e);
                                }
                            }
                        }
                    }
                }

                ScalingTarget::MemoryAllocation => {
                    if let Some(predicted_usage) = self.memory_scaler.predict_memory_usage(&data_point.features) {
                        let prediction = ScalingPrediction {
                            target: ScalingTarget::MemoryAllocation,
                            predicted_value: predicted_usage,
                            current_value: self.memory_scaler.current_usage,
                            confidence: 0.8,
                            time_horizon: self.config.prediction_horizon,
                            timestamp: Instant::now(),
                            factors: {
                                let mut factors = HashMap::new();
                                factors.insert("memory_usage".to_string(), 
                                             data_point.features.get("memory_usage").unwrap_or(&0.0));
                                factors.insert("swap_usage".to_string(), 
                                             data_point.features.get("swap_usage").unwrap_or(&0.0));
                                factors
                            },
                        };

                        if let Err(e) = self.prediction_sender.send(prediction) {
                            log::error!("Failed to send memory prediction: {}", e);
                        }

                        if self.config.auto_scaling {
                            if let Some(action) = self.memory_scaler.get_scaling_action(predicted_usage) {
                                if let Err(e) = self.action_sender.send(action) {
                                    log::error!("Failed to send memory scaling action: {}", e);
                                }
                            }
                        }
                    }
                }

                ScalingTarget::ProcessPriority => {
                    // Predict priority for all tracked processes
                    for &pid in self.process_scaler.current_priorities.keys() {
                        if let Some(predicted_priority) = self.process_scaler.predict_process_priority(pid, &data_point.features) {
                            let current_priority = self.process_scaler.current_priorities.get(&pid).unwrap_or(&0);
                            
                            let prediction = ScalingPrediction {
                                target: ScalingTarget::ProcessPriority,
                                predicted_value: predicted_priority as f64,
                                current_value: *current_priority as f64,
                                confidence: 0.7,
                                time_horizon: self.config.prediction_horizon,
                                timestamp: Instant::now(),
                                factors: {
                                    let mut factors = HashMap::new();
                                    factors.insert("process_cpu".to_string(), 
                                                 data_point.features.get("process_cpu").unwrap_or(&0.0));
                                    factors.insert("pid".to_string(), pid as f64);
                                    factors
                                },
                            };

                            if let Err(e) = self.prediction_sender.send(prediction) {
                                log::error!("Failed to send process priority prediction: {}", e);
                            }

                            if self.config.auto_scaling {
                                if let Some(action) = self.process_scaler.get_scaling_action(pid, predicted_priority) {
                                    if let Err(e) = self.action_sender.send(action) {
                                        log::error!("Failed to send process priority action: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }

                _ => {
                    // Handle custom scaling targets
                    log::debug!("Custom scaling target not implemented: {:?}", target);
                }
            }
        }

        Ok(())
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PredictiveScalerConfig) {
        self.config = config;
    }

    /// Get scaling statistics
    pub fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("cpu_history_size".to_string(), self.cpu_scaler.history.len());
        stats.insert("memory_history_size".to_string(), self.memory_scaler.history.len());
        stats.insert("tracked_processes".to_string(), self.process_scaler.current_priorities.len());
        stats.insert("enabled_targets".to_string(), self.config.enabled_targets.len());
        stats
    }

    /// Execute a scaling action
    pub fn execute_scaling_action(&self, action: &ScalingAction) -> Result<(), Box<dyn std::error::Error>> {
        match &action.target {
            ScalingTarget::CpuFrequency => {
                self.execute_cpu_scaling(action)?;
            }
            ScalingTarget::MemoryAllocation => {
                self.execute_memory_scaling(action)?;
            }
            ScalingTarget::ProcessPriority => {
                self.execute_process_priority_scaling(action)?;
            }
            _ => {
                log::warn!("Scaling target not implemented: {:?}", action.target);
            }
        }
        Ok(())
    }

    /// Execute CPU frequency scaling
    fn execute_cpu_scaling(&self, action: &ScalingAction) -> Result<(), Box<dyn std::error::Error>> {
        // This would typically involve writing to /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed
        // For now, we'll just log the action
        log::info!("CPU scaling action: {} -> {:.0} MHz", 
                   action.action_type, action.value);
        Ok(())
    }

    /// Execute memory scaling
    fn execute_memory_scaling(&self, action: &ScalingAction) -> Result<(), Box<dyn std::error::Error>> {
        // This would typically involve adjusting memory pressure or swap settings
        log::info!("Memory scaling action: {} (usage: {:.1}%)", 
                   action.action_type, action.value * 100.0);
        Ok(())
    }

    /// Execute process priority scaling
    fn execute_process_priority_scaling(&self, action: &ScalingAction) -> Result<(), Box<dyn std::error::Error>> {
        // This would typically involve calling setpriority() or writing to /proc/pid/oom_adj
        log::info!("Process priority scaling action: {} -> priority {}", 
                   action.action_type, action.value as i32);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn test_predictive_scaler_creation() {
        let (action_tx, _action_rx) = mpsc::unbounded_channel();
        let (prediction_tx, _prediction_rx) = mpsc::unbounded_channel();
        let config = PredictiveScalerConfig::default();
        
        let scaler = PredictiveScaler::new(config, action_tx, prediction_tx);
        assert!(!scaler.config.enabled_targets.is_empty());
    }

    #[test]
    fn test_cpu_frequency_scaler() {
        let mut scaler = CpuFrequencyScaler::new();
        
        // Add some historical data
        for i in 0..20 {
            scaler.add_measurement(Instant::now(), 2000.0 + i as f64 * 10.0);
        }
        
        let mut features = FeatureSet::new();
        features.insert("cpu_usage".to_string(), 0.8);
        features.insert("load_average".to_string(), 2.0);
        
        let prediction = scaler.predict_frequency(&features);
        assert!(prediction.is_some());
    }

    #[test]
    fn test_memory_allocation_scaler() {
        let mut scaler = MemoryAllocationScaler::new();
        
        // Add some historical data
        for i in 0..20 {
            scaler.add_measurement(Instant::now(), 0.5 + i as f64 * 0.01, 0.1);
        }
        
        let mut features = FeatureSet::new();
        features.insert("memory_usage".to_string(), 0.7);
        features.insert("swap_usage".to_string(), 0.2);
        
        let prediction = scaler.predict_memory_usage(&features);
        assert!(prediction.is_some());
    }
} 