use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use crate::ml_engine::LinearRegression;
use crate::data_pipeline::{DataPoint, FeatureSet};

/// Anomaly types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    CpuSpike,
    MemoryLeak,
    DiskIoAnomaly,
    NetworkAnomaly,
    ProcessAnomaly,
    SecurityThreat,
    PerformanceDegradation,
    ResourceExhaustion,
    Custom(String),
}

/// Anomaly severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub confidence: f64,
    pub timestamp: Instant,
    pub description: String,
    pub metrics: HashMap<String, f64>,
    pub recommendations: Vec<String>,
}

/// Anomaly detector configuration
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    pub enabled_detectors: Vec<String>,
    pub window_size: usize,
    pub threshold_multiplier: f64,
    pub min_confidence: f64,
    pub update_interval: Duration,
    pub history_size: usize,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            enabled_detectors: vec![
                "statistical".to_string(),
                "ml_based".to_string(),
                "threshold".to_string(),
                "pattern".to_string(),
            ],
            window_size: 100,
            threshold_multiplier: 2.0,
            min_confidence: 0.7,
            update_interval: Duration::from_secs(60),
            history_size: 1000,
        }
    }
}

/// Statistical anomaly detector
struct StatisticalDetector {
    window_size: usize,
    threshold_multiplier: f64,
    history: VecDeque<f64>,
}

impl StatisticalDetector {
    fn new(window_size: usize, threshold_multiplier: f64) -> Self {
        Self {
            window_size,
            threshold_multiplier,
            history: VecDeque::with_capacity(window_size),
        }
    }

    fn add_value(&mut self, value: f64) {
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(value);
    }

    fn detect_anomaly(&self, current_value: f64) -> Option<(f64, f64)> {
        if self.history.len() < self.window_size / 2 {
            return None;
        }

        let mean = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let variance = self.history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.history.len() as f64;
        let std_dev = variance.sqrt();

        let upper_threshold = mean + self.threshold_multiplier * std_dev;
        let lower_threshold = mean - self.threshold_multiplier * std_dev;

        if current_value > upper_threshold || current_value < lower_threshold {
            let confidence = if std_dev > 0.0 {
                let z_score = (current_value - mean).abs() / std_dev;
                1.0 - (-z_score).exp()
            } else {
                0.5
            };
            Some((confidence, mean))
        } else {
            None
        }
    }
}

/// ML-based anomaly detector
struct MLDetector {
    model: LinearRegression,
    feature_history: VecDeque<FeatureSet>,
    prediction_window: usize,
}

impl MLDetector {
    fn new(prediction_window: usize) -> Self {
        Self {
            model: LinearRegression::new(0.01, 0.9, 0.001),
            feature_history: VecDeque::with_capacity(prediction_window),
            prediction_window,
        }
    }

    fn add_features(&mut self, features: FeatureSet) {
        if self.feature_history.len() >= self.prediction_window {
            self.feature_history.pop_front();
        }
        self.feature_history.push_back(features);
    }

    fn detect_anomaly(&mut self, current_features: &FeatureSet, target_value: f64) -> Option<f64> {
        if self.feature_history.len() < self.prediction_window / 2 {
            return None;
        }

        // Train model on historical data
        let mut training_data = Vec::new();
        for (i, features) in self.feature_history.iter().enumerate() {
            if i + 1 < self.feature_history.len() {
                let next_features = &self.feature_history[i + 1];
                let target = next_features.get("cpu_usage").unwrap_or(0.0);
                training_data.push((features.clone(), target));
            }
        }

        // Update model
        for (features, target) in training_data {
            self.model.update(&features, target);
        }

        // Make prediction
        let prediction = self.model.predict(current_features);
        let error = (target_value - prediction).abs();
        let confidence = 1.0 - (error / target_value.max(1.0));

        if confidence < 0.7 {
            Some(confidence)
        } else {
            None
        }
    }
}

/// Threshold-based detector
struct ThresholdDetector {
    thresholds: HashMap<String, (f64, f64)>, // (min, max)
}

impl ThresholdDetector {
    fn new() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("cpu_usage".to_string(), (0.0, 95.0));
        thresholds.insert("memory_usage".to_string(), (0.0, 90.0));
        thresholds.insert("disk_io".to_string(), (0.0, 1000.0));
        thresholds.insert("network_io".to_string(), (0.0, 100.0));
        thresholds.insert("load_average".to_string(), (0.0, 10.0));

        Self { thresholds }
    }

    fn detect_anomaly(&self, metric_name: &str, value: f64) -> Option<f64> {
        if let Some((min, max)) = self.thresholds.get(metric_name) {
            if value < *min || value > *max {
                let confidence = if value > *max {
                    (value - max) / max
                } else {
                    (min - value) / min.max(1.0)
                };
                Some(confidence.min(1.0))
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Pattern-based detector
struct PatternDetector {
    patterns: HashMap<String, Vec<f64>>,
    pattern_window: usize,
    similarity_threshold: f64,
}

impl PatternDetector {
    fn new(pattern_window: usize, similarity_threshold: f64) -> Self {
        Self {
            patterns: HashMap::new(),
            pattern_window,
            similarity_threshold,
        }
    }

    fn add_pattern(&mut self, pattern_name: String, values: Vec<f64>) {
        self.patterns.insert(pattern_name, values);
    }

    fn detect_anomaly(&self, current_values: &[f64]) -> Option<(String, f64)> {
        for (pattern_name, pattern_values) in &self.patterns {
            if current_values.len() == pattern_values.len() {
                let similarity = self.calculate_similarity(current_values, pattern_values);
                if similarity < self.similarity_threshold {
                    return Some((pattern_name.clone(), similarity));
                }
            }
        }
        None
    }

    fn calculate_similarity(&self, values1: &[f64], values2: &[f64]) -> f64 {
        if values1.len() != values2.len() {
            return 0.0;
        }

        let mut sum_squared_diff = 0.0;
        for (v1, v2) in values1.iter().zip(values2.iter()) {
            sum_squared_diff += (v1 - v2).powi(2);
        }

        let mse = sum_squared_diff / values1.len() as f64;
        1.0 / (1.0 + mse)
    }
}

/// Main anomaly detector
pub struct AnomalyDetector {
    config: AnomalyConfig,
    statistical_detectors: HashMap<String, StatisticalDetector>,
    ml_detectors: HashMap<String, MLDetector>,
    threshold_detector: ThresholdDetector,
    pattern_detector: PatternDetector,
    anomaly_sender: mpsc::UnboundedSender<AnomalyResult>,
    history: Arc<Mutex<VecDeque<DataPoint>>>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new(config: AnomalyConfig, anomaly_sender: mpsc::UnboundedSender<AnomalyResult>) -> Self {
        let statistical_detectors = HashMap::new();
        let ml_detectors = HashMap::new();
        let threshold_detector = ThresholdDetector::new();
        let pattern_detector = PatternDetector::new(50, 0.8);
        let history = Arc::new(Mutex::new(VecDeque::with_capacity(config.history_size)));

        Self {
            config,
            statistical_detectors,
            ml_detectors,
            threshold_detector,
            pattern_detector,
            anomaly_sender,
            history,
        }
    }

    /// Process a new data point
    pub fn process_data_point(&mut self, data_point: DataPoint) -> Result<(), Box<dyn std::error::Error>> {
        // Add to history
        {
            let mut history = self.history.lock().unwrap();
            if history.len() >= self.config.history_size {
                history.pop_front();
            }
            history.push_back(data_point.clone());
        }

        // Run all enabled detectors
        let mut anomalies = Vec::new();

        if self.config.enabled_detectors.contains(&"statistical".to_string()) {
            anomalies.extend(self.run_statistical_detection(&data_point));
        }

        if self.config.enabled_detectors.contains(&"ml_based".to_string()) {
            anomalies.extend(self.run_ml_detection(&data_point));
        }

        if self.config.enabled_detectors.contains(&"threshold".to_string()) {
            anomalies.extend(self.run_threshold_detection(&data_point));
        }

        if self.config.enabled_detectors.contains(&"pattern".to_string()) {
            anomalies.extend(self.run_pattern_detection(&data_point));
        }

        // Send anomalies
        for anomaly in anomalies {
            if let Err(e) = self.anomaly_sender.send(anomaly) {
                log::error!("Failed to send anomaly: {}", e);
            }
        }

        Ok(())
    }

    /// Run statistical anomaly detection
    fn run_statistical_detection(&mut self, data_point: &DataPoint) -> Vec<AnomalyResult> {
        let mut anomalies = Vec::new();

        for (metric_name, value) in &data_point.features {
            let detector = self.statistical_detectors
                .entry(metric_name.clone())
                .or_insert_with(|| StatisticalDetector::new(
                    self.config.window_size,
                    self.config.threshold_multiplier
                ));

            detector.add_value(*value);

            if let Some((confidence, mean)) = detector.detect_anomaly(*value) {
                if confidence >= self.config.min_confidence {
                    let anomaly_type = self.get_anomaly_type(metric_name);
                    let severity = self.get_severity(confidence, *value, mean);
                    let description = self.get_description(anomaly_type, metric_name, *value, mean);
                    let recommendations = self.get_recommendations(anomaly_type, metric_name, *value);

                    let mut metrics = HashMap::new();
                    metrics.insert(metric_name.clone(), *value);
                    metrics.insert("mean".to_string(), mean);
                    metrics.insert("confidence".to_string(), confidence);

                    anomalies.push(AnomalyResult {
                        anomaly_type,
                        severity,
                        confidence,
                        timestamp: Instant::now(),
                        description,
                        metrics,
                        recommendations,
                    });
                }
            }
        }

        anomalies
    }

    /// Run ML-based anomaly detection
    fn run_ml_detection(&mut self, data_point: &DataPoint) -> Vec<AnomalyResult> {
        let mut anomalies = Vec::new();

        for (metric_name, value) in &data_point.features {
            let detector = self.ml_detectors
                .entry(metric_name.clone())
                .or_insert_with(|| MLDetector::new(20));

            detector.add_features(data_point.features.clone());

            if let Some(confidence) = detector.detect_anomaly(&data_point.features, *value) {
                if confidence >= self.config.min_confidence {
                    let anomaly_type = self.get_anomaly_type(metric_name);
                    let severity = self.get_severity(confidence, *value, *value);
                    let description = format!("ML-based anomaly detected for {}: confidence={:.2}", metric_name, confidence);
                    let recommendations = self.get_recommendations(anomaly_type, metric_name, *value);

                    let mut metrics = HashMap::new();
                    metrics.insert(metric_name.clone(), *value);
                    metrics.insert("confidence".to_string(), confidence);

                    anomalies.push(AnomalyResult {
                        anomaly_type,
                        severity,
                        confidence,
                        timestamp: Instant::now(),
                        description,
                        metrics,
                        recommendations,
                    });
                }
            }
        }

        anomalies
    }

    /// Run threshold-based detection
    fn run_threshold_detection(&self, data_point: &DataPoint) -> Vec<AnomalyResult> {
        let mut anomalies = Vec::new();

        for (metric_name, value) in &data_point.features {
            if let Some(confidence) = self.threshold_detector.detect_anomaly(metric_name, *value) {
                if confidence >= self.config.min_confidence {
                    let anomaly_type = self.get_anomaly_type(metric_name);
                    let severity = self.get_severity(confidence, *value, *value);
                    let description = format!("Threshold exceeded for {}: value={:.2}", metric_name, value);
                    let recommendations = self.get_recommendations(anomaly_type, metric_name, *value);

                    let mut metrics = HashMap::new();
                    metrics.insert(metric_name.clone(), *value);
                    metrics.insert("confidence".to_string(), confidence);

                    anomalies.push(AnomalyResult {
                        anomaly_type,
                        severity,
                        confidence,
                        timestamp: Instant::now(),
                        description,
                        metrics,
                        recommendations,
                    });
                }
            }
        }

        anomalies
    }

    /// Run pattern-based detection
    fn run_pattern_detection(&self, data_point: &DataPoint) -> Vec<AnomalyResult> {
        let mut anomalies = Vec::new();

        // Convert recent history to pattern
        let history = self.history.lock().unwrap();
        if history.len() >= 10 {
            let recent_values: Vec<f64> = history.iter()
                .rev()
                .take(10)
                .flat_map(|dp| dp.features.values().cloned())
                .collect();

            if let Some((pattern_name, similarity)) = self.pattern_detector.detect_anomaly(&recent_values) {
                let confidence = 1.0 - similarity;
                if confidence >= self.config.min_confidence {
                    let anomaly_type = AnomalyType::Custom("Pattern Anomaly".to_string());
                    let severity = self.get_severity(confidence, 0.0, 0.0);
                    let description = format!("Pattern anomaly detected: {} (similarity={:.2})", pattern_name, similarity);
                    let recommendations = vec![
                        "Investigate pattern deviation".to_string(),
                        "Check for system changes".to_string(),
                    ];

                    let mut metrics = HashMap::new();
                    metrics.insert("similarity".to_string(), similarity);
                    metrics.insert("confidence".to_string(), confidence);

                    anomalies.push(AnomalyResult {
                        anomaly_type,
                        severity,
                        confidence,
                        timestamp: Instant::now(),
                        description,
                        metrics,
                        recommendations,
                    });
                }
            }
        }

        anomalies
    }

    /// Get anomaly type from metric name
    fn get_anomaly_type(&self, metric_name: &str) -> AnomalyType {
        match metric_name {
            "cpu_usage" => AnomalyType::CpuSpike,
            "memory_usage" => AnomalyType::MemoryLeak,
            "disk_io" => AnomalyType::DiskIoAnomaly,
            "network_io" => AnomalyType::NetworkAnomaly,
            "load_average" => AnomalyType::PerformanceDegradation,
            _ => AnomalyType::Custom(metric_name.to_string()),
        }
    }

    /// Get severity level
    fn get_severity(&self, confidence: f64, current_value: f64, baseline: f64) -> AnomalySeverity {
        let deviation = (current_value - baseline).abs() / baseline.max(1.0);
        
        match (confidence, deviation) {
            (c, d) if c >= 0.9 && d >= 3.0 => AnomalySeverity::Critical,
            (c, d) if c >= 0.8 && d >= 2.0 => AnomalySeverity::High,
            (c, d) if c >= 0.7 && d >= 1.5 => AnomalySeverity::Medium,
            _ => AnomalySeverity::Low,
        }
    }

    /// Get description
    fn get_description(&self, anomaly_type: AnomalyType, metric_name: &str, current_value: f64, baseline: f64) -> String {
        let deviation = ((current_value - baseline) / baseline.max(1.0) * 100.0) as i32;
        match anomaly_type {
            AnomalyType::CpuSpike => format!("CPU usage spike detected: {}% (baseline: {:.1}%)", current_value as i32, baseline),
            AnomalyType::MemoryLeak => format!("Memory usage anomaly: {}% (baseline: {:.1}%)", current_value as i32, baseline),
            AnomalyType::DiskIoAnomaly => format!("Disk I/O anomaly: {:.1} MB/s (baseline: {:.1} MB/s)", current_value, baseline),
            AnomalyType::NetworkAnomaly => format!("Network I/O anomaly: {:.1} MB/s (baseline: {:.1} MB/s)", current_value, baseline),
            AnomalyType::PerformanceDegradation => format!("Performance degradation: load {:.2} (baseline: {:.2})", current_value, baseline),
            _ => format!("Anomaly in {}: {:.2} (baseline: {:.2})", metric_name, current_value, baseline),
        }
    }

    /// Get recommendations
    fn get_recommendations(&self, anomaly_type: AnomalyType, metric_name: &str, value: f64) -> Vec<String> {
        match anomaly_type {
            AnomalyType::CpuSpike => vec![
                "Check for runaway processes".to_string(),
                "Monitor CPU-intensive applications".to_string(),
                "Consider CPU scaling".to_string(),
            ],
            AnomalyType::MemoryLeak => vec![
                "Check for memory leaks".to_string(),
                "Monitor memory usage patterns".to_string(),
                "Consider memory optimization".to_string(),
            ],
            AnomalyType::DiskIoAnomaly => vec![
                "Check disk I/O patterns".to_string(),
                "Monitor disk usage".to_string(),
                "Consider I/O optimization".to_string(),
            ],
            AnomalyType::NetworkAnomaly => vec![
                "Check network traffic patterns".to_string(),
                "Monitor network connections".to_string(),
                "Consider network optimization".to_string(),
            ],
            AnomalyType::PerformanceDegradation => vec![
                "Check system load".to_string(),
                "Monitor resource usage".to_string(),
                "Consider performance tuning".to_string(),
            ],
            _ => vec![
                format!("Investigate {} anomaly", metric_name),
                "Check system logs".to_string(),
                "Monitor related metrics".to_string(),
            ],
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: AnomalyConfig) {
        self.config = config;
    }

    /// Get detection statistics
    pub fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("statistical_detectors".to_string(), self.statistical_detectors.len());
        stats.insert("ml_detectors".to_string(), self.ml_detectors.len());
        stats.insert("history_size".to_string(), self.history.lock().unwrap().len());
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn test_statistical_detector() {
        let mut detector = StatisticalDetector::new(10, 2.0);
        
        // Add normal values
        for i in 0..10 {
            detector.add_value(i as f64);
        }
        
        // Test normal value
        assert!(detector.detect_anomaly(5.0).is_none());
        
        // Test anomalous value
        assert!(detector.detect_anomaly(50.0).is_some());
    }

    #[test]
    fn test_threshold_detector() {
        let detector = ThresholdDetector::new();
        
        // Test normal value
        assert!(detector.detect_anomaly("cpu_usage", 50.0).is_none());
        
        // Test threshold exceeded
        assert!(detector.detect_anomaly("cpu_usage", 100.0).is_some());
    }

    #[test]
    fn test_anomaly_detector_creation() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let config = AnomalyConfig::default();
        
        let detector = AnomalyDetector::new(config, tx);
        assert!(detector.statistical_detectors.is_empty());
        assert!(detector.ml_detectors.is_empty());
    }
} 