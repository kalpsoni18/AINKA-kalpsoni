use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use tokio::sync::mpsc;
use ndarray::{Array1, Array2};
use crate::{TelemetryEvent, AinkaError};

/// Data preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Enable normalization
    pub enable_normalization: bool,
    
    /// Enable standardization
    pub enable_standardization: bool,
    
    /// Enable outlier detection
    pub enable_outlier_detection: bool,
    
    /// Outlier detection threshold
    pub outlier_threshold: f64,
    
    /// Enable missing value handling
    pub enable_missing_value_handling: bool,
    
    /// Missing value strategy
    pub missing_value_strategy: MissingValueStrategy,
    
    /// Enable data smoothing
    pub enable_smoothing: bool,
    
    /// Smoothing window size
    pub smoothing_window: usize,
}

/// Missing value handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingValueStrategy {
    Mean,
    Median,
    Zero,
    ForwardFill,
    BackwardFill,
    Interpolate,
}

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Enable rolling features
    pub enable_rolling_features: bool,
    
    /// Rolling window sizes
    pub rolling_windows: Vec<usize>,
    
    /// Enable lag features
    pub enable_lag_features: bool,
    
    /// Lag periods
    pub lag_periods: Vec<usize>,
    
    /// Enable interaction features
    pub enable_interaction_features: bool,
    
    /// Enable polynomial features
    pub enable_polynomial_features: bool,
    
    /// Polynomial degree
    pub polynomial_degree: usize,
    
    /// Enable time-based features
    pub enable_time_features: bool,
    
    /// Enable statistical features
    pub enable_statistical_features: bool,
    
    /// Statistical window size
    pub statistical_window: usize,
}

/// Data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Features
    pub features: Vec<f64>,
    
    /// Target value (if available)
    pub target: Option<f64>,
    
    /// Metadata
    pub metadata: serde_json::Value,
}

/// Feature importance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Feature name
    pub name: String,
    
    /// Importance score
    pub importance: f64,
    
    /// Feature type
    pub feature_type: FeatureType,
}

/// Feature types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureType {
    Raw,
    Rolling,
    Lag,
    Interaction,
    Polynomial,
    Time,
    Statistical,
}

/// Data pipeline for preprocessing and feature engineering
pub struct DataPipeline {
    /// Event sender
    event_tx: mpsc::UnboundedSender<TelemetryEvent>,
    
    /// Preprocessing configuration
    preprocessing_config: PreprocessingConfig,
    
    /// Feature configuration
    feature_config: FeatureConfig,
    
    /// Data buffer
    data_buffer: Arc<Mutex<VecDeque<DataPoint>>>,
    
    /// Maximum buffer size
    max_buffer_size: usize,
    
    /// Feature scaler
    feature_scaler: Option<FeatureScaler>,
    
    /// Feature importance tracker
    feature_importance: Arc<Mutex<Vec<FeatureImportance>>>,
    
    /// Statistics
    stats: DataPipelineStats,
}

/// Feature scaler for normalization/standardization
#[derive(Debug, Clone)]
pub struct FeatureScaler {
    /// Mean values for each feature
    means: Vec<f64>,
    
    /// Standard deviation values for each feature
    stds: Vec<f64>,
    
    /// Min values for each feature
    mins: Vec<f64>,
    
    /// Max values for each feature
    maxs: Vec<f64>,
    
    /// Number of samples used for fitting
    n_samples: usize,
}

/// Data pipeline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPipelineStats {
    /// Total data points processed
    pub total_points: u64,
    
    /// Data points by type
    pub points_by_type: std::collections::HashMap<String, u64>,
    
    /// Average processing time
    pub avg_processing_time_ms: f64,
    
    /// Last processing timestamp
    pub last_processing: Option<DateTime<Utc>>,
    
    /// Buffer utilization
    pub buffer_utilization: f64,
}

impl Default for DataPipelineStats {
    fn default() -> Self {
        Self {
            total_points: 0,
            points_by_type: std::collections::HashMap::new(),
            avg_processing_time_ms: 0.0,
            last_processing: None,
            buffer_utilization: 0.0,
        }
    }
}

impl DataPipeline {
    /// Create a new data pipeline
    pub fn new(
        event_tx: mpsc::UnboundedSender<TelemetryEvent>,
        preprocessing_config: PreprocessingConfig,
        feature_config: FeatureConfig,
    ) -> Self {
        Self {
            event_tx,
            preprocessing_config,
            feature_config,
            data_buffer: Arc::new(Mutex::new(VecDeque::new())),
            max_buffer_size: 10000,
            feature_scaler: None,
            feature_importance: Arc::new(Mutex::new(Vec::new())),
            stats: DataPipelineStats::default(),
        }
    }
    
    /// Process events and extract features
    pub async fn process_events(&mut self, events: Vec<TelemetryEvent>) -> Result<Vec<DataPoint>> {
        let start_time = std::time::Instant::now();
        let mut data_points = Vec::new();
        
        for event in events {
            if let Some(data_point) = self.extract_features_from_event(&event).await? {
                data_points.push(data_point);
            }
        }
        
        // Apply preprocessing
        if !data_points.is_empty() {
            self.apply_preprocessing(&mut data_points).await?;
            
            // Apply feature engineering
            self.apply_feature_engineering(&mut data_points).await?;
            
            // Add to buffer
            self.add_to_buffer(data_points.clone()).await?;
            
            // Update statistics
            self.update_stats(start_time.elapsed(), data_points.len()).await?;
        }
        
        Ok(data_points)
    }
    
    /// Extract features from a telemetry event
    async fn extract_features_from_event(&self, event: &TelemetryEvent) -> Result<Option<DataPoint>> {
        let mut features = Vec::new();
        let mut metadata = serde_json::Map::new();
        
        // Extract basic features from event data
        if let Some(obj) = event.data.as_object() {
            for (key, value) in obj {
                if let Some(num) = value.as_f64() {
                    features.push(num);
                    metadata.insert(key.clone(), value.clone());
                }
            }
        }
        
        // Add time-based features
        if self.feature_config.enable_time_features {
            features.extend(self.extract_time_features(&event.timestamp));
        }
        
        // Add event type features
        features.extend(self.extract_event_type_features(&event.event_type));
        
        if features.is_empty() {
            return Ok(None);
        }
        
        Ok(Some(DataPoint {
            timestamp: event.timestamp,
            features,
            target: None, // Will be set later if needed
            metadata: serde_json::Value::Object(metadata),
        }))
    }
    
    /// Extract time-based features
    fn extract_time_features(&self, timestamp: &DateTime<Utc>) -> Vec<f64> {
        let mut features = Vec::new();
        
        // Hour of day (0-23)
        features.push(timestamp.hour() as f64);
        
        // Day of week (0-6, where 0 is Sunday)
        features.push(timestamp.weekday().num_days_from_sunday() as f64);
        
        // Day of month (1-31)
        features.push(timestamp.day() as f64);
        
        // Month (1-12)
        features.push(timestamp.month() as f64);
        
        // Unix timestamp (normalized)
        features.push(timestamp.timestamp() as f64 / 86400.0); // Days since epoch
        
        features
    }
    
    /// Extract event type features
    fn extract_event_type_features(&self, event_type: &crate::EventType) -> Vec<f64> {
        let mut features = vec![0.0; 10]; // One-hot encoding for event types
        
        let index = match event_type {
            crate::EventType::HighCpuUsage => 0,
            crate::EventType::HighMemoryUsage => 1,
            crate::EventType::HighLoadAverage => 2,
            crate::EventType::HighDiskIO => 3,
            crate::EventType::NetworkErrors => 4,
            crate::EventType::HighTemperature => 5,
            crate::EventType::ServiceFailure => 6,
            crate::EventType::PerformanceOptimization => 7,
            crate::EventType::AnomalyDetected => 8,
            crate::EventType::SecurityAlert => 9,
            _ => 0, // Default to first category
        };
        
        if index < features.len() {
            features[index] = 1.0;
        }
        
        features
    }
    
    /// Apply preprocessing to data points
    async fn apply_preprocessing(&mut self, data_points: &mut [DataPoint]) -> Result<()> {
        if data_points.is_empty() {
            return Ok(());
        }
        
        let feature_count = data_points[0].features.len();
        
        // Handle missing values
        if self.preprocessing_config.enable_missing_value_handling {
            self.handle_missing_values(data_points).await?;
        }
        
        // Detect and handle outliers
        if self.preprocessing_config.enable_outlier_detection {
            self.handle_outliers(data_points).await?;
        }
        
        // Apply normalization/standardization
        if self.preprocessing_config.enable_normalization || self.preprocessing_config.enable_standardization {
            self.fit_and_transform_features(data_points).await?;
        }
        
        // Apply smoothing
        if self.preprocessing_config.enable_smoothing {
            self.apply_smoothing(data_points).await?;
        }
        
        Ok(())
    }
    
    /// Handle missing values in data points
    async fn handle_missing_values(&self, data_points: &mut [DataPoint]) -> Result<()> {
        if data_points.is_empty() {
            return Ok(());
        }
        
        let feature_count = data_points[0].features.len();
        
        for feature_idx in 0..feature_count {
            let mut values: Vec<f64> = data_points.iter()
                .map(|dp| dp.features[feature_idx])
                .filter(|&v| !v.is_nan() && !v.is_infinite())
                .collect();
            
            if values.is_empty() {
                continue;
            }
            
            let fill_value = match self.preprocessing_config.missing_value_strategy {
                MissingValueStrategy::Mean => values.iter().sum::<f64>() / values.len() as f64,
                MissingValueStrategy::Median => {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    values[values.len() / 2]
                }
                MissingValueStrategy::Zero => 0.0,
                MissingValueStrategy::ForwardFill => values[0],
                MissingValueStrategy::BackwardFill => values[values.len() - 1],
                MissingValueStrategy::Interpolate => values.iter().sum::<f64>() / values.len() as f64,
            };
            
            for data_point in data_points.iter_mut() {
                if data_point.features[feature_idx].is_nan() || data_point.features[feature_idx].is_infinite() {
                    data_point.features[feature_idx] = fill_value;
                }
            }
        }
        
        Ok(())
    }
    
    /// Handle outliers in data points
    async fn handle_outliers(&self, data_points: &mut [DataPoint]) -> Result<()> {
        if data_points.is_empty() {
            return Ok(());
        }
        
        let feature_count = data_points[0].features.len();
        let threshold = self.preprocessing_config.outlier_threshold;
        
        for feature_idx in 0..feature_count {
            let values: Vec<f64> = data_points.iter()
                .map(|dp| dp.features[feature_idx])
                .collect();
            
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();
            
            for data_point in data_points.iter_mut() {
                let value = data_point.features[feature_idx];
                let z_score = (value - mean) / std_dev;
                
                if z_score.abs() > threshold {
                    // Cap the outlier
                    if z_score > 0.0 {
                        data_point.features[feature_idx] = mean + threshold * std_dev;
                    } else {
                        data_point.features[feature_idx] = mean - threshold * std_dev;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Fit and transform features using normalization/standardization
    async fn fit_and_transform_features(&mut self, data_points: &mut [DataPoint]) -> Result<()> {
        if data_points.is_empty() {
            return Ok(());
        }
        
        let feature_count = data_points[0].features.len();
        
        // Initialize scaler if not exists
        if self.feature_scaler.is_none() {
            self.feature_scaler = Some(FeatureScaler {
                means: vec![0.0; feature_count],
                stds: vec![1.0; feature_count],
                mins: vec![f64::INFINITY; feature_count],
                maxs: vec![f64::NEG_INFINITY; feature_count],
                n_samples: 0,
            });
        }
        
        let scaler = self.feature_scaler.as_mut().unwrap();
        
        // Update scaler statistics
        for data_point in data_points.iter() {
            for (feature_idx, &value) in data_point.features.iter().enumerate() {
                scaler.means[feature_idx] += value;
                scaler.mins[feature_idx] = scaler.mins[feature_idx].min(value);
                scaler.maxs[feature_idx] = scaler.maxs[feature_idx].max(value);
            }
            scaler.n_samples += 1;
        }
        
        // Calculate means
        for mean in &mut scaler.means {
            *mean /= scaler.n_samples as f64;
        }
        
        // Calculate standard deviations
        for data_point in data_points.iter() {
            for (feature_idx, &value) in data_point.features.iter().enumerate() {
                let diff = value - scaler.means[feature_idx];
                scaler.stds[feature_idx] += diff * diff;
            }
        }
        
        for std in &mut scaler.stds {
            *std = (*std / scaler.n_samples as f64).sqrt();
            if *std == 0.0 {
                *std = 1.0; // Avoid division by zero
            }
        }
        
        // Apply transformation
        for data_point in data_points.iter_mut() {
            for (feature_idx, value) in data_point.features.iter_mut().enumerate() {
                if self.preprocessing_config.enable_standardization {
                    *value = (*value - scaler.means[feature_idx]) / scaler.stds[feature_idx];
                } else if self.preprocessing_config.enable_normalization {
                    let range = scaler.maxs[feature_idx] - scaler.mins[feature_idx];
                    if range > 0.0 {
                        *value = (*value - scaler.mins[feature_idx]) / range;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply smoothing to data points
    async fn apply_smoothing(&self, data_points: &mut [DataPoint]) -> Result<()> {
        if data_points.len() < self.preprocessing_config.smoothing_window {
            return Ok(());
        }
        
        let window_size = self.preprocessing_config.smoothing_window;
        let feature_count = data_points[0].features.len();
        
        for feature_idx in 0..feature_count {
            for i in window_size..data_points.len() {
                let window_sum: f64 = data_points[i-window_size..i]
                    .iter()
                    .map(|dp| dp.features[feature_idx])
                    .sum();
                data_points[i].features[feature_idx] = window_sum / window_size as f64;
            }
        }
        
        Ok(())
    }
    
    /// Apply feature engineering to data points
    async fn apply_feature_engineering(&mut self, data_points: &mut [DataPoint]) -> Result<()> {
        if data_points.is_empty() {
            return Ok(());
        }
        
        // Apply rolling features
        if self.feature_config.enable_rolling_features {
            self.add_rolling_features(data_points).await?;
        }
        
        // Apply lag features
        if self.feature_config.enable_lag_features {
            self.add_lag_features(data_points).await?;
        }
        
        // Apply interaction features
        if self.feature_config.enable_interaction_features {
            self.add_interaction_features(data_points).await?;
        }
        
        // Apply polynomial features
        if self.feature_config.enable_polynomial_features {
            self.add_polynomial_features(data_points).await?;
        }
        
        // Apply statistical features
        if self.feature_config.enable_statistical_features {
            self.add_statistical_features(data_points).await?;
        }
        
        Ok(())
    }
    
    /// Add rolling features
    async fn add_rolling_features(&self, data_points: &mut [DataPoint]) -> Result<()> {
        for window_size in &self.feature_config.rolling_windows {
            if data_points.len() < *window_size {
                continue;
            }
            
            let feature_count = data_points[0].features.len();
            
            for feature_idx in 0..feature_count {
                for i in *window_size..data_points.len() {
                    let window_values: Vec<f64> = data_points[i-window_size..i]
                        .iter()
                        .map(|dp| dp.features[feature_idx])
                        .collect();
                    
                    // Add rolling mean
                    let mean = window_values.iter().sum::<f64>() / window_values.len() as f64;
                    data_points[i].features.push(mean);
                    
                    // Add rolling std
                    let variance = window_values.iter()
                        .map(|&v| (v - mean).powi(2))
                        .sum::<f64>() / window_values.len() as f64;
                    let std = variance.sqrt();
                    data_points[i].features.push(std);
                }
            }
        }
        
        Ok(())
    }
    
    /// Add lag features
    async fn add_lag_features(&self, data_points: &mut [DataPoint]) -> Result<()> {
        for lag in &self.feature_config.lag_periods {
            if data_points.len() <= *lag {
                continue;
            }
            
            let feature_count = data_points[0].features.len();
            
            for feature_idx in 0..feature_count {
                for i in *lag..data_points.len() {
                    let lag_value = data_points[i - lag].features[feature_idx];
                    data_points[i].features.push(lag_value);
                }
            }
        }
        
        Ok(())
    }
    
    /// Add interaction features
    async fn add_interaction_features(&self, data_points: &mut [DataPoint]) -> Result<()> {
        if data_points.is_empty() {
            return Ok(());
        }
        
        let original_feature_count = data_points[0].features.len();
        
        for i in 0..original_feature_count {
            for j in i+1..original_feature_count {
                for data_point in data_points.iter_mut() {
                    let interaction = data_point.features[i] * data_point.features[j];
                    data_point.features.push(interaction);
                }
            }
        }
        
        Ok(())
    }
    
    /// Add polynomial features
    async fn add_polynomial_features(&self, data_points: &mut [DataPoint]) -> Result<()> {
        if data_points.is_empty() {
            return Ok(());
        }
        
        let original_feature_count = data_points[0].features.len();
        let degree = self.feature_config.polynomial_degree;
        
        for feature_idx in 0..original_feature_count {
            for d in 2..=degree {
                for data_point in data_points.iter_mut() {
                    let polynomial = data_point.features[feature_idx].powi(d as i32);
                    data_point.features.push(polynomial);
                }
            }
        }
        
        Ok(())
    }
    
    /// Add statistical features
    async fn add_statistical_features(&self, data_points: &mut [DataPoint]) -> Result<()> {
        let window_size = self.feature_config.statistical_window;
        if data_points.len() < window_size {
            return Ok(());
        }
        
        let feature_count = data_points[0].features.len();
        
        for feature_idx in 0..feature_count {
            for i in window_size..data_points.len() {
                let window_values: Vec<f64> = data_points[i-window_size..i]
                    .iter()
                    .map(|dp| dp.features[feature_idx])
                    .collect();
                
                // Add various statistical features
                let mean = window_values.iter().sum::<f64>() / window_values.len() as f64;
                data_points[i].features.push(mean);
                
                let variance = window_values.iter()
                    .map(|&v| (v - mean).powi(2))
                    .sum::<f64>() / window_values.len() as f64;
                data_points[i].features.push(variance);
                
                let min = window_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                data_points[i].features.push(min);
                
                let max = window_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                data_points[i].features.push(max);
            }
        }
        
        Ok(())
    }
    
    /// Add data points to buffer
    async fn add_to_buffer(&mut self, data_points: Vec<DataPoint>) -> Result<()> {
        let mut buffer = self.data_buffer.lock().unwrap();
        
        for data_point in data_points {
            buffer.push_back(data_point);
            
            // Remove old data points if buffer is full
            while buffer.len() > self.max_buffer_size {
                buffer.pop_front();
            }
        }
        
        Ok(())
    }
    
    /// Get data from buffer
    pub async fn get_data_from_buffer(&self, count: usize) -> Vec<DataPoint> {
        let buffer = self.data_buffer.lock().unwrap();
        buffer.iter().rev().take(count).cloned().collect()
    }
    
    /// Update statistics
    async fn update_stats(&mut self, processing_time: std::time::Duration, points_processed: usize) -> Result<()> {
        self.stats.total_points += points_processed as u64;
        self.stats.last_processing = Some(Utc::now());
        
        let processing_time_ms = processing_time.as_millis() as f64;
        self.stats.avg_processing_time_ms = (self.stats.avg_processing_time_ms + processing_time_ms) / 2.0;
        
        let buffer = self.data_buffer.lock().unwrap();
        self.stats.buffer_utilization = buffer.len() as f64 / self.max_buffer_size as f64;
        
        Ok(())
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> DataPipelineStats {
        self.stats.clone()
    }
    
    /// Get feature importance
    pub fn get_feature_importance(&self) -> Vec<FeatureImportance> {
        self.feature_importance.lock().unwrap().clone()
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            enable_normalization: true,
            enable_standardization: false,
            enable_outlier_detection: true,
            outlier_threshold: 3.0,
            enable_missing_value_handling: true,
            missing_value_strategy: MissingValueStrategy::Mean,
            enable_smoothing: false,
            smoothing_window: 5,
        }
    }
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            enable_rolling_features: true,
            rolling_windows: vec![5, 10, 20],
            enable_lag_features: true,
            lag_periods: vec![1, 2, 5],
            enable_interaction_features: false,
            enable_polynomial_features: false,
            polynomial_degree: 2,
            enable_time_features: true,
            enable_statistical_features: true,
            statistical_window: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_data_pipeline_creation() {
        let (event_tx, _) = mpsc::unbounded_channel();
        let pipeline = DataPipeline::new(
            event_tx,
            PreprocessingConfig::default(),
            FeatureConfig::default(),
        );
        
        assert_eq!(pipeline.get_stats().total_points, 0);
    }
    
    #[tokio::test]
    async fn test_feature_extraction() {
        let (event_tx, _) = mpsc::unbounded_channel();
        let mut pipeline = DataPipeline::new(
            event_tx,
            PreprocessingConfig::default(),
            FeatureConfig::default(),
        );
        
        let event = TelemetryEvent {
            event_type: crate::EventType::HighCpuUsage,
            timestamp: Utc::now(),
            data: serde_json::json!({"cpu_usage": 95.5, "memory_usage": 80.2}),
            severity: Some(crate::EventSeverity::Warning),
            source: Some("test".to_string()),
            id: Some("test-1".to_string()),
        };
        
        let data_point = pipeline.extract_features_from_event(&event).await.unwrap();
        assert!(data_point.is_some());
        assert!(!data_point.unwrap().features.is_empty());
    }
} 