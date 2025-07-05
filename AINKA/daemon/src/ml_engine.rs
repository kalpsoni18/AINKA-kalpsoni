use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;
use crate::{TelemetryEvent, DataPoint, AinkaError};

/// ML model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Mean squared error
    pub mse: f64,
    
    /// R-squared coefficient
    pub r2: f64,
    
    /// Mean absolute error
    pub mae: f64,
    
    /// Root mean squared error
    pub rmse: f64,
    
    /// Training samples count
    pub training_samples: usize,
    
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

/// Regression pipeline for system load prediction
pub struct RegressionPipeline {
    /// Model parameters (weights)
    weights: Array1<f64>,
    
    /// Bias term
    bias: f64,
    
    /// Learning rate
    learning_rate: f64,
    
    /// Momentum coefficient
    momentum: f64,
    
    /// Regularization strength
    regularization: f64,
    
    /// Feature scaling parameters
    feature_means: Array1<f64>,
    feature_stds: Array1<f64>,
    
    /// Training history
    training_history: VecDeque<TrainingStep>,
    
    /// Maximum history size
    max_history_size: usize,
    
    /// Target variable name
    target_name: String,
    
    /// Feature names
    feature_names: Vec<String>,
    
    /// Performance metrics
    performance: ModelPerformance,
    
    /// Adaptive learning rate parameters
    adaptive_lr: AdaptiveLearningRate,
    
    /// Feature importance tracking
    feature_importance: Arc<Mutex<Vec<FeatureImportance>>>,
    
    /// Model state
    model_state: ModelState,
}

/// Training step information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStep {
    /// Step number
    pub step: usize,
    
    /// Loss value
    pub loss: f64,
    
    /// Learning rate used
    pub learning_rate: f64,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Feature importance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Feature name
    pub name: String,
    
    /// Importance score
    pub importance: f64,
    
    /// Last update
    pub last_update: DateTime<Utc>,
}

/// Adaptive learning rate
#[derive(Debug, Clone)]
pub struct AdaptiveLearningRate {
    /// Initial learning rate
    initial_lr: f64,
    
    /// Decay factor
    decay_factor: f64,
    
    /// Minimum learning rate
    min_lr: f64,
    
    /// Patience for reducing learning rate
    patience: usize,
    
    /// Current patience counter
    patience_counter: usize,
    
    /// Best loss so far
    best_loss: f64,
}

/// Model state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelState {
    Uninitialized,
    Training,
    Trained,
    Failed,
}

impl RegressionPipeline {
    /// Create a new regression pipeline
    pub fn new(feature_count: usize, target_name: String) -> Self {
        let mut rng = rand::thread_rng();
        
        // Initialize weights with small random values
        let weights = Array1::from_iter((0..feature_count).map(|_| rng.gen_range(-0.1..0.1)));
        
        Self {
            weights,
            bias: 0.0,
            learning_rate: 0.01,
            momentum: 0.9,
            regularization: 0.001,
            feature_means: Array1::zeros(feature_count),
            feature_stds: Array1::ones(feature_count),
            training_history: VecDeque::new(),
            max_history_size: 1000,
            target_name,
            feature_names: (0..feature_count).map(|i| format!("feature_{}", i)).collect(),
            performance: ModelPerformance {
                mse: f64::INFINITY,
                r2: 0.0,
                mae: f64::INFINITY,
                rmse: f64::INFINITY,
                training_samples: 0,
                last_update: Utc::now(),
            },
            adaptive_lr: AdaptiveLearningRate {
                initial_lr: 0.01,
                decay_factor: 0.95,
                min_lr: 0.0001,
                patience: 10,
                patience_counter: 0,
                best_loss: f64::INFINITY,
            },
            feature_importance: Arc::new(Mutex::new(Vec::new())),
            model_state: ModelState::Uninitialized,
        }
    }
    
    /// Process events and update model
    pub async fn process_events(&mut self, events: Vec<TelemetryEvent>) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        
        // Convert events to data points
        let data_points = self.events_to_data_points(events).await?;
        
        // Update model with new data
        self.update_model(data_points).await?;
        
        Ok(())
    }
    
    /// Make predictions
    pub fn predict(&self, features: &[f64]) -> Result<Vec<f64>> {
        if self.model_state != ModelState::Trained {
            return Err(AinkaError::MachineLearning("Model not trained".to_string()).into());
        }
        
        if features.len() != self.weights.len() {
            return Err(AinkaError::MachineLearning(format!(
                "Expected {} features, got {}", self.weights.len(), features.len()
            )).into());
        }
        
        let features_array = Array1::from_vec(features.to_vec());
        let normalized_features = self.normalize_features(&features_array)?;
        
        let prediction = self.predict_single(&normalized_features)?;
        Ok(vec![prediction])
    }
    
    /// Get model performance metrics
    pub fn get_performance_metrics(&self) -> ModelPerformance {
        self.performance.clone()
    }
    
    /// Get training history
    pub fn get_training_history(&self) -> Vec<TrainingStep> {
        self.training_history.iter().cloned().collect()
    }
    
    /// Get feature importance
    pub fn get_feature_importance(&self) -> Vec<FeatureImportance> {
        self.feature_importance.lock().unwrap().clone()
    }
    
    /// Save model to file
    pub fn save_model(&self, path: &str) -> Result<()> {
        let model_data = ModelData {
            weights: self.weights.clone(),
            bias: self.bias,
            feature_means: self.feature_means.clone(),
            feature_stds: self.feature_stds.clone(),
            feature_names: self.feature_names.clone(),
            target_name: self.target_name.clone(),
            performance: self.performance.clone(),
            model_state: self.model_state.clone(),
        };
        
        let json = serde_json::to_string_pretty(&model_data)?;
        std::fs::write(path, json)?;
        
        log::info!("Model saved to: {}", path);
        Ok(())
    }
    
    /// Load model from file
    pub fn load_model(&mut self, path: &str) -> Result<()> {
        let json = std::fs::read_to_string(path)?;
        let model_data: ModelData = serde_json::from_str(&json)?;
        
        self.weights = model_data.weights;
        self.bias = model_data.bias;
        self.feature_means = model_data.feature_means;
        self.feature_stds = model_data.feature_stds;
        self.feature_names = model_data.feature_names;
        self.target_name = model_data.target_name;
        self.performance = model_data.performance;
        self.model_state = model_data.model_state;
        
        log::info!("Model loaded from: {}", path);
        Ok(())
    }
    
    /// Convert events to data points
    async fn events_to_data_points(&self, events: Vec<TelemetryEvent>) -> Result<Vec<DataPoint>> {
        let mut data_points = Vec::new();
        
        for event in events {
            if let Some(data_point) = self.extract_features_from_event(&event).await? {
                data_points.push(data_point);
            }
        }
        
        Ok(data_points)
    }
    
    /// Extract features from event
    async fn extract_features_from_event(&self, event: &TelemetryEvent) -> Result<Option<DataPoint>> {
        let mut features = Vec::new();
        let mut metadata = serde_json::Map::new();
        
        // Extract numeric features from event data
        if let Some(obj) = event.data.as_object() {
            for (key, value) in obj {
                if let Some(num) = value.as_f64() {
                    features.push(num);
                    metadata.insert(key.clone(), value.clone());
                }
            }
        }
        
        // Add time-based features
        features.extend(self.extract_time_features(&event.timestamp));
        
        // Add event type features
        features.extend(self.extract_event_type_features(&event.event_type));
        
        if features.is_empty() {
            return Ok(None);
        }
        
        // Pad or truncate features to match model dimensions
        while features.len() < self.weights.len() {
            features.push(0.0);
        }
        if features.len() > self.weights.len() {
            features.truncate(self.weights.len());
        }
        
        Ok(Some(DataPoint {
            timestamp: event.timestamp,
            features,
            target: None, // Will be set later
            metadata: serde_json::Value::Object(metadata),
        }))
    }
    
    /// Extract time features
    fn extract_time_features(&self, timestamp: &DateTime<Utc>) -> Vec<f64> {
        vec![
            timestamp.hour() as f64 / 24.0, // Normalized hour
            timestamp.weekday().num_days_from_sunday() as f64 / 7.0, // Normalized day of week
            timestamp.day() as f64 / 31.0, // Normalized day of month
            timestamp.month() as f64 / 12.0, // Normalized month
        ]
    }
    
    /// Extract event type features
    fn extract_event_type_features(&self, event_type: &crate::EventType) -> Vec<f64> {
        let mut features = vec![0.0; 5]; // Simplified event type encoding
        
        let index = match event_type {
            crate::EventType::HighCpuUsage => 0,
            crate::EventType::HighMemoryUsage => 1,
            crate::EventType::HighLoadAverage => 2,
            crate::EventType::HighDiskIO => 3,
            crate::EventType::NetworkErrors => 4,
            _ => 0,
        };
        
        if index < features.len() {
            features[index] = 1.0;
        }
        
        features
    }
    
    /// Update model with new data
    async fn update_model(&mut self, data_points: Vec<DataPoint>) -> Result<()> {
        if data_points.is_empty() {
            return Ok(());
        }
        
        self.model_state = ModelState::Training;
        
        // Prepare training data
        let (features, targets) = self.prepare_training_data(data_points).await?;
        
        if features.is_empty() {
            return Ok(());
        }
        
        // Update feature scaling parameters
        self.update_feature_scaling(&features).await?;
        
        // Normalize features
        let normalized_features = self.normalize_features_batch(&features).await?;
        
        // Train model
        self.train_model(&normalized_features, &targets).await?;
        
        // Update performance metrics
        self.update_performance_metrics(&normalized_features, &targets).await?;
        
        // Update feature importance
        self.update_feature_importance().await?;
        
        self.model_state = ModelState::Trained;
        
        Ok(())
    }
    
    /// Prepare training data
    async fn prepare_training_data(&self, data_points: Vec<DataPoint>) -> Result<(Array2<f64>, Array1<f64>)> {
        let mut features = Vec::new();
        let mut targets = Vec::new();
        
        for data_point in data_points {
            if data_point.features.len() == self.weights.len() {
                features.push(data_point.features);
                
                // Use a simple heuristic for target: average of first few features
                let target = data_point.features.iter().take(3).sum::<f64>() / 3.0;
                targets.push(target);
            }
        }
        
        if features.is_empty() {
            return Ok((Array2::zeros((0, self.weights.len())), Array1::zeros(0)));
        }
        
        let features_array = Array2::from_shape_vec(
            (features.len(), self.weights.len()),
            features.into_iter().flatten().collect()
        )?;
        
        let targets_array = Array1::from_vec(targets);
        
        Ok((features_array, targets_array))
    }
    
    /// Update feature scaling parameters
    async fn update_feature_scaling(&mut self, features: &Array2<f64>) -> Result<()> {
        if features.nrows() == 0 {
            return Ok(());
        }
        
        for j in 0..features.ncols() {
            let column = features.column(j);
            let mean = column.mean().unwrap_or(0.0);
            let variance = column.var(0.0);
            let std = variance.sqrt();
            
            // Update running statistics
            let n = features.nrows() as f64;
            let old_mean = self.feature_means[j];
            let old_std = self.feature_stds[j];
            
            self.feature_means[j] = (old_mean * (n - 1.0) + mean * n) / (2.0 * n - 1.0);
            self.feature_stds[j] = ((old_std * old_std * (n - 1.0) + variance * n) / (2.0 * n - 1.0)).sqrt();
            
            // Avoid division by zero
            if self.feature_stds[j] < 1e-8 {
                self.feature_stds[j] = 1.0;
            }
        }
        
        Ok(())
    }
    
    /// Normalize features
    fn normalize_features(&self, features: &Array1<f64>) -> Result<Array1<f64>> {
        if features.len() != self.feature_means.len() {
            return Err(AinkaError::MachineLearning("Feature dimension mismatch".to_string()).into());
        }
        
        let normalized = (features - &self.feature_means) / &self.feature_stds;
        Ok(normalized)
    }
    
    /// Normalize features batch
    async fn normalize_features_batch(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        if features.ncols() != self.feature_means.len() {
            return Err(AinkaError::MachineLearning("Feature dimension mismatch".to_string()).into());
        }
        
        let mut normalized = features.clone();
        for j in 0..features.ncols() {
            let column = normalized.column_mut(j);
            for i in 0..column.len() {
                column[i] = (column[i] - self.feature_means[j]) / self.feature_stds[j];
            }
        }
        
        Ok(normalized)
    }
    
    /// Train model using gradient descent
    async fn train_model(&mut self, features: &Array2<f64>, targets: &Array1<f64>) -> Result<()> {
        if features.nrows() == 0 || targets.len() == 0 {
            return Ok(());
        }
        
        let n_samples = features.nrows();
        let n_features = features.ncols();
        
        // Initialize momentum
        let mut weight_momentum = Array1::zeros(n_features);
        let mut bias_momentum = 0.0;
        
        // Training loop
        let max_epochs = 100;
        let batch_size = (n_samples / 10).max(1);
        
        for epoch in 0..max_epochs {
            let mut total_loss = 0.0;
            
            // Mini-batch training
            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_features = features.slice(s![batch_start..batch_end, ..]);
                let batch_targets = targets.slice(s![batch_start..batch_end]);
                
                // Forward pass
                let predictions = self.predict_batch(&batch_features)?;
                
                // Compute loss
                let loss = self.compute_loss(&predictions, &batch_targets)?;
                total_loss += loss;
                
                // Compute gradients
                let (weight_gradients, bias_gradient) = self.compute_gradients(
                    &batch_features, &predictions, &batch_targets
                )?;
                
                // Update weights with momentum
                weight_momentum = &weight_momentum * self.momentum + &weight_gradients * (1.0 - self.momentum);
                bias_momentum = bias_momentum * self.momentum + bias_gradient * (1.0 - self.momentum);
                
                self.weights -= &weight_momentum * self.learning_rate;
                self.bias -= bias_momentum * self.learning_rate;
            }
            
            // Record training step
            let avg_loss = total_loss / (n_samples / batch_size) as f64;
            self.record_training_step(epoch, avg_loss).await?;
            
            // Adaptive learning rate
            self.update_learning_rate(avg_loss).await?;
            
            // Early stopping
            if avg_loss < 1e-6 {
                break;
            }
        }
        
        Ok(())
    }
    
    /// Predict batch
    fn predict_batch(&self, features: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let mut predictions = Array1::zeros(features.nrows());
        
        for i in 0..features.nrows() {
            let feature_row = features.row(i);
            predictions[i] = self.predict_single(&feature_row.to_owned())?;
        }
        
        Ok(predictions)
    }
    
    /// Predict single sample
    fn predict_single(&self, features: &Array1<f64>) -> Result<f64> {
        if features.len() != self.weights.len() {
            return Err(AinkaError::MachineLearning("Feature dimension mismatch".to_string()).into());
        }
        
        let prediction = features.dot(&self.weights) + self.bias;
        Ok(prediction)
    }
    
    /// Compute loss
    fn compute_loss(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(AinkaError::MachineLearning("Prediction/target dimension mismatch".to_string()).into());
        }
        
        let mse = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        // Add regularization
        let regularization_term = self.regularization * self.weights.iter().map(|w| w.powi(2)).sum::<f64>();
        
        Ok(mse + regularization_term)
    }
    
    /// Compute gradients
    fn compute_gradients(
        &self,
        features: &ArrayView2<f64>,
        predictions: &Array1<f64>,
        targets: &Array1<f64>,
    ) -> Result<(Array1<f64>, f64)> {
        let n_samples = features.nrows() as f64;
        
        // Compute gradients
        let mut weight_gradients = Array1::zeros(self.weights.len());
        let mut bias_gradient = 0.0;
        
        for i in 0..features.nrows() {
            let error = predictions[i] - targets[i];
            let feature_row = features.row(i);
            
            // Weight gradients
            for j in 0..self.weights.len() {
                weight_gradients[j] += error * feature_row[j];
            }
            
            // Bias gradient
            bias_gradient += error;
        }
        
        // Average gradients
        weight_gradients /= n_samples;
        bias_gradient /= n_samples;
        
        // Add regularization to weight gradients
        weight_gradients += &self.weights * (2.0 * self.regularization);
        
        Ok((weight_gradients, bias_gradient))
    }
    
    /// Record training step
    async fn record_training_step(&mut self, step: usize, loss: f64) -> Result<()> {
        let training_step = TrainingStep {
            step,
            loss,
            learning_rate: self.learning_rate,
            timestamp: Utc::now(),
        };
        
        self.training_history.push_back(training_step);
        
        // Remove old training steps if history is too large
        while self.training_history.len() > self.max_history_size {
            self.training_history.pop_front();
        }
        
        Ok(())
    }
    
    /// Update learning rate adaptively
    async fn update_learning_rate(&mut self, current_loss: f64) -> Result<()> {
        if current_loss < self.adaptive_lr.best_loss {
            self.adaptive_lr.best_loss = current_loss;
            self.adaptive_lr.patience_counter = 0;
        } else {
            self.adaptive_lr.patience_counter += 1;
        }
        
        if self.adaptive_lr.patience_counter >= self.adaptive_lr.patience {
            self.learning_rate *= self.adaptive_lr.decay_factor;
            self.learning_rate = self.learning_rate.max(self.adaptive_lr.min_lr);
            self.adaptive_lr.patience_counter = 0;
        }
        
        Ok(())
    }
    
    /// Update performance metrics
    async fn update_performance_metrics(&mut self, features: &Array2<f64>, targets: &Array1<f64>) -> Result<()> {
        if features.nrows() == 0 || targets.len() == 0 {
            return Ok(());
        }
        
        let predictions = self.predict_batch(&features.view())?;
        
        // Compute metrics
        let mse = self.compute_mse(&predictions, targets)?;
        let mae = self.compute_mae(&predictions, targets)?;
        let r2 = self.compute_r2(&predictions, targets)?;
        let rmse = mse.sqrt();
        
        self.performance = ModelPerformance {
            mse,
            r2,
            mae,
            rmse,
            training_samples: features.nrows(),
            last_update: Utc::now(),
        };
        
        Ok(())
    }
    
    /// Compute MSE
    fn compute_mse(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(AinkaError::MachineLearning("Dimension mismatch".to_string()).into());
        }
        
        let mse = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        Ok(mse)
    }
    
    /// Compute MAE
    fn compute_mae(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(AinkaError::MachineLearning("Dimension mismatch".to_string()).into());
        }
        
        let mae = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>() / predictions.len() as f64;
        
        Ok(mae)
    }
    
    /// Compute R-squared
    fn compute_r2(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(AinkaError::MachineLearning("Dimension mismatch".to_string()).into());
        }
        
        let mean_target = targets.mean().unwrap_or(0.0);
        let ss_res: f64 = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        let ss_tot: f64 = targets.iter()
            .map(|t| (t - mean_target).powi(2))
            .sum();
        
        let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
        Ok(r2)
    }
    
    /// Update feature importance
    async fn update_feature_importance(&mut self) -> Result<()> {
        let mut importance = self.feature_importance.lock().unwrap();
        importance.clear();
        
        for (i, &weight) in self.weights.iter().enumerate() {
            let feature_name = if i < self.feature_names.len() {
                self.feature_names[i].clone()
            } else {
                format!("feature_{}", i)
            };
            
            importance.push(FeatureImportance {
                name: feature_name,
                importance: weight.abs(),
                last_update: Utc::now(),
            });
        }
        
        // Sort by importance
        importance.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        
        Ok(())
    }
}

/// Model data for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelData {
    weights: Array1<f64>,
    bias: f64,
    feature_means: Array1<f64>,
    feature_stds: Array1<f64>,
    feature_names: Vec<String>,
    target_name: String,
    performance: ModelPerformance,
    model_state: ModelState,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_regression_pipeline_creation() {
        let pipeline = RegressionPipeline::new(10, "test_target".to_string());
        assert_eq!(pipeline.weights.len(), 10);
        assert_eq!(pipeline.target_name, "test_target");
    }
    
    #[tokio::test]
    async fn test_prediction() {
        let mut pipeline = RegressionPipeline::new(3, "test_target".to_string());
        pipeline.model_state = ModelState::Trained;
        
        let features = vec![1.0, 2.0, 3.0];
        let predictions = pipeline.predict(&features).unwrap();
        assert_eq!(predictions.len(), 1);
    }
    
    #[tokio::test]
    async fn test_model_save_load() {
        let mut pipeline = RegressionPipeline::new(5, "test_target".to_string());
        pipeline.model_state = ModelState::Trained;
        
        let temp_path = "/tmp/test_model.json";
        pipeline.save_model(temp_path).unwrap();
        
        let mut new_pipeline = RegressionPipeline::new(5, "test_target".to_string());
        new_pipeline.load_model(temp_path).unwrap();
        
        assert_eq!(new_pipeline.weights.len(), 5);
        assert_eq!(new_pipeline.model_state, ModelState::Trained);
        
        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }
} 