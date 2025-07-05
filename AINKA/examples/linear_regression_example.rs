use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use anyhow::Result;
use log::{info, warn, error, debug};
use serde_json::json;

use ainka_daemon::{
    ml_engine::{OnlineLinearRegression, RegressionPipeline, FeatureEngineer},
    data_pipeline::{DataPipeline, PreprocessingConfig, FeatureConfig, MissingValueStrategy, NormalizationStrategy},
    telemetry_hub::{TelemetryEvent, EventType},
    ebpf_manager::EbpfManager,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    info!("AINKA Intelligent Linear Regression Example");
    info!("==========================================");

    // Create event channel
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<TelemetryEvent>();

    // Initialize data pipeline with intelligent configuration
    let preprocessing_config = PreprocessingConfig {
        buffer_size: 10000,
        batch_size: 100,
        sampling_rate: 1.0, // Process all events
        outlier_threshold: 3.0, // 3-sigma rule
        missing_value_strategy: MissingValueStrategy::Mean,
        normalization_strategy: NormalizationStrategy::StandardScaler,
    };

    let feature_config = FeatureConfig {
        window_sizes: vec![5, 10, 30, 60, 300], // Multiple time windows
        aggregation_functions: vec![
            ainka_daemon::data_pipeline::AggregationFunction::Count,
            ainka_daemon::data_pipeline::AggregationFunction::Sum,
            ainka_daemon::data_pipeline::AggregationFunction::Mean,
            ainka_daemon::data_pipeline::AggregationFunction::Std,
            ainka_daemon::data_pipeline::AggregationFunction::Min,
            ainka_daemon::data_pipeline::AggregationFunction::Max,
        ],
        interaction_features: true,
        polynomial_features: true,
        polynomial_degree: 2,
        lag_features: vec![1, 2, 5, 10],
        rolling_features: true,
        rolling_window: 10,
    };

    let mut data_pipeline = DataPipeline::new(
        event_tx.clone(),
        preprocessing_config,
        feature_config,
    );

    // Initialize eBPF manager
    let ebpf_config = ainka_daemon::ebpf_manager::EbpfConfig::default();
    let mut ebpf_manager = EbpfManager::new(event_tx.clone(), ebpf_config);
    
    // Initialize eBPF programs
    ebpf_manager.initialize().await?;
    info!("eBPF monitoring programs initialized");

    // Create regression pipeline for predicting system load
    let target_feature = "system_load".to_string();
    let mut regression_pipeline = RegressionPipeline::new(50, target_feature.clone());
    info!("Regression pipeline created for target: {}", target_feature);

    // Event processing loop
    let mut event_buffer = Vec::new();
    let mut iteration = 0;
    
    info!("Starting intelligent data processing and learning...");
    
    loop {
        // Collect events from eBPF
        while let Ok(event) = event_rx.try_recv() {
            event_buffer.push(event);
        }

        if !event_buffer.is_empty() {
            // Process events through data pipeline
            data_pipeline.process_events(event_buffer.clone()).await?;
            
            // Update regression model
            regression_pipeline.process_events(event_buffer.clone()).await?;
            
            // Get performance metrics
            let metrics = regression_pipeline.get_performance_metrics();
            let model_stats = regression_pipeline.get_model_stats();
            
            // Log progress every 100 iterations
            if iteration % 100 == 0 {
                info!("Iteration {}: MSE={:.6}, MAE={:.6}, R²={:.6}, Samples={}", 
                    iteration, metrics.mse, metrics.mae, metrics.r2, model_stats.sample_count);
                
                // Show feature importance
                let feature_importance = regression_pipeline.get_feature_importance();
                info!("Top 5 most important features:");
                let mut importance_pairs: Vec<(usize, f64)> = feature_importance.iter()
                    .enumerate()
                    .map(|(i, &val)| (i, val))
                    .collect();
                importance_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                for (i, (feature_idx, importance)) in importance_pairs.iter().take(5).enumerate() {
                    info!("  {}. Feature {}: {:.4}", i + 1, feature_idx, importance);
                }
            }
            
            // Make predictions on recent data
            if event_buffer.len() >= 10 {
                let recent_events = event_buffer.iter().rev().take(10).cloned().collect::<Vec<_>>();
                let predictions = regression_pipeline.predict(&recent_events)?;
                
                if !predictions.is_empty() {
                    info!("Recent predictions: {:?}", predictions.to_vec());
                }
            }
            
            iteration += 1;
        }
        
        // Clear buffer for next iteration
        event_buffer.clear();
        
        // Sleep to prevent busy waiting
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Example: Stop after 1000 iterations
        if iteration >= 1000 {
            info!("Completed 1000 iterations. Finalizing...");
            break;
        }
    }

    // Final model evaluation
    info!("Final Model Evaluation");
    info!("=====================");
    
    let final_metrics = regression_pipeline.get_performance_metrics();
    let final_stats = regression_pipeline.get_model_stats();
    let final_importance = regression_pipeline.get_feature_importance();
    
    info!("Final Performance:");
    info!("  MSE: {:.6}", final_metrics.mse);
    info!("  MAE: {:.6}", final_metrics.mae);
    info!("  R²: {:.6}", final_metrics.r2);
    info!("  Total Samples: {}", final_stats.sample_count);
    info!("  Learning Rate: {:.6}", final_stats.learning_rate);
    
    info!("Feature Importance Analysis:");
    let mut importance_pairs: Vec<(usize, f64)> = final_importance.iter()
        .enumerate()
        .map(|(i, &val)| (i, val))
        .collect();
    importance_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    for (i, (feature_idx, importance)) in importance_pairs.iter().take(10).enumerate() {
        info!("  {}. Feature {}: {:.4}", i + 1, feature_idx, importance);
    }
    
    // Get prediction history for analysis
    let prediction_history = regression_pipeline.get_prediction_history();
    if !prediction_history.is_empty() {
        info!("Prediction History Analysis:");
        info!("  Total Predictions: {}", prediction_history.len());
        
        let errors: Vec<f64> = prediction_history.iter()
            .map(|(_, actual, predicted)| (actual - predicted).abs())
            .collect();
        
        let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let max_error = errors.iter().fold(0.0, |a, &b| a.max(b));
        let min_error = errors.iter().fold(f64::INFINITY, |a, &b| a.min(*b));
        
        info!("  Average Error: {:.6}", avg_error);
        info!("  Max Error: {:.6}", max_error);
        info!("  Min Error: {:.6}", min_error);
    }
    
    // Get data pipeline statistics
    let data_stats = data_pipeline.get_data_stats();
    info!("Data Pipeline Statistics:");
    info!("  Total Events Processed: {}", data_stats.total_events);
    info!("  Processing Time: {:?}", data_stats.processing_time);
    info!("  Events by Type: {:?}", data_stats.events_by_type);
    
    // Get feature cache for analysis
    let feature_cache = data_pipeline.get_feature_cache();
    info!("Feature Cache Statistics:");
    info!("  Total Features: {}", feature_cache.len());
    
    for (feature_name, cache) in feature_cache.iter().take(5) {
        info!("  {}: {} values, last update: {:?}", 
            feature_name, cache.get_values().len(), cache.last_update);
    }
    
    // Example: Make predictions on synthetic data
    info!("Making predictions on synthetic data...");
    
    let synthetic_events = generate_synthetic_events(100);
    let predictions = regression_pipeline.predict(&synthetic_events)?;
    
    info!("Synthetic data predictions:");
    info!("  Predictions: {:?}", predictions.to_vec());
    info!("  Mean prediction: {:.6}", predictions.mean().unwrap());
    info!("  Std prediction: {:.6}", predictions.std(0.0));
    
    // Cleanup
    ebpf_manager.shutdown().await?;
    info!("AINKA Linear Regression Example completed successfully!");
    
    Ok(())
}

/// Generate synthetic telemetry events for testing
fn generate_synthetic_events(count: usize) -> Vec<TelemetryEvent> {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let mut events = Vec::new();
    let start_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    
    for i in 0..count {
        let timestamp = start_time + (i as u64 * 1_000_000); // 1ms intervals
        
        // Generate different types of events
        let event_type = match i % 4 {
            0 => EventType::Syscall,
            1 => EventType::Memory,
            2 => EventType::Io,
            3 => EventType::Scheduler,
            _ => EventType::Syscall,
        };
        
        let data = match event_type {
            EventType::Syscall => json!({
                "pid": 1000 + (i % 100),
                "syscall_nr": 1 + (i % 10),
                "duration": 1000 + (i % 10000), // 1-10ms
                "retval": 0,
                "comm": format!("process_{}", i % 10),
                "system_load": 0.5 + (i as f64 * 0.01).sin() * 0.3, // Synthetic load
            }),
            EventType::Memory => json!({
                "pid": 1000 + (i % 100),
                "size": 4096 + (i % 1024) * 1024, // 4KB - 1MB
                "operation": 0,
                "comm": format!("process_{}", i % 10),
                "system_load": 0.3 + (i as f64 * 0.02).cos() * 0.4,
            }),
            EventType::Io => json!({
                "pid": 1000 + (i % 100),
                "bytes": 512 + (i % 4096),
                "latency": 100 + (i % 1000), // 100μs - 1ms
                "operation": i % 2,
                "comm": format!("process_{}", i % 10),
                "system_load": 0.4 + (i as f64 * 0.015).sin() * 0.35,
            }),
            EventType::Scheduler => json!({
                "prev_pid": 1000 + (i % 100),
                "next_pid": 1000 + ((i + 1) % 100),
                "runtime": 1000 + (i % 10000), // 1-10ms
                "cpu": i % 8,
                "prev_comm": format!("process_{}", i % 10),
                "next_comm": format!("process_{}", (i + 1) % 10),
                "system_load": 0.6 + (i as f64 * 0.025).cos() * 0.25,
            }),
            _ => json!({
                "value": i as f64,
                "system_load": 0.5,
            }),
        };
        
        events.push(TelemetryEvent {
            timestamp,
            event_type,
            source: "synthetic".to_string(),
            data,
        });
    }
    
    events
}

/// Example of using the linear regression system for specific predictions
async fn example_specific_predictions() -> Result<()> {
    info!("Example: Specific Predictions");
    info!("============================");
    
    // Create a specialized regression pipeline for CPU usage prediction
    let cpu_target = "cpu_usage".to_string();
    let mut cpu_pipeline = RegressionPipeline::new(30, cpu_target.clone());
    
    // Create a specialized regression pipeline for memory usage prediction
    let memory_target = "memory_usage".to_string();
    let mut memory_pipeline = RegressionPipeline::new(30, memory_target.clone());
    
    // Create a specialized regression pipeline for I/O latency prediction
    let io_target = "io_latency".to_string();
    let mut io_pipeline = RegressionPipeline::new(30, io_target.clone());
    
    // Generate synthetic data for each target
    let cpu_events = generate_cpu_events(1000);
    let memory_events = generate_memory_events(1000);
    let io_events = generate_io_events(1000);
    
    // Train models
    cpu_pipeline.process_events(cpu_events.clone()).await?;
    memory_pipeline.process_events(memory_events.clone()).await?;
    io_pipeline.process_events(io_events.clone()).await?;
    
    // Make predictions
    let cpu_predictions = cpu_pipeline.predict(&cpu_events[..10])?;
    let memory_predictions = memory_pipeline.predict(&memory_events[..10])?;
    let io_predictions = io_pipeline.predict(&io_events[..10])?;
    
    info!("CPU Usage Predictions: {:?}", cpu_predictions.to_vec());
    info!("Memory Usage Predictions: {:?}", memory_predictions.to_vec());
    info!("I/O Latency Predictions: {:?}", io_predictions.to_vec());
    
    // Compare model performance
    let cpu_metrics = cpu_pipeline.get_performance_metrics();
    let memory_metrics = memory_pipeline.get_performance_metrics();
    let io_metrics = io_pipeline.get_performance_metrics();
    
    info!("Model Performance Comparison:");
    info!("  CPU Model - MSE: {:.6}, R²: {:.6}", cpu_metrics.mse, cpu_metrics.r2);
    info!("  Memory Model - MSE: {:.6}, R²: {:.6}", memory_metrics.mse, memory_metrics.r2);
    info!("  I/O Model - MSE: {:.6}, R²: {:.6}", io_metrics.mse, io_metrics.r2);
    
    Ok(())
}

fn generate_cpu_events(count: usize) -> Vec<TelemetryEvent> {
    // Implementation for CPU-specific events
    generate_synthetic_events(count)
}

fn generate_memory_events(count: usize) -> Vec<TelemetryEvent> {
    // Implementation for memory-specific events
    generate_synthetic_events(count)
}

fn generate_io_events(count: usize) -> Vec<TelemetryEvent> {
    // Implementation for I/O-specific events
    generate_synthetic_events(count)
} 