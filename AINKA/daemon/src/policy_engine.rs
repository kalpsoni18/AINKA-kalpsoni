/*
 * AINKA Policy Engine
 * 
 * This module implements the policy engine for the AI-Native kernel
 * assistant, providing rule-based decision making and policy management.
 * 
 * Copyright (C) 2024 AINKA Community
 * Licensed under Apache 2.0
 */

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use log::{info, warn, error, debug};

/// Policy rule structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub conditions: Vec<Condition>,
    pub actions: Vec<Action>,
    pub priority: u32,
    pub enabled: bool,
    pub created_at: u64,
    pub updated_at: u64,
    pub execution_count: u32,
    pub last_executed: Option<u64>,
}

/// Condition for policy evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub metric: String,
    pub operator: ConditionOperator,
    pub value: f64,
    pub duration: Option<Duration>,
}

/// Condition operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Between,
    NotBetween,
}

/// Action to execute when policy is triggered
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub action_type: ActionType,
    pub target: String,
    pub parameters: HashMap<String, String>,
    pub timeout: Option<Duration>,
}

/// Action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    SetCPUFrequency,
    SetMemoryLimit,
    SetIoScheduler,
    SetNetworkQoS,
    SetProcessPriority,
    KillProcess,
    RestartService,
    SendAlert,
    ExecuteCommand,
    Custom,
}

/// Threshold management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Threshold {
    pub metric: String,
    pub warning: f64,
    pub critical: f64,
    pub emergency: f64,
    pub auto_adjust: bool,
    pub learning_rate: f64,
}

/// Policy evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvaluation {
    pub rule_id: String,
    pub triggered: bool,
    pub conditions_met: Vec<bool>,
    pub actions_executed: Vec<String>,
    pub timestamp: u64,
    pub execution_time: Duration,
}

/// Policy engine configuration
#[derive(Debug, Clone)]
pub struct PolicyEngineConfig {
    pub evaluation_interval: Duration,
    pub max_rules: usize,
    pub enable_learning: bool,
    pub auto_threshold_adjustment: bool,
    pub rule_timeout: Duration,
    pub max_concurrent_executions: usize,
}

impl Default for PolicyEngineConfig {
    fn default() -> Self {
        Self {
            evaluation_interval: Duration::from_secs(10),
            max_rules: 1000,
            enable_learning: true,
            auto_threshold_adjustment: true,
            rule_timeout: Duration::from_secs(30),
            max_concurrent_executions: 10,
        }
    }
}

/// Rule engine
pub struct RuleEngine {
    rules: HashMap<String, PolicyRule>,
    config: PolicyEngineConfig,
    evaluation_history: Vec<PolicyEvaluation>,
}

/// Threshold manager
pub struct ThresholdManager {
    thresholds: HashMap<String, Threshold>,
    config: PolicyEngineConfig,
    adjustment_history: Vec<ThresholdAdjustment>,
}

/// Action generator
pub struct ActionGenerator {
    action_templates: HashMap<String, ActionTemplate>,
    config: PolicyEngineConfig,
    execution_history: Vec<ActionExecution>,
}

/// Policy validator
pub struct PolicyValidator {
    validation_rules: Vec<ValidationRule>,
    config: PolicyEngineConfig,
}

/// Action template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionTemplate {
    pub name: String,
    pub action_type: ActionType,
    pub default_parameters: HashMap<String, String>,
    pub validation_schema: Option<String>,
}

/// Threshold adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdAdjustment {
    pub metric: String,
    pub old_threshold: f64,
    pub new_threshold: f64,
    pub reason: String,
    pub timestamp: u64,
}

/// Action execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionExecution {
    pub action_id: String,
    pub action_type: ActionType,
    pub target: String,
    pub parameters: HashMap<String, String>,
    pub status: ExecutionStatus,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub error_message: Option<String>,
}

/// Execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Timeout,
}

/// Validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub name: String,
    pub condition: Box<dyn Fn(&PolicyRule) -> bool + Send + Sync>,
    pub error_message: String,
}

/// Main Policy Engine
pub struct PolicyEngine {
    rule_engine: RuleEngine,
    threshold_manager: ThresholdManager,
    action_generator: ActionGenerator,
    policy_validator: PolicyValidator,
    config: PolicyEngineConfig,
    metrics_receiver: Option<mpsc::Receiver<SystemMetrics>>,
    running: Arc<Mutex<bool>>,
}

impl PolicyEngine {
    /// Create a new Policy Engine
    pub fn new(config: PolicyEngineConfig) -> Self {
        Self {
            rule_engine: RuleEngine::new(config.clone()),
            threshold_manager: ThresholdManager::new(config.clone()),
            action_generator: ActionGenerator::new(config.clone()),
            policy_validator: PolicyValidator::new(config.clone()),
            config,
            metrics_receiver: None,
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the Policy Engine
    pub async fn start(&mut self, metrics_receiver: mpsc::Receiver<SystemMetrics>) {
        info!("Starting AINKA Policy Engine");
        
        self.metrics_receiver = Some(metrics_receiver);
        *self.running.lock().unwrap() = true;
        
        // Start evaluation loop
        self.evaluation_loop().await;
    }

    /// Stop the Policy Engine
    pub fn stop(&self) {
        info!("Stopping AINKA Policy Engine");
        *self.running.lock().unwrap() = false;
    }

    /// Main evaluation loop
    async fn evaluation_loop(&mut self) {
        let mut interval = tokio::time::interval(self.config.evaluation_interval);
        
        while *self.running.lock().unwrap() {
            tokio::select! {
                _ = interval.tick() => {
                    self.run_evaluation_cycle().await;
                }
                
                metrics = self.receive_metrics() => {
                    if let Some(metrics) = metrics {
                        self.process_metrics(&metrics).await;
                    }
                }
            }
        }
    }

    /// Receive metrics from the channel
    async fn receive_metrics(&mut self) -> Option<SystemMetrics> {
        if let Some(receiver) = &mut self.metrics_receiver {
            receiver.recv().await
        } else {
            None
        }
    }

    /// Process incoming metrics
    async fn process_metrics(&mut self, metrics: &SystemMetrics) {
        debug!("Processing metrics for policy evaluation: {:?}", metrics);
        
        // Update thresholds if auto-adjustment is enabled
        if self.config.auto_threshold_adjustment {
            self.threshold_manager.adjust_thresholds(metrics);
        }
        
        // Evaluate rules
        let evaluations = self.rule_engine.evaluate_rules(metrics);
        
        // Execute triggered actions
        for evaluation in evaluations {
            if evaluation.triggered {
                self.execute_policy_actions(&evaluation).await;
            }
        }
    }

    /// Run evaluation cycle
    async fn run_evaluation_cycle(&mut self) {
        debug!("Running policy evaluation cycle");
        
        // Validate all policies
        let validation_results = self.policy_validator.validate_all_policies();
        
        // Generate new actions based on current state
        let new_actions = self.action_generator.generate_actions();
        
        // Apply any pending actions
        for action in new_actions {
            self.execute_action(&action).await;
        }
    }

    /// Execute policy actions
    async fn execute_policy_actions(&self, evaluation: &PolicyEvaluation) {
        info!("Executing policy actions for rule: {}", evaluation.rule_id);
        
        if let Some(rule) = self.rule_engine.get_rule(&evaluation.rule_id) {
            for action in &rule.actions {
                self.execute_action(action).await;
            }
        }
    }

    /// Execute a single action
    async fn execute_action(&self, action: &Action) {
        let execution = ActionExecution {
            action_id: format!("action_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()),
            action_type: action.action_type.clone(),
            target: action.target.clone(),
            parameters: action.parameters.clone(),
            status: ExecutionStatus::Running,
            start_time: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            end_time: None,
            error_message: None,
        };
        
        info!("Executing action: {:?}", action.action_type);
        
        match action.action_type {
            ActionType::SetCPUFrequency => {
                if let Some(freq) = action.parameters.get("frequency") {
                    if let Ok(freq_value) = freq.parse::<u32>() {
                        // Send to kernel module
                        info!("Setting CPU frequency to {} MHz", freq_value);
                    }
                }
            }
            ActionType::SetMemoryLimit => {
                if let Some(limit) = action.parameters.get("limit") {
                    if let Ok(limit_value) = limit.parse::<u64>() {
                        // Send to kernel module
                        info!("Setting memory limit to {} MB", limit_value);
                    }
                }
            }
            ActionType::SetIoScheduler => {
                if let Some(scheduler) = action.parameters.get("scheduler") {
                    // Send to kernel module
                    info!("Setting I/O scheduler to {}", scheduler);
                }
            }
            ActionType::SetNetworkQoS => {
                if let Some(qos) = action.parameters.get("qos") {
                    // Send to kernel module
                    info!("Setting network QoS to {}", qos);
                }
            }
            ActionType::SendAlert => {
                if let Some(message) = action.parameters.get("message") {
                    info!("Sending alert: {}", message);
                }
            }
            ActionType::ExecuteCommand => {
                if let Some(command) = action.parameters.get("command") {
                    info!("Executing command: {}", command);
                    // Execute system command
                }
            }
            _ => {
                warn!("Unsupported action type: {:?}", action.action_type);
            }
        }
    }

    /// Add a new policy rule
    pub fn add_rule(&mut self, rule: PolicyRule) -> Result<(), String> {
        // Validate the rule
        if !self.policy_validator.validate_rule(&rule) {
            return Err("Invalid policy rule".to_string());
        }
        
        // Check if we have space for more rules
        if self.rule_engine.rules.len() >= self.config.max_rules {
            return Err("Maximum number of rules reached".to_string());
        }
        
        self.rule_engine.add_rule(rule)
    }

    /// Remove a policy rule
    pub fn remove_rule(&mut self, rule_id: &str) -> Result<(), String> {
        self.rule_engine.remove_rule(rule_id)
    }

    /// Update a policy rule
    pub fn update_rule(&mut self, rule: PolicyRule) -> Result<(), String> {
        // Validate the rule
        if !self.policy_validator.validate_rule(&rule) {
            return Err("Invalid policy rule".to_string());
        }
        
        self.rule_engine.update_rule(rule)
    }

    /// Get all rules
    pub fn get_rules(&self) -> Vec<PolicyRule> {
        self.rule_engine.get_all_rules()
    }

    /// Get rule by ID
    pub fn get_rule(&self, rule_id: &str) -> Option<PolicyRule> {
        self.rule_engine.get_rule(rule_id).cloned()
    }

    /// Add threshold
    pub fn add_threshold(&mut self, threshold: Threshold) {
        self.threshold_manager.add_threshold(threshold);
    }

    /// Get thresholds
    pub fn get_thresholds(&self) -> Vec<Threshold> {
        self.threshold_manager.get_all_thresholds()
    }

    /// Get evaluation history
    pub fn get_evaluation_history(&self) -> Vec<PolicyEvaluation> {
        self.rule_engine.get_evaluation_history()
    }

    /// Get execution history
    pub fn get_execution_history(&self) -> Vec<ActionExecution> {
        self.action_generator.get_execution_history()
    }
}

impl RuleEngine {
    /// Create a new rule engine
    pub fn new(config: PolicyEngineConfig) -> Self {
        Self {
            rules: HashMap::new(),
            config,
            evaluation_history: Vec::new(),
        }
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: PolicyRule) -> Result<(), String> {
        let rule_id = rule.id.clone();
        self.rules.insert(rule_id, rule);
        Ok(())
    }

    /// Remove a rule
    pub fn remove_rule(&mut self, rule_id: &str) -> Result<(), String> {
        if self.rules.remove(rule_id).is_some() {
            Ok(())
        } else {
            Err("Rule not found".to_string())
        }
    }

    /// Update a rule
    pub fn update_rule(&mut self, rule: PolicyRule) -> Result<(), String> {
        let rule_id = rule.id.clone();
        if self.rules.contains_key(&rule_id) {
            self.rules.insert(rule_id, rule);
            Ok(())
        } else {
            Err("Rule not found".to_string())
        }
    }

    /// Get a rule by ID
    pub fn get_rule(&self, rule_id: &str) -> Option<&PolicyRule> {
        self.rules.get(rule_id)
    }

    /// Get all rules
    pub fn get_all_rules(&self) -> Vec<PolicyRule> {
        self.rules.values().cloned().collect()
    }

    /// Evaluate rules against metrics
    pub fn evaluate_rules(&mut self, metrics: &SystemMetrics) -> Vec<PolicyEvaluation> {
        let mut evaluations = Vec::new();
        
        for rule in self.rules.values() {
            if !rule.enabled {
                continue;
            }
            
            let start_time = Instant::now();
            let (triggered, conditions_met) = self.evaluate_rule(rule, metrics);
            let execution_time = start_time.elapsed();
            
            let evaluation = PolicyEvaluation {
                rule_id: rule.id.clone(),
                triggered,
                conditions_met,
                actions_executed: Vec::new(), // Will be filled after execution
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                execution_time,
            };
            
            evaluations.push(evaluation.clone());
            self.evaluation_history.push(evaluation);
            
            // Keep history size manageable
            if self.evaluation_history.len() > 1000 {
                self.evaluation_history.remove(0);
            }
        }
        
        evaluations
    }

    /// Evaluate a single rule
    fn evaluate_rule(&self, rule: &PolicyRule, metrics: &SystemMetrics) -> (bool, Vec<bool>) {
        let mut conditions_met = Vec::new();
        
        for condition in &rule.conditions {
            let met = self.evaluate_condition(condition, metrics);
            conditions_met.push(met);
        }
        
        // All conditions must be met for the rule to be triggered
        let triggered = conditions_met.iter().all(|&met| met);
        
        (triggered, conditions_met)
    }

    /// Evaluate a single condition
    fn evaluate_condition(&self, condition: &Condition, metrics: &SystemMetrics) -> bool {
        let metric_value = self.get_metric_value(&condition.metric, metrics);
        
        match condition.operator {
            ConditionOperator::GreaterThan => metric_value > condition.value,
            ConditionOperator::LessThan => metric_value < condition.value,
            ConditionOperator::Equal => (metric_value - condition.value).abs() < f64::EPSILON,
            ConditionOperator::NotEqual => (metric_value - condition.value).abs() >= f64::EPSILON,
            ConditionOperator::GreaterThanOrEqual => metric_value >= condition.value,
            ConditionOperator::LessThanOrEqual => metric_value <= condition.value,
            ConditionOperator::Between => {
                if let Some(duration) = condition.duration {
                    // For between, we need two values
                    let value2 = condition.value + duration.as_secs_f64();
                    metric_value >= condition.value && metric_value <= value2
                } else {
                    false
                }
            }
            ConditionOperator::NotBetween => {
                if let Some(duration) = condition.duration {
                    let value2 = condition.value + duration.as_secs_f64();
                    metric_value < condition.value || metric_value > value2
                } else {
                    true
                }
            }
        }
    }

    /// Get metric value from system metrics
    fn get_metric_value(&self, metric: &str, metrics: &SystemMetrics) -> f64 {
        match metric {
            "cpu_usage" => metrics.cpu_usage,
            "memory_usage" => metrics.memory_usage,
            "io_wait" => metrics.io_wait,
            "disk_usage" => metrics.disk_usage,
            "temperature" => metrics.temperature,
            "power_consumption" => metrics.power_consumption,
            "load_average_1m" => metrics.load_average[0],
            "load_average_5m" => metrics.load_average[1],
            "load_average_15m" => metrics.load_average[2],
            "network_rx" => metrics.network_rx as f64,
            "network_tx" => metrics.network_tx as f64,
            _ => 0.0,
        }
    }

    /// Get evaluation history
    pub fn get_evaluation_history(&self) -> Vec<PolicyEvaluation> {
        self.evaluation_history.clone()
    }
}

impl ThresholdManager {
    /// Create a new threshold manager
    pub fn new(config: PolicyEngineConfig) -> Self {
        Self {
            thresholds: HashMap::new(),
            config,
            adjustment_history: Vec::new(),
        }
    }

    /// Add a threshold
    pub fn add_threshold(&mut self, threshold: Threshold) {
        self.thresholds.insert(threshold.metric.clone(), threshold);
    }

    /// Get all thresholds
    pub fn get_all_thresholds(&self) -> Vec<Threshold> {
        self.thresholds.values().cloned().collect()
    }

    /// Adjust thresholds based on current metrics
    pub fn adjust_thresholds(&mut self, metrics: &SystemMetrics) {
        for threshold in self.thresholds.values_mut() {
            if threshold.auto_adjust {
                self.adjust_threshold(threshold, metrics);
            }
        }
    }

    /// Adjust a single threshold
    fn adjust_threshold(&mut self, threshold: &mut Threshold, metrics: &SystemMetrics) {
        let current_value = self.get_metric_value(&threshold.metric, metrics);
        
        // Simple adaptive threshold adjustment
        if current_value > threshold.critical {
            // Increase threshold if we're hitting critical levels too often
            let adjustment = threshold.critical * threshold.learning_rate;
            let new_threshold = threshold.critical + adjustment;
            
            let adjustment_record = ThresholdAdjustment {
                metric: threshold.metric.clone(),
                old_threshold: threshold.critical,
                new_threshold,
                reason: "Critical level exceeded".to_string(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            };
            
            threshold.critical = new_threshold;
            self.adjustment_history.push(adjustment_record);
        }
    }

    /// Get metric value
    fn get_metric_value(&self, metric: &str, metrics: &SystemMetrics) -> f64 {
        match metric {
            "cpu_usage" => metrics.cpu_usage,
            "memory_usage" => metrics.memory_usage,
            "io_wait" => metrics.io_wait,
            "disk_usage" => metrics.disk_usage,
            "temperature" => metrics.temperature,
            "power_consumption" => metrics.power_consumption,
            _ => 0.0,
        }
    }
}

impl ActionGenerator {
    /// Create a new action generator
    pub fn new(config: PolicyEngineConfig) -> Self {
        let mut action_templates = HashMap::new();
        
        // Add default action templates
        action_templates.insert("cpu_frequency".to_string(), ActionTemplate {
            name: "CPU Frequency Control".to_string(),
            action_type: ActionType::SetCPUFrequency,
            default_parameters: {
                let mut params = HashMap::new();
                params.insert("frequency".to_string(), "2000".to_string());
                params
            },
            validation_schema: None,
        });
        
        action_templates.insert("memory_limit".to_string(), ActionTemplate {
            name: "Memory Limit Control".to_string(),
            action_type: ActionType::SetMemoryLimit,
            default_parameters: {
                let mut params = HashMap::new();
                params.insert("limit".to_string(), "8192".to_string());
                params
            },
            validation_schema: None,
        });
        
        Self {
            action_templates,
            config,
            execution_history: Vec::new(),
        }
    }

    /// Generate actions based on current state
    pub fn generate_actions(&self) -> Vec<Action> {
        // This would analyze current system state and generate appropriate actions
        Vec::new()
    }

    /// Get execution history
    pub fn get_execution_history(&self) -> Vec<ActionExecution> {
        self.execution_history.clone()
    }
}

impl PolicyValidator {
    /// Create a new policy validator
    pub fn new(config: PolicyEngineConfig) -> Self {
        let mut validation_rules = Vec::new();
        
        // Add default validation rules
        validation_rules.push(ValidationRule {
            name: "Rule ID Required".to_string(),
            condition: Box::new(|rule| !rule.id.is_empty()),
            error_message: "Rule ID is required".to_string(),
        });
        
        validation_rules.push(ValidationRule {
            name: "Rule Name Required".to_string(),
            condition: Box::new(|rule| !rule.name.is_empty()),
            error_message: "Rule name is required".to_string(),
        });
        
        validation_rules.push(ValidationRule {
            name: "At Least One Condition".to_string(),
            condition: Box::new(|rule| !rule.conditions.is_empty()),
            error_message: "At least one condition is required".to_string(),
        });
        
        validation_rules.push(ValidationRule {
            name: "At Least One Action".to_string(),
            condition: Box::new(|rule| !rule.actions.is_empty()),
            error_message: "At least one action is required".to_string(),
        });
        
        Self {
            validation_rules,
            config,
        }
    }

    /// Validate a single rule
    pub fn validate_rule(&self, rule: &PolicyRule) -> bool {
        for validation_rule in &self.validation_rules {
            if !(validation_rule.condition)(rule) {
                return false;
            }
        }
        true
    }

    /// Validate all policies
    pub fn validate_all_policies(&self) -> Vec<String> {
        // This would validate all policies and return any errors
        Vec::new()
    }
}

// Import SystemMetrics from ai_engine module
use crate::ai_engine::SystemMetrics;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_engine_creation() {
        let config = PolicyEngineConfig::default();
        let engine = PolicyEngine::new(config);
        assert_eq!(engine.get_rules().len(), 0);
        assert_eq!(engine.get_thresholds().len(), 0);
    }

    #[test]
    fn test_rule_creation() {
        let config = PolicyEngineConfig::default();
        let mut engine = PolicyEngine::new(config);
        
        let rule = PolicyRule {
            id: "test_rule".to_string(),
            name: "Test Rule".to_string(),
            description: "A test rule".to_string(),
            conditions: vec![
                Condition {
                    metric: "cpu_usage".to_string(),
                    operator: ConditionOperator::GreaterThan,
                    value: 0.8,
                    duration: None,
                }
            ],
            actions: vec![
                Action {
                    action_type: ActionType::SendAlert,
                    target: "admin".to_string(),
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert("message".to_string(), "High CPU usage".to_string());
                        params
                    },
                    timeout: None,
                }
            ],
            priority: 1,
            enabled: true,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            updated_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            execution_count: 0,
            last_executed: None,
        };
        
        assert!(engine.add_rule(rule).is_ok());
        assert_eq!(engine.get_rules().len(), 1);
    }

    #[test]
    fn test_condition_evaluation() {
        let config = PolicyEngineConfig::default();
        let rule_engine = RuleEngine::new(config);
        
        let condition = Condition {
            metric: "cpu_usage".to_string(),
            operator: ConditionOperator::GreaterThan,
            value: 0.8,
            duration: None,
        };
        
        let metrics = SystemMetrics {
            timestamp: 1000,
            cpu_usage: 0.9,
            memory_usage: 0.6,
            io_wait: 0.1,
            network_rx: 1000,
            network_tx: 500,
            load_average: [1.0, 1.1, 1.2],
            disk_usage: 0.7,
            temperature: 45.0,
            power_consumption: 50.0,
        };
        
        assert!(rule_engine.evaluate_condition(&condition, &metrics));
    }
} 