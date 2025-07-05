use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use rusqlite::{Connection, Result as SqliteResult, params, Row, Statement};
use tokio::sync::mpsc;
use crate::telemetry_hub::TelemetryData;
use crate::anomaly_detector::AnomalyResult;
use crate::security_monitor::SecurityEvent;
use crate::predictive_scaler::ScalingPrediction;

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub path: String,
    pub max_connections: usize,
    pub auto_vacuum: bool,
    pub journal_mode: String,
    pub synchronous: String,
    pub cache_size: i32,
    pub temp_store: String,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: "/var/lib/ainka/ainka.db".to_string(),
            max_connections: 10,
            auto_vacuum: true,
            journal_mode: "WAL".to_string(), // Write-Ahead Logging for better performance
            synchronous: "NORMAL".to_string(),
            cache_size: -64000, // 64MB cache
            temp_store: "MEMORY".to_string(),
        }
    }
}

/// Database record types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryRecord {
    pub id: Option<i64>,
    pub timestamp: i64,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_io_read: f64,
    pub disk_io_write: f64,
    pub network_rx: f64,
    pub network_tx: f64,
    pub load_average: f64,
    pub temperature: Option<f64>,
    pub power_consumption: Option<f64>,
    pub metadata: String, // JSON string
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyRecord {
    pub id: Option<i64>,
    pub timestamp: i64,
    pub anomaly_type: String,
    pub severity: String,
    pub confidence: f64,
    pub description: String,
    pub metrics: String, // JSON string
    pub recommendations: String, // JSON string
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRecord {
    pub id: Option<i64>,
    pub timestamp: i64,
    pub threat_type: String,
    pub severity: String,
    pub source_ip: Option<String>,
    pub source_pid: Option<i32>,
    pub target_pid: Option<i32>,
    pub description: String,
    pub evidence: String, // JSON string
    pub confidence: f64,
    pub action_taken: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    pub id: Option<i64>,
    pub timestamp: i64,
    pub target: String,
    pub predicted_value: f64,
    pub current_value: f64,
    pub confidence: f64,
    pub time_horizon: i64,
    pub factors: String, // JSON string
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingActionRecord {
    pub id: Option<i64>,
    pub timestamp: i64,
    pub target: String,
    pub action_type: String,
    pub value: f64,
    pub confidence: f64,
    pub description: String,
    pub executed: bool,
    pub success: Option<bool>,
}

/// Database manager
pub struct DatabaseManager {
    config: DatabaseConfig,
    connection_pool: Arc<Mutex<VecDeque<Connection>>>,
    max_connections: usize,
}

impl DatabaseManager {
    /// Create a new database manager
    pub fn new(config: DatabaseConfig) -> SqliteResult<Self> {
        // Ensure database directory exists
        if let Some(parent) = std::path::Path::new(&config.path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Initialize database
        let conn = Connection::open(&config.path)?;
        Self::initialize_database(&conn)?;
        Self::configure_database(&conn, &config)?;

        let mut pool = VecDeque::new();
        pool.push_back(conn);

        Ok(Self {
            config,
            connection_pool: Arc::new(Mutex::new(pool)),
            max_connections: config.max_connections,
        })
    }

    /// Initialize database schema
    fn initialize_database(conn: &Connection) -> SqliteResult<()> {
        // Create telemetry table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                cpu_usage REAL NOT NULL,
                memory_usage REAL NOT NULL,
                disk_io_read REAL NOT NULL,
                disk_io_write REAL NOT NULL,
                network_rx REAL NOT NULL,
                network_tx REAL NOT NULL,
                load_average REAL NOT NULL,
                temperature REAL,
                power_consumption REAL,
                metadata TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Create index on timestamp for efficient queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry(timestamp)",
            [],
        )?;

        // Create anomalies table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                anomaly_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                description TEXT NOT NULL,
                metrics TEXT NOT NULL,
                recommendations TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(timestamp)",
            [],
        )?;

        // Create security events table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                threat_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                source_ip TEXT,
                source_pid INTEGER,
                target_pid INTEGER,
                description TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence REAL NOT NULL,
                action_taken TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_security_timestamp ON security_events(timestamp)",
            [],
        )?;

        // Create predictions table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                target TEXT NOT NULL,
                predicted_value REAL NOT NULL,
                current_value REAL NOT NULL,
                confidence REAL NOT NULL,
                time_horizon INTEGER NOT NULL,
                factors TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)",
            [],
        )?;

        // Create scaling actions table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scaling_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                target TEXT NOT NULL,
                action_type TEXT NOT NULL,
                value REAL NOT NULL,
                confidence REAL NOT NULL,
                description TEXT NOT NULL,
                executed BOOLEAN NOT NULL DEFAULT 0,
                success BOOLEAN,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scaling_timestamp ON scaling_actions(timestamp)",
            [],
        )?;

        // Create system stats table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                stat_name TEXT NOT NULL,
                stat_value REAL NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON system_stats(timestamp)",
            [],
        )?;

        Ok(())
    }

    /// Configure database for performance
    fn configure_database(conn: &Connection, config: &DatabaseConfig) -> SqliteResult<()> {
        conn.execute_batch(&format!(
            "PRAGMA journal_mode = {}; \
             PRAGMA synchronous = {}; \
             PRAGMA cache_size = {}; \
             PRAGMA temp_store = {}; \
             PRAGMA auto_vacuum = {};",
            config.journal_mode,
            config.synchronous,
            config.cache_size,
            config.temp_store,
            if config.auto_vacuum { "INCREMENTAL" } else { "NONE" }
        ))?;

        Ok(())
    }

    /// Get a connection from the pool
    fn get_connection(&self) -> SqliteResult<Connection> {
        let mut pool = self.connection_pool.lock().unwrap();
        
        if let Some(conn) = pool.pop_front() {
            Ok(conn)
        } else {
            // Create new connection if pool is empty
            let conn = Connection::open(&self.config.path)?;
            Self::configure_database(&conn, &self.config)?;
            Ok(conn)
        }
    }

    /// Return a connection to the pool
    fn return_connection(&self, conn: Connection) {
        let mut pool = self.connection_pool.lock().unwrap();
        
        if pool.len() < self.max_connections {
            pool.push_back(conn);
        }
        // If pool is full, connection will be dropped
    }

    /// Store telemetry data
    pub fn store_telemetry(&self, data: &TelemetryData) -> SqliteResult<()> {
        let conn = self.get_connection()?;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let metadata = serde_json::to_string(&data.metadata).unwrap_or_default();

        conn.execute(
            "INSERT INTO telemetry (
                timestamp, cpu_usage, memory_usage, disk_io_read, disk_io_write,
                network_rx, network_tx, load_average, temperature, power_consumption, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                timestamp,
                data.cpu_usage,
                data.memory_usage,
                data.disk_io_read,
                data.disk_io_write,
                data.network_rx,
                data.network_tx,
                data.load_average,
                data.temperature,
                data.power_consumption,
                metadata,
            ],
        )?;

        self.return_connection(conn);
        Ok(())
    }

    /// Store anomaly result
    pub fn store_anomaly(&self, anomaly: &AnomalyResult) -> SqliteResult<()> {
        let conn = self.get_connection()?;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let metrics = serde_json::to_string(&anomaly.metrics).unwrap_or_default();
        let recommendations = serde_json::to_string(&anomaly.recommendations).unwrap_or_default();

        conn.execute(
            "INSERT INTO anomalies (
                timestamp, anomaly_type, severity, confidence, description, metrics, recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                timestamp,
                format!("{:?}", anomaly.anomaly_type),
                format!("{:?}", anomaly.severity),
                anomaly.confidence,
                anomaly.description,
                metrics,
                recommendations,
            ],
        )?;

        self.return_connection(conn);
        Ok(())
    }

    /// Store security event
    pub fn store_security_event(&self, event: &SecurityEvent) -> SqliteResult<()> {
        let conn = self.get_connection()?;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let evidence = serde_json::to_string(&event.evidence).unwrap_or_default();

        conn.execute(
            "INSERT INTO security_events (
                timestamp, threat_type, severity, source_ip, source_pid, target_pid,
                description, evidence, confidence, action_taken
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                timestamp,
                format!("{:?}", event.threat_type),
                format!("{:?}", event.severity),
                event.source_ip,
                event.source_pid,
                event.target_pid,
                event.description,
                evidence,
                event.confidence,
                event.action_taken,
            ],
        )?;

        self.return_connection(conn);
        Ok(())
    }

    /// Store scaling prediction
    pub fn store_prediction(&self, prediction: &ScalingPrediction) -> SqliteResult<()> {
        let conn = self.get_connection()?;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let factors = serde_json::to_string(&prediction.factors).unwrap_or_default();

        conn.execute(
            "INSERT INTO predictions (
                timestamp, target, predicted_value, current_value, confidence, time_horizon, factors
            ) VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                timestamp,
                format!("{:?}", prediction.target),
                prediction.predicted_value,
                prediction.current_value,
                prediction.confidence,
                prediction.time_horizon.as_secs() as i64,
                factors,
            ],
        )?;

        self.return_connection(conn);
        Ok(())
    }

    /// Store scaling action
    pub fn store_scaling_action(&self, action: &crate::predictive_scaler::ScalingAction) -> SqliteResult<()> {
        let conn = self.get_connection()?;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO scaling_actions (
                timestamp, target, action_type, value, confidence, description, executed
            ) VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                timestamp,
                format!("{:?}", action.target),
                action.action_type,
                action.value,
                action.confidence,
                action.description,
                false, // Not executed yet
            ],
        )?;

        self.return_connection(conn);
        Ok(())
    }

    /// Store system statistic
    pub fn store_system_stat(&self, name: &str, value: f64, metadata: Option<HashMap<String, String>>) -> SqliteResult<()> {
        let conn = self.get_connection()?;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let metadata_json = metadata
            .map(|m| serde_json::to_string(&m).unwrap_or_default())
            .unwrap_or_default();

        conn.execute(
            "INSERT INTO system_stats (timestamp, stat_name, stat_value, metadata) VALUES (?, ?, ?, ?)",
            params![timestamp, name, value, metadata_json],
        )?;

        self.return_connection(conn);
        Ok(())
    }

    /// Query telemetry data
    pub fn query_telemetry(&self, start_time: i64, end_time: i64, limit: Option<i64>) -> SqliteResult<Vec<TelemetryRecord>> {
        let conn = self.get_connection()?;
        
        let limit_clause = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();
        
        let mut stmt = conn.prepare(&format!(
            "SELECT id, timestamp, cpu_usage, memory_usage, disk_io_read, disk_io_write,
                    network_rx, network_tx, load_average, temperature, power_consumption, metadata
             FROM telemetry 
             WHERE timestamp BETWEEN ? AND ?
             ORDER BY timestamp DESC{}",
            limit_clause
        ))?;

        let rows = stmt.query_map(params![start_time, end_time], |row| {
            Ok(TelemetryRecord {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                cpu_usage: row.get(2)?,
                memory_usage: row.get(3)?,
                disk_io_read: row.get(4)?,
                disk_io_write: row.get(5)?,
                network_rx: row.get(6)?,
                network_tx: row.get(7)?,
                load_average: row.get(8)?,
                temperature: row.get(9)?,
                power_consumption: row.get(10)?,
                metadata: row.get(11)?,
            })
        })?;

        let mut records = Vec::new();
        for row in rows {
            records.push(row?);
        }

        self.return_connection(conn);
        Ok(records)
    }

    /// Query anomalies
    pub fn query_anomalies(&self, start_time: i64, end_time: i64, severity: Option<&str>) -> SqliteResult<Vec<AnomalyRecord>> {
        let conn = self.get_connection()?;
        
        let severity_clause = severity.map(|s| format!(" AND severity = '{}'", s)).unwrap_or_default();
        
        let mut stmt = conn.prepare(&format!(
            "SELECT id, timestamp, anomaly_type, severity, confidence, description, metrics, recommendations
             FROM anomalies 
             WHERE timestamp BETWEEN ? AND ?{}
             ORDER BY timestamp DESC",
            severity_clause
        ))?;

        let rows = stmt.query_map(params![start_time, end_time], |row| {
            Ok(AnomalyRecord {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                anomaly_type: row.get(2)?,
                severity: row.get(3)?,
                confidence: row.get(4)?,
                description: row.get(5)?,
                metrics: row.get(6)?,
                recommendations: row.get(7)?,
            })
        })?;

        let mut records = Vec::new();
        for row in rows {
            records.push(row?);
        }

        self.return_connection(conn);
        Ok(records)
    }

    /// Query security events
    pub fn query_security_events(&self, start_time: i64, end_time: i64, threat_type: Option<&str>) -> SqliteResult<Vec<SecurityRecord>> {
        let conn = self.get_connection()?;
        
        let threat_clause = threat_type.map(|t| format!(" AND threat_type = '{}'", t)).unwrap_or_default();
        
        let mut stmt = conn.prepare(&format!(
            "SELECT id, timestamp, threat_type, severity, source_ip, source_pid, target_pid,
                    description, evidence, confidence, action_taken
             FROM security_events 
             WHERE timestamp BETWEEN ? AND ?{}
             ORDER BY timestamp DESC",
            threat_clause
        ))?;

        let rows = stmt.query_map(params![start_time, end_time], |row| {
            Ok(SecurityRecord {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                threat_type: row.get(2)?,
                severity: row.get(3)?,
                source_ip: row.get(4)?,
                source_pid: row.get(5)?,
                target_pid: row.get(6)?,
                description: row.get(7)?,
                evidence: row.get(8)?,
                confidence: row.get(9)?,
                action_taken: row.get(10)?,
            })
        })?;

        let mut records = Vec::new();
        for row in rows {
            records.push(row?);
        }

        self.return_connection(conn);
        Ok(records)
    }

    /// Get database statistics
    pub fn get_database_stats(&self) -> SqliteResult<HashMap<String, i64>> {
        let conn = self.get_connection()?;
        let mut stats = HashMap::new();

        // Count records in each table
        let tables = ["telemetry", "anomalies", "security_events", "predictions", "scaling_actions", "system_stats"];
        
        for table in &tables {
            let count: i64 = conn.query_row(&format!("SELECT COUNT(*) FROM {}", table), [], |row| row.get(0))?;
            stats.insert(format!("{}_count", table), count);
        }

        // Get database size
        let size: i64 = conn.query_row("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()", [], |row| row.get(0))?;
        stats.insert("database_size_bytes".to_string(), size);

        self.return_connection(conn);
        Ok(stats)
    }

    /// Clean up old data
    pub fn cleanup_old_data(&self, days_to_keep: i64) -> SqliteResult<()> {
        let conn = self.get_connection()?;
        
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64 - (days_to_keep * 24 * 3600);

        let tables = ["telemetry", "anomalies", "security_events", "predictions", "scaling_actions", "system_stats"];
        
        for table in &tables {
            conn.execute(&format!("DELETE FROM {} WHERE timestamp < ?", table), params![cutoff_time])?;
        }

        // Optimize database
        conn.execute("VACUUM", [])?;

        self.return_connection(conn);
        Ok(())
    }

    /// Export data to JSON
    pub fn export_data(&self, start_time: i64, end_time: i64) -> SqliteResult<HashMap<String, serde_json::Value>> {
        let mut export = HashMap::new();

        // Export telemetry
        let telemetry = self.query_telemetry(start_time, end_time, None)?;
        export.insert("telemetry".to_string(), serde_json::to_value(telemetry)?);

        // Export anomalies
        let anomalies = self.query_anomalies(start_time, end_time, None)?;
        export.insert("anomalies".to_string(), serde_json::to_value(anomalies)?);

        // Export security events
        let security_events = self.query_security_events(start_time, end_time, None)?;
        export.insert("security_events".to_string(), serde_json::to_value(security_events)?);

        Ok(export)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_database_creation() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        
        let config = DatabaseConfig {
            path: db_path.to_str().unwrap().to_string(),
            ..Default::default()
        };
        
        let db = DatabaseManager::new(config);
        assert!(db.is_ok());
    }

    #[test]
    fn test_telemetry_storage() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        
        let config = DatabaseConfig {
            path: db_path.to_str().unwrap().to_string(),
            ..Default::default()
        };
        
        let db = DatabaseManager::new(config).unwrap();
        
        let telemetry_data = crate::telemetry_hub::TelemetryData {
            cpu_usage: 50.0,
            memory_usage: 60.0,
            disk_io_read: 100.0,
            disk_io_write: 50.0,
            network_rx: 200.0,
            network_tx: 100.0,
            load_average: 1.5,
            temperature: Some(45.0),
            power_consumption: Some(80.0),
            metadata: HashMap::new(),
        };
        
        let result = db.store_telemetry(&telemetry_data);
        assert!(result.is_ok());
    }
} 