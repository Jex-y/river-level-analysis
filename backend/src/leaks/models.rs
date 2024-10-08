use chrono::{DateTime, Utc};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LeakMetadata {
    site_id: String,
    site_name: String,
    site_location: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LeakEvent {
    event_start: DateTime<Utc>,
    event_end: DateTime<Utc>,
    event_duration_mins: i64,
    metadata: LeakMetadata,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Leak {
    metadata: LeakMetadata,
    events: Vec<LeakEvent>,
}
