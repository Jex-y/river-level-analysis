use chrono::{DateTime, Utc};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum EventType {
    #[serde(rename = "spill")]
    Spill,

    #[serde(rename = "monitor offline")]
    MonitorOffline,

    #[serde(rename = "no recent spill")]
    NoRecentSpill,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Metadata {
    site_id: String,
    site_name: String,
    nearby: bool,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct OutflowEvent {
    #[serde(
        deserialize_with = "bson::serde_helpers::deserialize_chrono_datetime_from_bson_datetime"
    )]
    event_start: DateTime<Utc>,

    #[serde(
        deserialize_with = "bson::serde_helpers::deserialize_chrono_datetime_from_bson_datetime"
    )]
    event_end: DateTime<Utc>,

    // #[serde(
    //     deserialize_with = "bson::serde_helpers::deserialize_chrono_datetime_from_bson_datetime"
    // )]
    // last_updated: DateTime<Utc>,
    event_duration_mins: u64,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct StormOutflow {
    metadata: Metadata,
    events: Vec<OutflowEvent>,
}
