use super::{
    errors::LevelApiError,
    fetch_data::get_station_readings,
    models::{ObservationRecord, ServiceState, StationQuery},
    ColSpec,
};
use axum::{
    extract::{Query, State},
    response::Json,
};
use chrono::DateTime;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct ObservedQuery {
    num_readings: Option<usize>,
}

pub async fn get_observed(
    State(state): State<ServiceState>,
    query: Query<ObservedQuery>,
) -> Result<Json<Vec<ObservationRecord>>, LevelApiError> {
    let data = get_station_readings(
        &state.http_client,
        ColSpec {
            station_id: state.config.target_station_id.clone(),
            parameter: state.config.target_parameter,
        },
        StationQuery::last_n(query.num_readings.unwrap_or(4 * 24)),
    )
    .await?;

    let records: Vec<ObservationRecord> = data
        .column("datetime")?
        .datetime()?
        .into_iter()
        .zip(data.column("value")?.f32()?.into_iter())
        .map(|(datetime, value)| match (datetime, value) {
            (Some(datetime), Some(value)) => Ok(ObservationRecord::new(
                DateTime::from_timestamp_millis(datetime).expect("Invalid timestamp"),
                value,
            )),
            _ => Err(LevelApiError::MissingData(format!(
                "Observation record {:?} is missing either datetime or value",
                (datetime, value)
            ))),
        })
        .collect::<Result<Vec<ObservationRecord>, LevelApiError>>()?;

    Ok(Json(records))
}
