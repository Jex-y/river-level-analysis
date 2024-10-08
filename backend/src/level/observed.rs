use super::{
    errors::LevelApiError,
    fetch_data::get_station_readings,
    models::{ObservationRecord, ServiceState},
    ColSpec,
};
use axum::{
    extract::{Query, State},
    response::Json,
};
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
            parameter: state.config.target_parameter.clone(),
        },
        query.num_readings.unwrap_or(4 * 24),
    )
    .await?;

    Ok(Json(
        data.features
            .iter()
            .map(ObservationRecord::from)
            .collect::<Vec<ObservationRecord>>(),
    ))
}
