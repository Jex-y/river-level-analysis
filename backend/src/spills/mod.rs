use axum::{
    extract::{Query, State},
    response::IntoResponse,
    routing::get,
    Json,
};
use chrono::{DateTime, Duration, Utc};
use futures::{StreamExt, TryStreamExt};
use mongodb::{
    bson::{doc, Document},
    Client as MongoClient, Collection,
};

mod config;
mod errors;
mod models;

pub use config::SpillServiceConfig;
use errors::SpillServiceError;
use models::StormOutflow;
use tracing::info;

#[derive(Clone)]
struct ServiceState {
    mongo_client: MongoClient,
}

#[derive(serde::Deserialize)]
struct SpillsQuery {
    since: Option<DateTime<Utc>>,
}

async fn get_spills_from_db(
    client: &MongoClient,
    since: DateTime<Utc>,
) -> Result<Vec<StormOutflow>, SpillServiceError> {
    let spills: Collection<Document> = client.database("riverdata").collection("spills");

    let results = spills
        .aggregate(vec![
            doc! {
                "$addFields": doc! {
                    "event_duration_mins" :doc! {
                        "$dateDiff" : doc! {
                            "startDate": "$event_start",
                            "endDate": "$event_end",
                            "unit": "minute",
            }}
                } },
            doc! {
            "$unset": vec!["_id"]
            },
            doc! {
                "$match" : doc! {
                    "event_duration_mins": doc! {
                    "$gte": 5
                    }
                }

            },
            doc! {
                "$sort": doc! { "event_end" : -1 }
            },
            doc! {
                "$group": doc! {
                    "_id": "$metadata.site_id",
                    "metadata": doc! { "$first": "$metadata" },
                    "events" : doc! {
                        "$push": doc! {
                            "$unsetField" : doc! {
                                "field": "metadata",
                                "input": "$$ROOT"
                            }
                    }
                    }
                }
            },
            // doc! {
            //     "$match": doc! {
            //         "events.event_end": {
            //         "$gte": since.to_rfc3339()
            //         }
            //     }
            // },
        ])
        .await?;

    let results: Vec<StormOutflow> = results
        .map(|result| {
            Ok::<StormOutflow, SpillServiceError>(mongodb::bson::from_document::<StormOutflow>(
                result?,
            )?)
        })
        .try_collect()
        .await?;

    Ok(results)
}

async fn get_spills(
    State(service_state): State<ServiceState>,
    query: Query<SpillsQuery>,
) -> Result<impl IntoResponse, SpillServiceError> {
    Ok(Json(
        get_spills_from_db(
            &service_state.mongo_client,
            query
                .since
                .unwrap_or_else(|| Utc::now() - Duration::days(7)),
        )
        .await?,
    ))
}

pub async fn create_spill_routes(config: SpillServiceConfig) -> anyhow::Result<axum::Router<()>> {
    let mongo_client = MongoClient::with_uri_str(&config.mongodb_uri()).await?;
    let state = ServiceState { mongo_client };

    Ok(axum::Router::new()
        .route("/", get(get_spills))
        .with_state(state))
}
