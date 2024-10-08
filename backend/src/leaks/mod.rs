use axum::{
    extract::{Query, State},
    response::IntoResponse,
    routing::get,
    Json,
};
use chrono::{DateTime, Duration, Utc};
use futures::{FutureExt, StreamExt, TryStreamExt};
use mongodb::{
    bson::{doc, Document},
    Client as MongoClient, Collection,
};
use tracing::debug;

mod config;
mod errors;
mod models;

use config::LeakServiceConfig;
use errors::LeakServiceError;
use models::Leak;

#[derive(Clone)]
struct ServiceState {
    mongo_client: MongoClient,
}

#[derive(serde::Deserialize)]
struct LeakQuery {
    days: Option<i64>,
}

async fn get_leaks_from_db(client: &MongoClient, days: i64) -> Result<Vec<Leak>, LeakServiceError> {
    let leaks: Collection<Document> = client.database("riverdata").collection("spills");

    let query_end: DateTime<Utc> = Utc::now() - Duration::days(days);

    let mut results = leaks
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
                    "$gte": 15
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
            doc! {
                "$match": doc! {
                    "events.event_end": {
                    "$gte": query_end.to_rfc3339()
                    }
                }
            },
        ])
        .await?;

    let results: Vec<Leak> = results
        .map(|result| Ok::<Leak, LeakServiceError>(mongodb::bson::from_document::<Leak>(result?)?))
        .try_collect()
        .await?;

    Ok(results)
}

async fn get_leaks(
    State(service_state): State<ServiceState>,
    query: Query<LeakQuery>,
) -> Result<impl IntoResponse, LeakServiceError> {
    Ok(Json(
        get_leaks_from_db(&service_state.mongo_client, query.days.unwrap_or(7)).await?,
    ))
}

pub async fn create_leak_routes(config: LeakServiceConfig) -> anyhow::Result<axum::Router<()>> {
    let mongo_client = MongoClient::with_uri_str(&config.mongodb_uri()).await?;
    let state = ServiceState { mongo_client };

    Ok(axum::Router::new()
        .route("/", get(get_leaks))
        .with_state(state))
}
