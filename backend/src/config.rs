use crate::{level::LevelServiceConfig, spills::SpillServiceConfig};
use std::str::FromStr;
use tracing::info;

#[derive(Debug, PartialEq, Eq)]
pub enum Env {
    Dev,
    Prod,
}

impl FromStr for Env {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dev" => Ok(Self::Dev),
            "prod" => Ok(Self::Prod),
            _ => Err(format!("Unknown environment: {}", s)),
        }
    }
}

fn get_env() -> Env {
    let env: Env = std::env::var("APP_ENV")
        .unwrap_or_else(|_| "dev".to_string())
        .parse()
        .expect("Failed to parse APP_ENV");

    info!("App env: {:?}", env);
    env
}

#[derive(serde::Deserialize, Debug)]
pub struct HttpConfig {
    pub host: String,
    pub port: u16,
    pub cache_duration_sec: u32,
}

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    pub level_service: LevelServiceConfig,

    pub spill_service: SpillServiceConfig,

    pub http: HttpConfig,
}

pub fn load_config() -> anyhow::Result<Config> {
    Ok(config::Config::builder()
        .add_source(config::File::with_name("./config/default.json"))
        .add_source(
            config::File::with_name(match get_env() {
                Env::Dev => "./config/dev.json",
                Env::Prod => "./config/prod.json",
            })
            .required(false),
        )
        .add_source(config::Environment::with_prefix("APP"))
        .build()?
        .try_deserialize()?)
}
