use crate::{level::LevelServiceConfig, spills::SpillServiceConfig};
use std::str::FromStr;

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
    std::env::var("APP_ENV")
        .unwrap_or_else(|_| "dev".to_string())
        .parse()
        .expect("Failed to parse APP_ENV")
}

#[derive(serde::Deserialize)]
pub struct HttpConfig {
    pub host: String,
    pub port: u16,
    pub cache_duration_sec: u32,
}

#[derive(serde::Deserialize)]
pub struct Config {
    pub level_service: LevelServiceConfig,

    pub spill_service: SpillServiceConfig,

    pub http: HttpConfig,
}

pub fn load_config() -> Config {
    let env = get_env();

    let mut config_builder = config::Config::builder()
        .add_source(config::File::with_name("./config/default.json"))
        .add_source(config::Environment::with_prefix("APP"));

    if env == Env::Dev {
        config_builder = config_builder.add_source(config::File::with_name("./config/dev.json"));
    }

    let config: Config = config_builder
        .build()
        .expect("Failed to build config")
        .try_deserialize()
        .expect("Failed to deserialize config");

    config
}
