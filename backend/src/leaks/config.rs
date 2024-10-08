#[derive(serde::Deserialize)]
pub struct LeakServiceConfig {
    mongodb_uri: String,
}

impl LeakServiceConfig {
    pub fn mongodb_uri(&self) -> &str {
        &self.mongodb_uri
    }
}
