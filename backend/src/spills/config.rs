#[derive(serde::Deserialize)]
pub struct SpillServiceConfig {
    mongodb_uri: String,
}

impl SpillServiceConfig {
    pub fn mongodb_uri(&self) -> &str {
        &self.mongodb_uri
    }
}
