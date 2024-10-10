#[derive(serde::Deserialize, Debug)]
pub struct SpillServiceConfig {
    mongodb_uri: String,
}

impl SpillServiceConfig {
    pub fn mongodb_uri(&self) -> &str {
        &self.mongodb_uri
    }
}
