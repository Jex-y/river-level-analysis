provider "google" {
  project = "durham-river-level"
  region  = "europe-west2"
}

module "backend" {
  source = "./modules/backend"
}
