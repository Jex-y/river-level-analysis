resource "google_project_service" "run_api" {
  service            = "run.googleapis.com"
  disable_on_destroy = true
}

resource "google_cloud_run_v2_service" "backend" {
  name                = "backend"
  location            = "europe-west2"
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"



  template {
    containers {
      image = "europe-west2-docker.pkg.dev/durham-river-level/containers/backend:1.0"

      ports {
        container_port = 8080
      }

      resources {
        cpu_idle = true

        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }

    scaling {
      max_instance_count = 2

    }

  }
  depends_on = [google_project_service.run_api]
}

# Allow unauthenticated users to invoke the service
resource "google_cloud_run_service_iam_member" "run_all_users" {
  service  = google_cloud_run_v2_service.backend.name
  location = google_cloud_run_v2_service.backend.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
