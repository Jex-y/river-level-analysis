output "backend_url" {
  value = google_cloud_run_v2_service.backend.uri
}
