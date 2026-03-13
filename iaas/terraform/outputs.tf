output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.primary.location
}

output "gcs_bucket_name" {
  description = "GCS bucket name for data storage"
  value       = google_storage_bucket.wf_data_bucket.name
}

output "gcs_bucket_url" {
  description = "GCS bucket URL"
  value       = google_storage_bucket.wf_data_bucket.url
}

output "node_service_account_email" {
  description = "Email of the node service account"
  value       = google_service_account.gke_node_sa.email
}

output "argo_service_account_email" {
  description = "Email of the Argo Workflows service account"
  value       = google_service_account.argo_sa.email
}

output "vpc_network_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.vpc_network.name
}

output "subnet_name" {
  description = "Name of the subnet"
  value       = google_compute_subnetwork.subnet.name
}

output "kubectl_connection_command" {
  description = "Command to connect kubectl to the cluster"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --zone=${google_container_cluster.primary.location} --project=${var.project_id}"
}