# Configure the Google Cloud Provider
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.84"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.9"
    }
  }
  required_version = ">= 1.0"
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Enable required APIs
resource "google_project_service" "container_api" {
  service = "container.googleapis.com"
  
  disable_on_destroy = false
}

resource "google_project_service" "compute_api" {
  service = "compute.googleapis.com"
  
  disable_on_destroy = false
}

resource "google_project_service" "storage_api" {
  service = "storage.googleapis.com"
  
  disable_on_destroy = false
}

# Create VPC network
resource "google_compute_network" "vpc_network" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false

  depends_on = [google_project_service.compute_api]
}

# Create subnet
resource "google_compute_subnetwork" "subnet" {
  name          = "${var.cluster_name}-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.vpc_network.id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# Create GKE cluster
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone

  network    = google_compute_network.vpc_network.id
  subnetwork = google_compute_subnetwork.subnet.id

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Enable Autopilot features for better resource management
  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
    network_policy_config {
      disabled = false
    }
  }

  network_policy {
    enabled = true
  }

  depends_on = [
    google_project_service.container_api,
    google_project_service.compute_api,
  ]
}

# Create node pool
resource "google_container_node_pool" "primary_nodes" {
  name       = var.node_pool_name
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = var.node_count

  # Ensure APIs are fully ready before nodes try to join
  depends_on = [google_container_cluster.primary]

  node_config {
    spot         = false
    machine_type = var.machine_type
    disk_size_gb = var.disk_size_gb
    disk_type    = "pd-balanced"

    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    service_account = google_service_account.gke_node_sa.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = {
      purpose = "wf-data-collection"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  lifecycle {
    ignore_changes = [
      # GKE auto-populates these after creation; ignoring prevents spurious
      # UpdateNodePool calls that the API rejects when no real field changes.
      management,
      node_config[0].resource_labels,
      node_config[0].kubelet_config,
    ]
  }
}

# Service Account for GKE nodes
resource "google_service_account" "gke_node_sa" {
  account_id   = "${var.cluster_name}-node-sa"
  display_name = "GKE Node Service Account"
  description  = "Service account for GKE nodes in WF data collection cluster"
}

# IAM bindings for the node service account
resource "google_project_iam_member" "gke_node_sa_registry" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.gke_node_sa.email}"
}

# GCR now uses Artifact Registry as its backend — nodes need this to pull images.
resource "google_project_iam_member" "gke_node_sa_artifact_registry" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.gke_node_sa.email}"
}

resource "google_project_iam_member" "gke_node_sa_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gke_node_sa.email}"
}

resource "google_project_iam_member" "gke_node_sa_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gke_node_sa.email}"
}

# Service Account for Workload Identity (Argo Workflows)
resource "google_service_account" "argo_sa" {
  account_id   = "${var.cluster_name}-argo-sa"
  display_name = "Argo Workflows Service Account"
  description  = "Service account for Argo Workflows to access GCS and other resources"
}

# Wait for Argo SA to propagate in GCP IAM before assigning project-level roles
resource "time_sleep" "wait_for_argo_sa" {
  depends_on      = [google_service_account.argo_sa]
  create_duration = "30s"
}

# IAM bindings for Argo Workflows service account
resource "google_project_iam_member" "argo_sa_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.argo_sa.email}"

  depends_on = [time_sleep.wait_for_argo_sa]
}

resource "google_project_iam_member" "argo_sa_compute" {
  project = var.project_id
  role    = "roles/compute.instanceAdmin.v1"
  member  = "serviceAccount:${google_service_account.argo_sa.email}"

  depends_on = [time_sleep.wait_for_argo_sa]
}

# Bind Kubernetes service account to Google service account for Workload Identity
resource "google_service_account_iam_member" "argo_workload_identity" {
  service_account_id = google_service_account.argo_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[argo/argo-workflow]"

  depends_on = [google_container_cluster.primary]
}