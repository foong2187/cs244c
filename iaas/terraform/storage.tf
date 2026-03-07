# Create GCS bucket for storing pcap files and processed data
resource "google_storage_bucket" "wf_data_bucket" {
  name          = "${var.gcs_bucket_name}-${random_id.bucket_suffix.hex}"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  # Enable object versioning for important data
  lifecycle_rule {
    condition {
      age                   = 7
      with_state           = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    purpose     = "wf-data-collection"
    environment = "research"
  }
}

# Random suffix for bucket name to ensure uniqueness
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Create bucket folders/prefixes for organization
# resource "google_storage_bucket_object" "pcap_folder" {
#   name    = "pcap/"
#   bucket  = google_storage_bucket.wf_data_bucket.name
#   content = " "
# }

# resource "google_storage_bucket_object" "pickle_folder" {
#   name    = "pickle/"
#   bucket  = google_storage_bucket.wf_data_bucket.name
#   content = " "
# }

# resource "google_storage_bucket_object" "analysis_folder" {
#   name    = "analysis/"
#   bucket  = google_storage_bucket.wf_data_bucket.name
#   content = " "
# }

# resource "google_storage_bucket_object" "logs_folder" {
#   name    = "logs/"
#   bucket  = google_storage_bucket.wf_data_bucket.name
#   content = " "
# }

# Single persistent disk for all workflow data.
# Sub-directories within the volume separate concerns:
#   pcap/       – raw tcpdump captures (~500 Gi worth of headroom)
#   pickle/     – processed direction-sequence numpy arrays
#   analysis/   – EDA plots and statistics
#   logs/       – collection.log, progress.csv
resource "google_compute_disk" "data_storage_disk" {
  name  = "${var.cluster_name}-data-storage"
  type  = "pd-balanced"
  zone  = var.zone
  size  = 200

  labels = {
    purpose = "wf-data-storage"
  }

  depends_on = [google_project_service.compute_api]
}