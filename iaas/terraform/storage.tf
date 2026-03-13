# Create GCS bucket for storing pcap files and processed data
# GCS bucket is the sole durable store. The persistent disk used by the old
# PV/PVC approach has been removed — all pods now use node-local emptyDir
# scratch space and hand off data via GCS between pipeline steps.
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