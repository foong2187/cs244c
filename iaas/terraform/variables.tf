variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "wflab"
}

variable "node_pool_name" {
  description = "GKE node pool name"
  type        = string
  default     = "wflab-nodes"
}

variable "machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "e2-standard-4"
}

variable "disk_size_gb" {
  description = "Disk size for GKE nodes in GB"
  type        = number
  default     = 100
}

variable "min_node_count" {
  description = "Minimum number of nodes in the node pool"
  type        = number
  default     = 1
}

variable "max_node_count" {
  description = "Maximum number of nodes in the node pool"
  type        = number
  default     = 5
}

variable "initial_node_count" {
  description = "Initial number of nodes in the node pool"
  type        = number
  default     = 2
}

variable "gcs_bucket_name" {
  description = "GCS bucket name for data storage"
  type        = string
  default     = "wflab-storage"
}