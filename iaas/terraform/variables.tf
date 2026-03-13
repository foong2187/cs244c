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
  description = "Machine type for the single GKE node. Must have enough CPU/RAM for all parallel collector pods. With 10 pods each requesting 2 CPU / 4 Gi, you need at least 20 vCPU / 40 Gi — e2-standard-32 (32 vCPU / 128 Gi) gives comfortable headroom."
  type        = string
  default     = "e2-standard-32"
}

variable "disk_size_gb" {
  description = "Node-local disk size in GB. All collector pods share this single node's disk via emptyDir. With 10 users each capturing ~2 GB of pcaps, 400 GB gives ample headroom for pcaps, pickles, and analysis outputs coexisting during processing steps."
  type        = number
  default     = 400
}

variable "node_count" {
  description = "Fixed number of nodes. Set to 1 — all parallelism comes from multiple pods on the same node, not from multiple nodes."
  type        = number
  default     = 1
}

variable "gcs_bucket_name" {
  description = "GCS bucket name for data storage"
  type        = string
  default     = "wflab-storage"
}
