#!/bin/bash
# Main deployment script for WF Data Collection Infrastructure
# This script automates the complete setup of GKE cluster, storage, and Argo workflows

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$ROOT_DIR/terraform"

# Check required tools
check_dependencies() {
    log_info "Checking dependencies..."
    
    local deps=("gcloud" "kubectl" "terraform" "docker")
    local missing=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install them before continuing"
        exit 1
    fi
    
    log_success "All dependencies found"
}

# Initialize Terraform configuration
init_terraform() {
    log_info "Initializing Terraform..."
    
    cd "$TERRAFORM_DIR"
    
    if [ ! -f "terraform.tfvars" ]; then
        log_warning "terraform.tfvars not found"
        log_info "Please copy terraform.tfvars.example to terraform.tfvars and fill in your values"
        cp terraform.tfvars.example terraform.tfvars
        log_error "Edit terraform.tfvars with your project details and re-run this script"
        exit 1
    fi
    
    terraform init
    log_success "Terraform initialized"
}

# Deploy infrastructure
deploy_infrastructure() {
    log_info "Deploying GKE cluster and storage infrastructure..."
    
    cd "$TERRAFORM_DIR"
    
    # Plan and apply
    terraform plan -out=tfplan
    terraform apply tfplan
    
    log_success "Infrastructure deployed"
    
    # Get outputs
    CLUSTER_NAME=$(terraform output -raw cluster_name)
    CLUSTER_LOCATION=$(terraform output -raw cluster_location)
    PROJECT_ID=$(terraform output -raw project_id)
    GCS_BUCKET=$(terraform output -raw gcs_bucket_name)
    ARGO_SA_EMAIL=$(terraform output -raw argo_service_account_email)

    log_info "Cluster: $CLUSTER_NAME in $CLUSTER_LOCATION"
    log_info "GCS Bucket: $GCS_BUCKET"
}

# Configure kubectl
configure_kubectl() {
    log_info "Configuring kubectl..."
    
    # Get cluster credentials
    gcloud container clusters get-credentials "$CLUSTER_NAME" \
        --zone="$CLUSTER_LOCATION" \
        --project="$PROJECT_ID"
    
    log_success "kubectl configured"
}

# Deploy Kubernetes resources
deploy_k8s_resources() {
    log_info "Deploying Kubernetes resources..."
    
    cd "$ROOT_DIR"
    
    # Substitute all placeholders with actual values from Terraform outputs
    sed -i.bak \
        -e "s|\${ARGO_SA_EMAIL}|${ARGO_SA_EMAIL}|g" \
        k8s-manifests/rbac.yaml

    sed -i.bak \
        -e "s|\${GCS_BUCKET}|${GCS_BUCKET}|g" \
        -e "s|\${PROJECT_ID}|${PROJECT_ID}|g" \
        k8s-manifests/storage.yaml

    # Apply Kubernetes manifests
    kubectl apply -f k8s-manifests/rbac.yaml
    kubectl apply -f k8s-manifests/storage.yaml
    
    log_success "Kubernetes resources deployed"
}

# Install Argo Workflows
install_argo() {
    log_info "Installing Argo Workflows..."
    
    cd "$ROOT_DIR"

    # Install Argo CRDs first (required before deploying controller/server)
    log_info "Installing Argo Workflows CRDs..."
    local ARGO_CRD_BASE="https://raw.githubusercontent.com/argoproj/argo-workflows/v3.5.5/manifests/base/crds/minimal"
    kubectl apply -f "${ARGO_CRD_BASE}/argoproj.io_workflows.yaml"
    kubectl apply -f "${ARGO_CRD_BASE}/argoproj.io_workflowtemplates.yaml"
    kubectl apply -f "${ARGO_CRD_BASE}/argoproj.io_clusterworkflowtemplates.yaml"
    kubectl apply -f "${ARGO_CRD_BASE}/argoproj.io_cronworkflows.yaml"
    kubectl apply -f "${ARGO_CRD_BASE}/argoproj.io_workflowtaskresults.yaml"
    kubectl apply -f "${ARGO_CRD_BASE}/argoproj.io_workflowtasksets.yaml"
    kubectl apply -f "${ARGO_CRD_BASE}/argoproj.io_workflowartifactgctasks.yaml"
    kubectl apply -f "${ARGO_CRD_BASE}/argoproj.io_workfloweventbindings.yaml"

    # Apply Argo controller, server, and configmap
    kubectl apply -f argo-workflows/install-argo.yaml

    # Force a rollout so pods immediately pick up any configmap changes
    kubectl rollout restart deployment/workflow-controller deployment/argo-server -n argo

    # Wait for Argo to be ready
    log_info "Waiting for Argo Workflows to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/workflow-controller -n argo
    kubectl wait --for=condition=available --timeout=300s deployment/argo-server -n argo

    # Substitute actual GCR project and bucket into workflow template
    sed -i.bak \
        -e "s|gcr.io/your-gcp-project-id|gcr.io/${PROJECT_ID}|g" \
        -e "s|wf-data-collection-storage-XXXXXXXX|${GCS_BUCKET}|g" \
        argo-workflows/wf-data-pipeline.yaml

    # Apply workflow template
    kubectl apply -f argo-workflows/wf-data-pipeline.yaml
    
    log_success "Argo Workflows installed and configured"
}

# Build and push Docker image
build_docker_image() {
    log_info "Building and pushing Docker image..."
    
    cd "$ROOT_DIR/docker"
    
    # Set environment variables for build script
    export PROJECT_ID="$PROJECT_ID"
    export TAG="latest"
    
    # Build and push the image
    ./build.sh push
    
    log_success "Docker image built and pushed"
}

# Get access information
show_access_info() {
    log_success "Deployment completed successfully!"
    echo ""
    echo "=== Access Information ==="
    echo "Cluster Name: $CLUSTER_NAME"
    echo "GCS Bucket: $GCS_BUCKET"
    echo ""
    
    # Get Argo server external IP
    ARGO_IP=""
    for i in {1..30}; do
        ARGO_IP=$(kubectl get service argo-server -n argo -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        if [ -n "$ARGO_IP" ]; then
            break
        fi
        log_info "Waiting for Argo server LoadBalancer IP... (attempt $i/30)"
        sleep 10
    done
    
    if [ -n "$ARGO_IP" ]; then
        echo "Argo Workflows UI: http://$ARGO_IP"
    else
        log_warning "Argo server LoadBalancer IP not yet available"
        echo "Check with: kubectl get service argo-server -n argo"
    fi
    
    echo ""
    echo "=== Next Steps ==="
    echo "1. Access Argo UI to monitor workflows"
    echo "2. Submit a workflow:"
    echo "   argo submit --from workflowtemplate/wf-data-collection-pipeline -n argo --parameter instances=10"
    echo "3. Monitor workflow progress:"
    echo "   argo list -n argo"
    echo "   argo logs <workflow-name> -n argo"
    echo ""
}

# Main deployment flow
main() {
    log_info "Starting WF Data Collection Infrastructure Deployment"
    
    check_dependencies
    init_terraform
    deploy_infrastructure
    configure_kubectl
    deploy_k8s_resources
    install_argo
    build_docker_image
    show_access_info
    
    log_success "Infrastructure deployment completed!"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "destroy")
        log_warning "Destroying infrastructure..."
        cd "$TERRAFORM_DIR"
        terraform destroy
        log_success "Infrastructure destroyed"
        ;;
    "status")
        log_info "Checking infrastructure status..."
        cd "$TERRAFORM_DIR"
        terraform show
        ;;
    *)
        echo "Usage: $0 [deploy|destroy|status]"
        echo "  deploy  - Deploy complete infrastructure (default)"
        echo "  destroy - Destroy infrastructure"
        echo "  status  - Show current infrastructure status"
        ;;
esac