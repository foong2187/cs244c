#!/bin/bash
# Quick workflow execution script for WF Data Collection

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Default parameters
INSTANCES=90
INTERFACE="eth0"
NUM_USERS=10
WORKFLOW_NAME=""
ARGO_INSTANCE_ID="argo-workflows-controller"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--instances)
            INSTANCES="$2"
            shift 2
            ;;
        --interface)
            INTERFACE="$2"
            shift 2
            ;;
        -u|--num-users)
            NUM_USERS="$2"
            shift 2
            ;;
        -n|--name)
            WORKFLOW_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  submit    - Submit a new workflow (default)"
            echo "  list      - List all workflows"
            echo "  logs      - Get logs for a workflow"
            echo "  delete    - Delete a workflow"
            echo "  status    - Get workflow status"
            echo ""
            echo "Options:"
            echo "  -i, --instances NUM    Number of instances per site (default: 90)"
            echo "  -u, --num-users NUM    Number of parallel collector pods (default: 10)"
            echo "  --interface IFACE      Network interface (default: eth0)"
            echo "  -n, --name NAME        Workflow name (for logs, delete, status commands)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 submit --instances 10"
            echo "  $0 list"
            echo "  $0 logs -n my-workflow"
            echo "  $0 status -n my-workflow"
            exit 0
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# Default command
COMMAND="${COMMAND:-submit}"

# Check if argo CLI is available
if ! command -v argo &> /dev/null; then
    log_error "argo CLI not found. Please install it first:"
    _OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    _ARCH=$(uname -m); [ "$_ARCH" = "x86_64" ] && _ARCH="amd64" || _ARCH="arm64"
    log_error "  curl -sLO https://github.com/argoproj/argo-workflows/releases/latest/download/argo-${_OS}-${_ARCH}.gz"
    log_error "  gunzip argo-${_OS}-${_ARCH}.gz"
    log_error "  chmod +x argo-${_OS}-${_ARCH}"
    log_error "  sudo mv argo-${_OS}-${_ARCH} /usr/local/bin/argo"
    log_error "Or on macOS: brew install argo"
    exit 1
fi

# Check if kubectl is configured
if ! kubectl get ns argo &>/dev/null; then
    log_error "Cannot access argo namespace. Make sure your cluster is deployed and kubectl is configured."
    log_error "Run: ./deploy.sh first"
    exit 1
fi

case "$COMMAND" in
    "submit")
        log_info "Submitting new workflow..."
        log_info "Instances: $INSTANCES"
        log_info "Interface: $INTERFACE"
        
        WORKFLOW_RESULT=$(argo submit \
            --from workflowtemplate/wf-data-collection-pipeline \
            -n argo \
            --instanceid "$ARGO_INSTANCE_ID" \
            --parameter instances="$INSTANCES" \
            --parameter interface="$INTERFACE" \
            --parameter num-users="$NUM_USERS" \
            --generate-name="wf-collection-" \
            -o name)

        WORKFLOW_NAME=$(echo "$WORKFLOW_RESULT" | cut -d'/' -f2)
        
        log_success "Workflow submitted: $WORKFLOW_NAME"
        log_info "Monitor progress with: $0 status -n $WORKFLOW_NAME"
        log_info "View logs with: $0 logs -n $WORKFLOW_NAME"
        ;;
        
    "list")
        log_info "Listing workflows..."
        argo list -n argo --instanceid "$ARGO_INSTANCE_ID"
        ;;
        
    "logs")
        if [ -z "$WORKFLOW_NAME" ]; then
            log_error "Workflow name required. Use -n or --name option."
            exit 1
        fi
        log_info "Getting logs for workflow: $WORKFLOW_NAME"
        argo logs "$WORKFLOW_NAME" -n argo --instanceid "$ARGO_INSTANCE_ID" -f
        ;;
        
    "status")
        if [ -z "$WORKFLOW_NAME" ]; then
            log_error "Workflow name required. Use -n or --name option."
            exit 1
        fi
        log_info "Getting status for workflow: $WORKFLOW_NAME"
        argo get "$WORKFLOW_NAME" -n argo --instanceid "$ARGO_INSTANCE_ID"
        ;;
        
    "delete")
        if [ -z "$WORKFLOW_NAME" ]; then
            log_error "Workflow name required. Use -n or --name option."
            exit 1
        fi
        log_warning "Deleting workflow: $WORKFLOW_NAME"
        argo delete "$WORKFLOW_NAME" -n argo --instanceid "$ARGO_INSTANCE_ID"
        log_success "Workflow deleted"
        ;;
        
    "ui")
        log_info "Getting Argo UI access information..."
        ARGO_IP=$(kubectl get service argo-server -n argo -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        
        if [ -n "$ARGO_IP" ]; then
            log_success "Argo UI available at: http://$ARGO_IP"
        else
            log_warning "Argo UI LoadBalancer IP not yet available"
            log_info "Port-forward to access locally:"
            echo "kubectl port-forward svc/argo-server -n argo 8080:80"
            echo "Then access: http://localhost:8080"
        fi
        ;;
        
    *)
        log_error "Unknown command: $COMMAND"
        log_info "Use -h or --help for usage information"
        exit 1
        ;;
esac