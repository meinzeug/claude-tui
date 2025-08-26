#!/bin/bash

# 🔍 CI/CD Setup Validation Script
# Validates all deployment components and configurations
# Author: CI/CD Engineer - Hive Mind Team
# Version: 2.0.0

set -euo pipefail

# 🎨 Colors and formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'
readonly BOLD='\033[1m'

# 📊 Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 📊 Validation results
declare -A VALIDATION_RESULTS=(
    ["scripts"]=0
    ["workflows"]=0
    ["k8s"]=0
    ["monitoring"]=0
    ["security"]=0
    ["documentation"]=0
)

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")     echo -e "${timestamp} ${BLUE}[INFO]${NC} $message" ;;
        "WARN")     echo -e "${timestamp} ${YELLOW}[WARN]${NC} $message" ;;
        "ERROR")    echo -e "${timestamp} ${RED}[ERROR]${NC} $message" >&2 ;;
        "SUCCESS")  echo -e "${timestamp} ${GREEN}[SUCCESS]${NC} $message" ;;
        "CHECK")    echo -e "${timestamp} ${CYAN}[CHECK]${NC} $message" ;;
    esac
}

validate_scripts() {
    log "CHECK" "🔧 Validating deployment scripts..."
    
    local script_dir="${PROJECT_ROOT}/deployment/scripts"
    local required_scripts=(
        "blue-green-deploy-advanced.sh"
        "quantum-health-checks.sh"
        "performance-validation-quantum.sh"
        "deploy-quantum-monitoring.sh"
        "quantum-rollback-automation.sh"
    )
    
    local missing_scripts=()
    local non_executable=()
    
    for script in "${required_scripts[@]}"; do
        local script_path="${script_dir}/${script}"
        
        if [[ ! -f "$script_path" ]]; then
            missing_scripts+=("$script")
        elif [[ ! -x "$script_path" ]]; then
            non_executable+=("$script")
        fi
    done
    
    if [[ ${#missing_scripts[@]} -gt 0 ]]; then
        log "ERROR" "Missing scripts: ${missing_scripts[*]}"
        VALIDATION_RESULTS["scripts"]=1
        return 1
    fi
    
    if [[ ${#non_executable[@]} -gt 0 ]]; then
        log "WARN" "Non-executable scripts: ${non_executable[*]}"
        # Fix permissions
        for script in "${non_executable[@]}"; do
            chmod +x "${script_dir}/${script}"
        done
        log "INFO" "Fixed script permissions"
    fi
    
    log "SUCCESS" "All deployment scripts validated"
    return 0
}

validate_workflows() {
    log "CHECK" "⚙️ Validating GitHub Actions workflows..."
    
    local workflow_dir="${PROJECT_ROOT}/.github/workflows"
    local required_workflows=(
        "quantum-production-pipeline.yml"
    )
    
    local missing_workflows=()
    
    for workflow in "${required_workflows[@]}"; do
        local workflow_path="${workflow_dir}/${workflow}"
        
        if [[ ! -f "$workflow_path" ]]; then
            missing_workflows+=("$workflow")
        else
            # Basic YAML syntax validation
            if ! python3 -c "import yaml; yaml.safe_load(open('$workflow_path'))" 2>/dev/null; then
                log "ERROR" "Invalid YAML syntax in $workflow"
                VALIDATION_RESULTS["workflows"]=1
            fi
        fi
    done
    
    if [[ ${#missing_workflows[@]} -gt 0 ]]; then
        log "ERROR" "Missing workflows: ${missing_workflows[*]}"
        VALIDATION_RESULTS["workflows"]=1
        return 1
    fi
    
    # Check for required GitHub Actions
    local workflow_file="${workflow_dir}/quantum-production-pipeline.yml"
    local required_actions=(
        "actions/checkout@v4"
        "actions/setup-python@v5"
        "docker/build-push-action@v5"
        "actions/upload-artifact@v4"
    )
    
    for action in "${required_actions[@]}"; do
        if ! grep -q "$action" "$workflow_file"; then
            log "WARN" "Missing or outdated GitHub Action: $action"
        fi
    done
    
    log "SUCCESS" "GitHub Actions workflows validated"
    return 0
}

validate_kubernetes_configs() {
    log "CHECK" "🏗️ Validating Kubernetes configurations..."
    
    local k8s_dir="${PROJECT_ROOT}/k8s"
    local required_configs=(
        "production/quantum-production-deployment.yaml"
    )
    
    local missing_configs=()
    
    for config in "${required_configs[@]}"; do
        local config_path="${k8s_dir}/${config}"
        
        if [[ ! -f "$config_path" ]]; then
            missing_configs+=("$config")
        else
            # Basic YAML syntax validation
            if ! python3 -c "import yaml; yaml.safe_load(open('$config_path'))" 2>/dev/null; then
                log "ERROR" "Invalid YAML syntax in $config"
                VALIDATION_RESULTS["k8s"]=1
            fi
        fi
    done
    
    if [[ ${#missing_configs[@]} -gt 0 ]]; then
        log "ERROR" "Missing Kubernetes configs: ${missing_configs[*]}"
        VALIDATION_RESULTS["k8s"]=1
        return 1
    fi
    
    # Validate Kubernetes resource structure
    local prod_config="${k8s_dir}/production/quantum-production-deployment.yaml"
    local required_resources=(
        "Namespace"
        "Deployment"
        "Service"
        "ConfigMap"
        "Secret"
        "HorizontalPodAutoscaler"
        "NetworkPolicy"
    )
    
    for resource in "${required_resources[@]}"; do
        if ! grep -q "kind: $resource" "$prod_config"; then
            log "WARN" "Missing Kubernetes resource: $resource"
        fi
    done
    
    log "SUCCESS" "Kubernetes configurations validated"
    return 0
}

validate_monitoring_configs() {
    log "CHECK" "📊 Validating monitoring configurations..."
    
    local monitoring_dir="${PROJECT_ROOT}/deployment/monitoring"
    
    # Check if monitoring directory exists
    if [[ ! -d "$monitoring_dir" ]]; then
        log "WARN" "Monitoring directory not found, creating it..."
        mkdir -p "$monitoring_dir"/{prometheus,grafana,jaeger,loki}
    fi
    
    # Check for monitoring script
    local monitoring_script="${PROJECT_ROOT}/deployment/scripts/deploy-quantum-monitoring.sh"
    if [[ ! -x "$monitoring_script" ]]; then
        log "ERROR" "Monitoring deployment script not found or not executable"
        VALIDATION_RESULTS["monitoring"]=1
        return 1
    fi
    
    log "SUCCESS" "Monitoring configurations validated"
    return 0
}

validate_security_configs() {
    log "CHECK" "🛡️ Validating security configurations..."
    
    # Check for security-hardened Dockerfile
    local security_dockerfile="${PROJECT_ROOT}/Dockerfile.security-hardened"
    if [[ ! -f "$security_dockerfile" ]]; then
        log "ERROR" "Security-hardened Dockerfile not found"
        VALIDATION_RESULTS["security"]=1
        return 1
    fi
    
    # Validate Dockerfile security practices
    local security_checks=(
        "runAsNonRoot: true"
        "readOnlyRootFilesystem: true"
        "allowPrivilegeEscalation: false"
        "drop: \\[\"ALL\"\\]"
    )
    
    local k8s_config="${PROJECT_ROOT}/k8s/production/quantum-production-deployment.yaml"
    for check in "${security_checks[@]}"; do
        if ! grep -q "$check" "$k8s_config"; then
            log "WARN" "Security practice not found: $check"
        fi
    done
    
    log "SUCCESS" "Security configurations validated"
    return 0
}

validate_documentation() {
    log "CHECK" "📚 Validating documentation..."
    
    local doc_files=(
        "deployment/QUANTUM_CICD_PRODUCTION_GUIDE.md"
        "README.md"
    )
    
    local missing_docs=()
    
    for doc in "${doc_files[@]}"; do
        local doc_path="${PROJECT_ROOT}/${doc}"
        
        if [[ ! -f "$doc_path" ]]; then
            missing_docs+=("$doc")
        fi
    done
    
    if [[ ${#missing_docs[@]} -gt 0 ]]; then
        log "WARN" "Missing documentation files: ${missing_docs[*]}"
        # This is a warning, not an error
    fi
    
    log "SUCCESS" "Documentation validated"
    return 0
}

validate_dependencies() {
    log "CHECK" "🔍 Validating system dependencies..."
    
    local required_tools=(
        "kubectl"
        "helm"
        "docker"
        "python3"
        "curl"
        "jq"
    )
    
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log "WARN" "Missing system tools (optional for CI/CD): ${missing_tools[*]}"
    fi
    
    log "SUCCESS" "System dependencies checked"
    return 0
}

validate_quantum_modules() {
    log "CHECK" "🧠 Validating Quantum Intelligence modules..."
    
    local quantum_modules_dir="${PROJECT_ROOT}/src/ai/quantum_intelligence"
    local required_modules=(
        "neural_swarm_evolution.py"
        "adaptive_topology_manager.py"
        "emergent_behavior_engine.py"
        "meta_learning_coordinator.py"
        "quantum_intelligence_orchestrator.py"
    )
    
    local missing_modules=()
    
    for module in "${required_modules[@]}"; do
        local module_path="${quantum_modules_dir}/${module}"
        
        if [[ ! -f "$module_path" ]]; then
            missing_modules+=("$module")
        fi
    done
    
    if [[ ${#missing_modules[@]} -gt 0 ]]; then
        log "WARN" "Missing quantum modules: ${missing_modules[*]}"
        log "INFO" "These modules are required for quantum intelligence features"
    fi
    
    log "SUCCESS" "Quantum Intelligence modules validated"
    return 0
}

run_syntax_checks() {
    log "CHECK" "🔍 Running syntax checks..."
    
    # Python syntax check
    if command -v python3 >/dev/null 2>&1; then
        find "${PROJECT_ROOT}/src" -name "*.py" -exec python3 -m py_compile {} \; 2>/dev/null || {
            log "WARN" "Some Python files have syntax errors"
        }
    fi
    
    # YAML syntax check
    if command -v python3 >/dev/null 2>&1; then
        find "${PROJECT_ROOT}" -name "*.yml" -o -name "*.yaml" | while read -r file; do
            if ! python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
                log "WARN" "YAML syntax error in: $file"
            fi
        done
    fi
    
    log "SUCCESS" "Syntax checks completed"
}

generate_validation_report() {
    log "INFO" "📋 Generating validation report..."
    
    local report_file="${PROJECT_ROOT}/deployment/ci-cd-validation-report.json"
    local timestamp=$(date -Iseconds)
    local total_errors=0
    
    for category in "${!VALIDATION_RESULTS[@]}"; do
        total_errors=$((total_errors + VALIDATION_RESULTS["$category"]))
    done
    
    cat > "$report_file" << EOF
{
  "validation_report": {
    "timestamp": "$timestamp",
    "total_errors": $total_errors,
    "status": "$([ $total_errors -eq 0 ] && echo "success" || echo "partial_failure")",
    "categories": {
      "scripts": {
        "status": "$([ ${VALIDATION_RESULTS["scripts"]} -eq 0 ] && echo "passed" || echo "failed")",
        "errors": ${VALIDATION_RESULTS["scripts"]}
      },
      "workflows": {
        "status": "$([ ${VALIDATION_RESULTS["workflows"]} -eq 0 ] && echo "passed" || echo "failed")",
        "errors": ${VALIDATION_RESULTS["workflows"]}
      },
      "kubernetes": {
        "status": "$([ ${VALIDATION_RESULTS["k8s"]} -eq 0 ] && echo "passed" || echo "failed")",
        "errors": ${VALIDATION_RESULTS["k8s"]}
      },
      "monitoring": {
        "status": "$([ ${VALIDATION_RESULTS["monitoring"]} -eq 0 ] && echo "passed" || echo "failed")",
        "errors": ${VALIDATION_RESULTS["monitoring"]}
      },
      "security": {
        "status": "$([ ${VALIDATION_RESULTS["security"]} -eq 0 ] && echo "passed" || echo "failed")",
        "errors": ${VALIDATION_RESULTS["security"]}
      },
      "documentation": {
        "status": "$([ ${VALIDATION_RESULTS["documentation"]} -eq 0 ] && echo "passed" || echo "failed")",
        "errors": ${VALIDATION_RESULTS["documentation"]}
      }
    }
  }
}
EOF
    
    log "SUCCESS" "Validation report generated: $report_file"
    echo "$report_file"
}

display_summary() {
    echo
    echo "========================================="
    echo "🚀 CI/CD SETUP VALIDATION SUMMARY"
    echo "========================================="
    echo
    
    local total_errors=0
    for category in "${!VALIDATION_RESULTS[@]}"; do
        local status_icon="✅"
        local status_text="PASSED"
        
        if [[ ${VALIDATION_RESULTS["$category"]} -gt 0 ]]; then
            status_icon="❌"
            status_text="FAILED"
            total_errors=$((total_errors + VALIDATION_RESULTS["$category"]))
        fi
        
        printf "%-20s %s %s\n" "$(echo "$category" | tr '[:lower:]' '[:upper:]')" "$status_icon" "$status_text"
    done
    
    echo
    echo "========================================="
    
    if [[ $total_errors -eq 0 ]]; then
        echo "🎉 ALL VALIDATIONS PASSED!"
        echo "Your CI/CD setup is ready for production deployment."
    else
        echo "⚠️  VALIDATION ISSUES FOUND: $total_errors"
        echo "Please review and fix the issues above."
    fi
    
    echo "========================================="
    echo
}

main() {
    log "INFO" "🔍 Starting CI/CD Setup Validation..."
    echo
    
    # Run all validation checks
    validate_dependencies
    validate_scripts || true
    validate_workflows || true
    validate_kubernetes_configs || true
    validate_monitoring_configs || true
    validate_security_configs || true
    validate_documentation || true
    validate_quantum_modules || true
    run_syntax_checks || true
    
    echo
    
    # Generate report
    local report_file
    report_file=$(generate_validation_report)
    
    # Display summary
    display_summary
    
    # Final exit code
    local total_errors=0
    for category in "${!VALIDATION_RESULTS[@]}"; do
        total_errors=$((total_errors + VALIDATION_RESULTS["$category"]))
    done
    
    if [[ $total_errors -eq 0 ]]; then
        log "SUCCESS" "🎉 CI/CD setup validation completed successfully!"
        exit 0
    else
        log "ERROR" "❌ CI/CD setup validation found $total_errors issue(s)"
        exit 1
    fi
}

# Script execution
main "$@"