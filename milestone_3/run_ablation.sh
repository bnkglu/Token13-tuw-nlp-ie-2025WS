#!/bin/bash
#
# Ablation Study Runner
# Runs all configuration combinations for MS2 Baseline vs Semantic Augmented comparison
# Results are collected in a central ablation/results/ folder
#
# Prediction Modes:
#   - first_match    : Rules ranked by precision & support, first matching rule wins
#   - priority_based : Rules ranked by semantic pattern type tier, best matching rule wins
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
RESULTS_DIR="${SCRIPT_DIR}/results"
ENV_DIR="${SCRIPT_DIR}/.venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_running() {
    echo -e "${YELLOW}[→]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[i]${NC} $1"
}

# ============================================================================
# Environment Setup - Always uses isolated .venv in ablation folder
# ============================================================================

setup_environment() {
    print_header "ENVIRONMENT SETUP"
    
    print_info "Using isolated environment at: ${ENV_DIR}"
    print_info "Your current environment will NOT be modified."
    echo ""
    
    if [[ -d "${ENV_DIR}" ]]; then
        print_status "Found existing environment"
        activate_env
        
        if ! python -c "import spacy; spacy.load('en_core_web_lg')" 2>/dev/null; then
            install_dependencies
        else
            print_status "Dependencies already installed"
        fi
    else
        print_running "Creating new virtual environment..."
        python3 -m venv "${ENV_DIR}"
        print_status "Virtual environment created"
        
        activate_env
        install_dependencies
    fi
    
    echo ""
    print_status "Environment ready: $(which python)"
}

activate_env() {
    print_running "Activating isolated environment..."
    source "${ENV_DIR}/bin/activate"
}

install_dependencies() {
    print_running "Installing dependencies..."
    pip install -q --upgrade pip
    pip install -q -r "${SCRIPT_DIR}/requirements.txt"
    print_status "Python packages installed"
    
    print_running "Downloading spaCy model (en_core_web_lg)..."
    python -m spacy download en_core_web_lg -q
    print_status "spaCy model downloaded"
    
    print_running "Downloading NLTK data..."
    python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('framenet_v17', quiet=True); nltk.download('omw-1.4', quiet=True)" 2>/dev/null
    print_status "NLTK data ready"
}

# ============================================================================
# Experiment Running
# ============================================================================

run_experiment() {
    local name="$1"
    local dir="$2"
    local args="$3"
    local result_name="$4"
    local log_file="${LOG_DIR}/${name}.log"
    local central_result_dir="${RESULTS_DIR}/${result_name}"
    
    print_running "Running: ${name}"
    echo "  Output: ${central_result_dir}"
    echo "  Log: ${log_file}"
    
    # Create output directory first
    mkdir -p "${central_result_dir}"
    
    cd "${dir}"
    # Pass output directory directly to main.py - results save directly there
    if python main.py ${args} --output-dir "${central_result_dir}" > "${log_file}" 2>&1; then
        print_status "Completed: ${name}"
        print_status "Results saved to: ${central_result_dir}"
    else
        echo -e "${RED}[✗] Failed: ${name}${NC}"
        echo "  Check log: ${log_file}"
        tail -20 "${log_file}"
    fi
    cd "${SCRIPT_DIR}"
}

run_all() {
    print_header "RUNNING COMPLETE ABLATION STUDY"
    
    mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    echo "Timestamp: ${TIMESTAMP}"
    echo "All results will be saved to: ${RESULTS_DIR}"
    echo ""
    
    # 1. FN & WN OFF + first_match (BASELINE)
    print_header "1/4 - FN & WN OFF + first_match (BASELINE)"
    run_experiment "fn_wn_off_first_match_${TIMESTAMP}" \
        "${SCRIPT_DIR}/statistical_rule_based_system" \
        "" \
        "01_fn_wn_off_first_match"
    
    # 2. FN & WN OFF + priority_based
    print_header "2/4 - FN & WN OFF + priority_based"
    run_experiment "fn_wn_off_priority_based_${TIMESTAMP}" \
        "${SCRIPT_DIR}/statistical_rule_based_system" \
        "--prediction-mode priority_based" \
        "02_fn_wn_off_priority_based"
    
    # 3. FN & WN ON + first_match
    print_header "3/4 - FN & WN ON + first_match"
    run_experiment "fn_wn_on_first_match_${TIMESTAMP}" \
        "${SCRIPT_DIR}/statistical_rule_based_system" \
        "--use-semantics" \
        "03_fn_wn_on_first_match"
    
    # 4. FN & WN ON + priority_based
    print_header "4/4 - FN & WN ON + priority_based"
    run_experiment "fn_wn_on_priority_based_${TIMESTAMP}" \
        "${SCRIPT_DIR}/statistical_rule_based_system" \
        "--use-semantics --prediction-mode priority_based" \
        "04_fn_wn_on_priority_based"
    
    generate_summary
    
    print_header "ABLATION STUDY COMPLETE"
    echo "All results saved to: ${RESULTS_DIR}"
    echo ""
    ls -la "${RESULTS_DIR}"
}

generate_summary() {
    print_header "GENERATING ABLATION SUMMARY"
    
    SUMMARY_FILE="${RESULTS_DIR}/ABLATION_SUMMARY.md"
    
    cat > "${SUMMARY_FILE}" << 'EOF'
# Ablation Study Results

## Experiment Configurations

| # | Config Name | FrameNet & WordNet | Prediction Mode | Description |
|---|-------------|--------------------|-----------------| ------------|
| 1 | Baseline (FN & WN OFF) | OFF | first_match | Statistical system (Semantics OFF) |
| 2 | FN & WN OFF + priority | OFF | priority_based | Statistical system (Priority Ranking) |
| 3 | FN & WN ON + first_match | ON | first_match | Statistical system (Semantics ON) |
| 4 | FN & WN ON + priority | ON | priority_based | Statistical system (Priority & Semantics) |

### Prediction Modes:
- **first_match** : Rules ranked by precision & support, first matching rule wins
- **priority_based** : Rules ranked by pattern type (PREP_STRUCT > LEXNAME > BIGRAM > SYNSET/FRAME/LEMMA)

## Summary Statistics (Test Set)

| Configuration | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---------------|----------|-----------------|--------------|----------|
EOF

    # Extract summary stats from each experiment
    for dir in "${RESULTS_DIR}"/0*; do
        if [[ -d "$dir" ]]; then
            name=$(basename "$dir")
            eval_file="${dir}/evaluation/evaluation_results.txt"
            if [[ ! -f "$eval_file" ]]; then
                eval_file="${dir}/evaluation_results.txt"
            fi
            
            if [[ -f "$eval_file" ]]; then
                accuracy=$(grep -A 30 "TEST SET RESULTS" "$eval_file" | grep "accuracy" | awk '{print $2}' | head -1)
                macro_line=$(grep -A 30 "TEST SET RESULTS" "$eval_file" | grep "macro avg" | head -1)
                macro_prec=$(echo "$macro_line" | awk '{print $3}')
                macro_rec=$(echo "$macro_line" | awk '{print $4}')
                macro_f1=$(echo "$macro_line" | awk '{print $5}')
                
                echo "| ${name} | ${accuracy:-N/A} | ${macro_prec:-N/A} | ${macro_rec:-N/A} | ${macro_f1:-N/A} |" >> "${SUMMARY_FILE}"
            fi
        fi
    done
    
    echo "" >> "${SUMMARY_FILE}"
    echo "## Detailed Results" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"

    for dir in "${RESULTS_DIR}"/0*; do
        if [[ -d "$dir" ]]; then
            name=$(basename "$dir")
            echo "### ${name}" >> "${SUMMARY_FILE}"
            echo "" >> "${SUMMARY_FILE}"
            
            eval_file="${dir}/evaluation/evaluation_results.txt"
            if [[ ! -f "$eval_file" ]]; then
                eval_file="${dir}/evaluation_results.txt"
            fi
            
            if [[ -f "$eval_file" ]]; then
                echo '```' >> "${SUMMARY_FILE}"
                grep -A 30 "TEST SET RESULTS" "$eval_file" >> "${SUMMARY_FILE}" || true
                echo '```' >> "${SUMMARY_FILE}"
            else
                echo "_Results not found_" >> "${SUMMARY_FILE}"
            fi
            echo "" >> "${SUMMARY_FILE}"
        fi
    done
    
    echo "---" >> "${SUMMARY_FILE}"
    echo "Generated: $(date)" >> "${SUMMARY_FILE}"
    print_status "Summary saved to: ${SUMMARY_FILE}"

    comparative_analysis
}

comparative_analysis() {
    print_header "RUNNING COMPARATIVE ANALYSIS"
    
    cat >> "${SUMMARY_FILE}" << 'EOF'

## Comparative Analysis
Analysis of agreement between **first_match** (Baseline) and **priority_based** strategies.

| Comparison | Same Rule % | Different Rule % | Label Agreement (Overall) | Label Agreement (Diff Rule) |
|------------|-------------|------------------|---------------------------|-----------------------------|
EOF

    # Define pairs to compare
    # 1. FN/WN OFF: 01 vs 02
    # 2. FN/WN ON:  03 vs 04
    
    # Python script to compute overlap
    local script=$(cat << 'PYTHON_EOF'
import json
import sys
from pathlib import Path

def analyze(p1_path, p2_path, name):
    try:
        if not Path(p1_path).exists() or not Path(p2_path).exists():
            return None
            
        with open(p1_path) as f: p1 = json.load(f)
        with open(p2_path) as f: p2 = json.load(f)
        
        # Build dicts by ID for safety
        d1 = {x['id']: x for x in p1}
        d2 = {x['id']: x for x in p2}
        
        common_ids = set(d1.keys()) & set(d2.keys())
        if not common_ids:
            return None
            
        total = len(common_ids)
        same_rule = 0
        same_label = 0
        diff_rule_same_label = 0
        diff_rule_count = 0
        
        for pid in common_ids:
            # Handle different field names if necessary (statistical uses 'relation_directed' sometimes? 
            # But the JSONs usually have 'predicted_label' standardized now? 
            # Actually statistical system output has 'predicted_label' or 'relation_directed' depending on version. 
            # Let's check keys.)
            
            item1 = d1[pid]
            item2 = d2[pid]
            
            # Extract rule name (first word of triggered_rule typically)
            # triggered_rule string: "Rule Name: Explanation..."
            r1 = item1.get('triggered_rule', '').split(':')[0]
            r2 = item2.get('triggered_rule', '').split(':')[0]
            
            # Extract label
            l1 = item1.get('predicted_label') or item1.get('relation_directed')
            l2 = item2.get('predicted_label') or item2.get('relation_directed')
            
            if r1 == r2:
                same_rule += 1
            else:
                diff_rule_count += 1
                if l1 == l2:
                    diff_rule_same_label += 1
            
            if l1 == l2:
                same_label += 1
                
        same_rule_pct = same_rule / total * 100
        diff_rule_pct = diff_rule_count / total * 100
        label_agree_pct = same_label / total * 100
        
        # Agreement when rules differ
        if diff_rule_count > 0:
            diff_rule_agree_pct = diff_rule_same_label / diff_rule_count * 100
        else:
            diff_rule_agree_pct = 0.0
            
        print(f"| {name} | {same_rule_pct:.1f}% | {diff_rule_pct:.1f}% | {label_agree_pct:.1f}% | {diff_rule_agree_pct:.1f}% |")
        
    except Exception as e:
        # print(f"Error: {e}", file=sys.stderr)
        pass

# Compare OFF
analyze(sys.argv[1], sys.argv[2], "FN/WN OFF (Baseline vs Priority)")
# Compare ON
analyze(sys.argv[3], sys.argv[4], "FN/WN ON (First vs Priority)")
PYTHON_EOF
)

    # Resolve paths
    # We find folders starting with 01, 02, 03, 04
    dir01=$(find "${RESULTS_DIR}" -maxdepth 1 -name "01_*" -type d | head -1)
    dir02=$(find "${RESULTS_DIR}" -maxdepth 1 -name "02_*" -type d | head -1)
    dir03=$(find "${RESULTS_DIR}" -maxdepth 1 -name "03_*" -type d | head -1)
    dir04=$(find "${RESULTS_DIR}" -maxdepth 1 -name "04_*" -type d | head -1)
    
    file01="${dir01}/predictions/test_predictions.json"
    file02="${dir02}/predictions/test_predictions.json"
    file03="${dir03}/predictions/test_predictions.json"
    file04="${dir04}/predictions/test_predictions.json"
    
    python3 -c "$script" "$file01" "$file02" "$file03" "$file04" >> "${SUMMARY_FILE}"

    cat >> "${SUMMARY_FILE}" << 'EOF'

### Key Insights
- **Different Rule %**: Shows how often `priority_based` picked a different rule than `first_match`.
- **Label Agreement (Diff Rule)**: Shows how often they agreed on the label *even when choosing different rules*.
- High agreement (99%+) implies the system is robust: Syntactic and Semantic rules reinforce each other.

EOF
}


run_baseline() {
    mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
    print_header "RUNNING MS2 BASELINE"
    # Assuming baseline is still in ms2_rule_based_baseline if it exists, otherwise warn
    if [[ -d "${SCRIPT_DIR}/ms2_rule_based_baseline" ]]; then
        run_experiment "baseline" \
            "${SCRIPT_DIR}/ms2_rule_based_baseline" \
            "" \
            "01_ms2_baseline"
    else
        echo "Baseline folder not found in ${SCRIPT_DIR}. Skipping."
    fi
}

run_fn_wn_off() {
    mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
    print_header "RUNNING FN & WN OFF"
    run_experiment "fn_wn_off" \
        "${SCRIPT_DIR}/statistical_rule_based_system" \
        "" \
        "02_fn_wn_off"
}

run_fn_wn_on_first() {
    mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
    print_header "RUNNING FN & WN ON + FIRST_MATCH"
    run_experiment "fn_wn_on_first_match" \
        "${SCRIPT_DIR}/statistical_rule_based_system" \
        "--use-semantics" \
        "03_fn_wn_on_first_match"
}

run_fn_wn_on_priority() {
    mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
    print_header "RUNNING FN & WN ON + PRIORITY_BASED"
    run_experiment "fn_wn_on_priority_based" \
        "${SCRIPT_DIR}/statistical_rule_based_system" \
        "--use-semantics --prediction-mode priority_based" \
        "04_fn_wn_on_priority_based"
}

# Interactive menu
show_menu() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║           ABLATION STUDY - Configuration Selection             ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Prediction Modes:"
    echo "  - first_match    : Rules ranked by precision & support"
    echo "  - priority_based : Rules ranked by pattern type (PREP_STRUCT > LEXNAME > BIGRAM > SYNSET/FRAME/LEMMA)"
    echo ""
    echo "Select what to run:"
    echo ""
    echo "  1) Run ALL experiments (4 configurations)"
    echo "  2) Run FrameNet & WordNet OFF + first_match -(BASELINE)"
    echo "  3) Run FrameNet & WordNet OFF + priority_based"
    echo "  4) Run FrameNet & WordNet ON + first_match"
    echo "  5) Run FrameNet & WordNet ON + priority_based"
    echo "  6) Exit"
    echo ""
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    # Always setup isolated environment first (unless --skip-setup)
    if [[ "$1" != "--skip-setup" ]] && [[ "$1" != "--help" ]] && [[ "$1" != "-h" ]]; then
        setup_environment
    fi
    
    # Handle command line args
    local run_arg="${1}"
    [[ "$1" == "--skip-setup" ]] && run_arg="${2}"
    
    case "${run_arg}" in
        --all)
            run_all
            ;;
        --help|-h)
            echo "Usage: $0 [OPTION]"
            echo ""
            echo "Options:"
            echo "  --all         Run all 4 ablation experiments"
            echo "  --skip-setup  Skip environment setup (use current env)"
            echo "  -h, --help    Show this help"
            echo ""
            echo "Prediction Modes:"
            echo "  first_match    - Rules ranked by precision & support, first match wins"
            echo "  priority_based - Rules ranked by pattern type tier, best match wins"
            echo ""
            echo "Environment: Creates isolated .venv in ablation folder"
            echo "Results: Saved to ${RESULTS_DIR}"
            ;;
        *)
            # Interactive mode
            show_menu
            read -p "Enter choice [1-6]: " choice
            
            case $choice in
                1) run_all ;;
                2) run_experiment "fn_wn_off_first_match_${TIMESTAMP}" "${SCRIPT_DIR}/statistical_rule_based_system" "" "01_fn_wn_off_first_match" ;;
                3) run_experiment "fn_wn_off_priority_based_${TIMESTAMP}" "${SCRIPT_DIR}/statistical_rule_based_system" "--prediction-mode priority_based" "02_fn_wn_off_priority_based" ;;
                4) run_experiment "fn_wn_on_first_match_${TIMESTAMP}" "${SCRIPT_DIR}/statistical_rule_based_system" "--use-semantics" "03_fn_wn_on_first_match" ;;
                5) run_experiment "fn_wn_on_priority_based_${TIMESTAMP}" "${SCRIPT_DIR}/statistical_rule_based_system" "--use-semantics --prediction-mode priority_based" "04_fn_wn_on_priority_based" ;;
                6) echo "Exiting."; exit 0 ;;
                *) echo "Invalid choice"; exit 1 ;;
            esac
            ;;
    esac
}

main "$@"