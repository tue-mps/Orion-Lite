#!/bin/bash
# Run one or both Orion-Lite training smoke tests on distill_data_test.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-all}"

case "${MODE}" in
    plain)
        bash "${REPO_ROOT}/distill/test_train_student.sh"
        ;;
    orion_loss)
        bash "${REPO_ROOT}/distill/test_train_student_with_orion_loss.sh"
        ;;
    all)
        bash "${REPO_ROOT}/distill/test_train_student.sh"
        bash "${REPO_ROOT}/distill/test_train_student_with_orion_loss.sh"
        ;;
    *)
        echo "Invalid mode '${MODE}'. Use plain, orion_loss, or all."
        exit 1
        ;;
esac
