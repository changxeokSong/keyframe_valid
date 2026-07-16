#!/usr/bin/env bash
# ksl_validator 앱 원클릭 실행 스크립트.
# conda 가상환경 또는 python venv 중 골라서 자동으로 만들고, 의존성 설치 후 실행한다.
#
# 사용법:
#   ./run.sh [--conda|--venv] <ksl_validator 명령 인자...>
#
# 예:
#   ./run.sh --conda fetch --origin-no 8240
#   ./run.sh --venv validate --metadata sample.xlsx --limit 5
#   ./run.sh fetch --origin-no 8240        # --conda/--venv 생략 시 conda 있으면 conda, 없으면 venv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="keyframe_valid"
VENV_DIR="$SCRIPT_DIR/.venv"
ENV_TYPE=""

usage() {
  echo "사용법: ./run.sh [--conda|--venv] [ksl_validator 명령 인자...]"
  echo "  --conda   conda 가상환경(${ENV_NAME}) 사용"
  echo "  --venv    python -m venv(.venv) 사용"
  echo "  옵션 생략 시 conda가 있으면 conda, 없으면 venv 자동 선택"
  echo
  echo "예:"
  echo "  ./run.sh --conda fetch --origin-no 8240"
  echo "  ./run.sh --venv validate --metadata sample.xlsx --limit 5"
  exit 1
}

case "${1:-}" in
  --conda) ENV_TYPE="conda"; shift ;;
  --venv)  ENV_TYPE="venv"; shift ;;
  -h|--help) usage ;;
esac

if [[ -z "$ENV_TYPE" ]]; then
  if command -v conda >/dev/null 2>&1; then
    ENV_TYPE="conda"
  else
    ENV_TYPE="venv"
  fi
fi

if [[ "$ENV_TYPE" == "conda" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[run.sh] conda를 찾을 수 없습니다. ./run.sh --venv 를 사용하세요." >&2
    exit 1
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "[run.sh] conda 환경 '${ENV_NAME}' 생성 중..."
    conda create -y -n "$ENV_NAME" python=3.11
  fi
  conda activate "$ENV_NAME"
  PYTHON_BIN="python"
elif [[ "$ENV_TYPE" == "venv" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[run.sh] python venv 생성 중... ($VENV_DIR)"
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="python"
else
  echo "[run.sh] 알 수 없는 환경 타입: $ENV_TYPE" >&2
  usage
fi

echo "[run.sh] 의존성 확인/설치 중 (${ENV_TYPE} 환경)..."
"$PYTHON_BIN" -m pip install -q -r requirements.txt

echo "[run.sh] ksl_validator 실행 (${ENV_TYPE} 환경): $*"
exec "$PYTHON_BIN" -m ksl_validator "$@"
