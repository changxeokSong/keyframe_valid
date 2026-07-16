"""프로젝트 전체가 공유하는 '로컬 저장' 고정 경로들.

실행할 때의 현재 폴더(cwd)에 의존하면, run.bat/run.sh를 어디서 부르는지에
따라 cache/reports/설정 파일이 엉뚱한 곳에 생길 수 있다. 그래서 여기서는
이 저장소 루트(ksl_validator/ 바로 위 폴더 = git pull 받은 그 폴더) 기준으로
전부 고정한다.

NAS 데이터셋/사전 사이트 동영상처럼 '읽는' 대상의 경로(dataset_root,
keyframe_images_dir 등)는 여기 대상이 아니다 — 그건 사용자가 지정하거나
dataset_config.py가 찾아내는 외부 위치다. 여기 있는 건 전부 이 컴퓨터에
'쓰는(저장하는)' 파일들이다.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 다운로드 캐시 (사전 사이트에서 받은 동영상/이미지 - 재생성 가능)
CACHE_DIR = PROJECT_ROOT / "cache"
VIDEOS_CACHE_DIR = CACHE_DIR / "videos"
HANDSHAPE_CACHE_DIR = CACHE_DIR / "handshape_images"

# 검증 결과/리뷰 기록
REPORTS_DIR = PROJECT_ROOT / "reports"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.csv"
REVIEW_LOG_PATH = REPORTS_DIR / "gui_review.csv"
SUMMARY_REPORT_PATH = REPORTS_DIR / "summary_report.md"
EXCEPTION_STAGING_DIR = REPORTS_DIR / "exception_staging"

# 이 컴퓨터 전용 설정 (NAS 마운트 경로 등, git에는 안 올라감)
LOCAL_SETTINGS_PATH = PROJECT_ROOT / ".local_settings.json"

# 함께 있는 기존 태깅 프로젝트(별도 git, 참고용 원본 파일 위치 추측에만 사용 - 읽기전용)
TAGGING_PROJECT_ROOT = PROJECT_ROOT / "online-sign-keyframe-detection-transformers"
DATASET_JSON_PATH = TAGGING_PROJECT_ROOT / "tools" / "tagging" / "config" / "dataset.json"
SAMPLE_EXCEPTION_CSV_PATH = (
    TAGGING_PROJECT_ROOT / "tools" / "tagging" / "config" / "etri_ksl_db" / "exception_videos.csv"
)
