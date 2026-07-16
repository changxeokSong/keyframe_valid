"""ksl_validator 커맨드라인 앱.

사용 예:
  # 사전 동영상 하나만 받기
  python -m ksl_validator fetch --origin-no 8240 --out cache/videos

  # 이미지 한 장을 특정 origin_no 동영상 전체 프레임과 pose 비교
  python -m ksl_validator compare-image --keyframe my.jpg --origin-no 8240

  # xlsx 메타데이터 전체를 훑으며 라벨 검증 리포트 생성 (NAS 없이도 다운로드는 됨,
  # 다만 '내 키프레임'은 --dataset-root가 마운트돼 있어야 추출 가능)
  python -m ksl_validator validate --metadata sample.xlsx --dataset-root /Volumes/nfs_shared/... --limit 20

  # GUI로 검토 (메타데이터 로드 → 검증 실행 → 눈으로 비교 → 예외처리/복구)
  python -m ksl_validator gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .metadata import load_dataset
from .pipeline import (
    Extractors,
    ValidationResult,
    scan_video_for_best_match,
    validate_entry,
    run_validation,
    extract_keyframe_from_image,
)
from . import sldict_client

DEFAULT_CACHE_DIR = Path("cache/videos")
DEFAULT_REPORT_PATH = Path("reports/validation_report.csv")


def cmd_fetch(args: argparse.Namespace) -> None:
    path = sldict_client.fetch_and_download(
        args.origin_no, Path(args.out), category=args.category, overwrite=args.overwrite
    )
    print(f"다운로드 완료: {path}")


def cmd_compare_image(args: argparse.Namespace) -> None:
    extractors = Extractors(args.model)
    my_sig, note = extract_keyframe_from_image(Path(args.keyframe), extractors)
    if my_sig is None:
        print(f"오류: {note}", file=sys.stderr)
        sys.exit(1)

    video_path = sldict_client.fetch_and_download(
        args.origin_no, Path(args.cache_dir), category=args.category
    )
    print(f"동영상: {video_path}")

    best_score, best_idx, hand_used, scanned = scan_video_for_best_match(
        video_path, my_sig, extractors, stride=args.stride
    )
    print(f"스캔한 프레임 수: {scanned}")
    if best_score is None:
        print("동영상에서 사람 pose를 검출하지 못했습니다.")
        return
    print(f"최고 유사도: {best_score:.4f} (프레임 #{best_idx}, 손모양 반영: {'Y' if hand_used else 'N'})")
    verdict = "일치 가능성 높음 (참고용 - 실제로 프레임을 눈으로 확인하세요)" if best_score >= args.threshold else "불일치 의심"
    print(f"판정 (threshold={args.threshold}): {verdict}")


def cmd_validate(args: argparse.Namespace) -> None:
    entries = load_dataset(Path(args.metadata))
    if args.only:
        wanted = set(args.only.split(","))
        entries = [e for e in entries if e.origin_no in wanted]
    if args.limit:
        entries = entries[: args.limit]

    print(f"검증 대상: {len(entries)}개 항목")

    def progress(i, total, result: ValidationResult):
        score_str = f"{result.best_score:.3f}" if result.best_score is not None else "-"
        print(f"[{i}/{total}] origin_no={result.origin_no} gloss={result.gloss_name} "
              f"status={result.status} score={score_str}")

    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    keyframe_images_dir = Path(args.keyframe_images_dir) if args.keyframe_images_dir else None
    results = run_validation(
        entries,
        report_path=Path(args.report),
        cache_dir=Path(args.cache_dir),
        dataset_root=dataset_root,
        keyframe_images_dir=keyframe_images_dir,
        threshold=args.threshold,
        stride=args.stride,
        model_name=args.model,
        progress_cb=progress,
    )

    suspects = [r for r in results if r.status == "SUSPECT"]
    no_kf = [r for r in results if r.status == "NO_MY_KEYFRAME"]
    print(f"\n=== 요약 ===")
    print(f"MATCH: {sum(1 for r in results if r.status == 'MATCH')}")
    print(f"SUSPECT(라벨 불일치 의심): {len(suspects)}")
    print(f"NO_MY_KEYFRAME(내 키프레임 확보 실패): {len(no_kf)}")
    print(f"DOWNLOAD_FAILED: {sum(1 for r in results if r.status == 'DOWNLOAD_FAILED')}")
    print(f"NO_POSE_DETECTED: {sum(1 for r in results if r.status == 'NO_POSE_DETECTED')}")
    print(f"리포트 저장: {args.report}")


def cmd_gui(args: argparse.Namespace) -> None:
    from PyQt5.QtWidgets import QApplication

    from .gui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksl_validator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="sldict.korean.go.kr에서 동영상 하나 다운로드")
    p_fetch.add_argument("--origin-no", required=True)
    p_fetch.add_argument("--category", default="")
    p_fetch.add_argument("--out", default=str(DEFAULT_CACHE_DIR))
    p_fetch.add_argument("--overwrite", action="store_true")
    p_fetch.set_defaults(func=cmd_fetch)

    p_cmp = sub.add_parser("compare-image", help="이미지 한 장 vs 특정 origin_no 동영상 pose 비교")
    p_cmp.add_argument("--keyframe", required=True, help="비교할 키프레임 이미지 경로")
    p_cmp.add_argument("--origin-no", required=True)
    p_cmp.add_argument("--category", default="")
    p_cmp.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p_cmp.add_argument("--threshold", type=float, default=0.7)
    p_cmp.add_argument("--stride", type=int, default=1)
    p_cmp.add_argument("--model", default=None, help="YOLO pose 가중치 경로 (기본: 내장 모델)")
    p_cmp.set_defaults(func=cmd_compare_image)

    p_val = sub.add_parser("validate", help="데이터셋 전체 라벨 검증 리포트 생성")
    p_val.add_argument("--metadata", required=True, help="xlsx(Sign_Gloss) 또는 metadata.csv")
    p_val.add_argument("--dataset-root", default=None, help="NAS 마운트 경로 (원본 비디오에서 키프레임 추출용, fallback)")
    p_val.add_argument("--keyframe-images-dir", default=None,
                        help="NAS keyframe_images 폴더 (실제 키프레임 사진, 우선 사용)")
    p_val.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p_val.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    p_val.add_argument("--threshold", type=float, default=0.7)
    p_val.add_argument("--stride", type=int, default=3, help="프레임 스캔 간격(속도용)")
    p_val.add_argument("--limit", type=int, default=0, help="테스트용 상위 N개만 처리")
    p_val.add_argument("--only", default=None, help="쉼표로 구분된 origin_no만 처리")
    p_val.add_argument("--model", default=None, help="YOLO pose 가중치 경로 (기본: 내장 모델)")
    p_val.set_defaults(func=cmd_validate)

    p_gui = sub.add_parser("gui", help="PyQt5 검토 GUI 실행")
    p_gui.set_defaults(func=cmd_gui)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
