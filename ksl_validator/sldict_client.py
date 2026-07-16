"""국립국어원 한국수어사전(sldict.korean.go.kr) 동영상 조회 클라이언트.

사이트는 <video> 태그를 정적 HTML로 내려주지 않고, 페이지 로드 후 JS가
POST /front/sign/include/controlVideoSpeed.do 로 AJAX 요청을 보내
동영상 영역을 채운다. 이 엔드포인트는 origin_no만 있으면 category 없이,
그리고 사전 GET 없이 완전히 새 세션에서도 바로 응답한다 (직접 확인함).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .logging_setup import log

BASE_URL = "https://sldict.korean.go.kr"
VIDEO_AJAX_PATH = "/front/sign/include/controlVideoSpeed.do"
CONTENT_VIEW_PATH = "/front/sign/signContentsView.do"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


class SldictNotFoundError(Exception):
    """origin_no에 해당하는 동영상을 찾지 못했을 때"""


@dataclass
class SldictVideo:
    origin_no: str
    mp4_url: str
    ogv_url: str | None
    webm_url: str | None


def _to_https(url: str) -> str:
    # http:// 소스는 이 사이트에서 응답하지 않는 경우가 있어 https:// 로 강제
    return re.sub(r"^http://", "https://", url)


def fetch_video(session: requests.Session, origin_no: str, category: str = "") -> SldictVideo:
    """origin_no로 동영상 소스 URL들을 조회한다. category는 없어도 동작함."""
    form_data = {
        "origin_no": str(origin_no),
        "category": category,
        "speed": "",
        "size": "high",
        "current_pos_index": "",
        "searchWay": "",
        "searchKeyword": "",
        "pageIndex": "1",
        "top_category": "",
        "refOriginNo": "0",
        "ref_category": "",
    }
    t0 = time.perf_counter()
    resp = session.post(
        f"{BASE_URL}{VIDEO_AJAX_PATH}", data=form_data, headers=HEADERS, timeout=15
    )
    resp.raise_for_status()
    log.debug(f"[sldict] fetch_video origin_no={origin_no}: {time.perf_counter() - t0:.3f}초")

    soup = BeautifulSoup(resp.text, "html.parser")
    sources = {}
    for tag in soup.find_all("source"):
        src = tag.get("src")
        if not src:
            continue
        ext = src.rsplit(".", 1)[-1].lower()
        sources[ext] = _to_https(src)

    if "mp4" not in sources:
        raise SldictNotFoundError(f"origin_no={origin_no}: 동영상 소스를 찾지 못했습니다.")

    return SldictVideo(
        origin_no=str(origin_no),
        mp4_url=sources["mp4"],
        ogv_url=sources.get("ogv"),
        webm_url=sources.get("webm"),
    )


def download_video(
    session: requests.Session,
    video: SldictVideo,
    dest_dir: Path,
    overwrite: bool = False,
) -> Path:
    """mp4를 dest_dir/{origin_no}.mp4 로 다운로드. 이미 있으면 재다운로드 생략."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{video.origin_no}.mp4"

    if dest_path.exists() and not overwrite:
        log.debug(f"[sldict] {video.origin_no}.mp4 이미 캐시됨 - 다운로드 생략")
        return dest_path

    t0 = time.perf_counter()
    resp = session.get(video.mp4_url, headers=HEADERS, stream=True, timeout=30)
    resp.raise_for_status()
    tmp_path = dest_path.with_suffix(".mp4.part")
    size = 0
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            size += len(chunk)
    tmp_path.rename(dest_path)
    elapsed = time.perf_counter() - t0
    log.info(f"[sldict] {video.origin_no}.mp4 다운로드 완료: {size/1024:.0f}KB, {elapsed:.3f}초")
    return dest_path


def fetch_and_download(
    origin_no: str,
    dest_dir: Path,
    category: str = "",
    overwrite: bool = False,
    session: requests.Session | None = None,
) -> Path:
    session = session or requests.Session()
    video = fetch_video(session, origin_no, category)
    return download_video(session, video, dest_dir, overwrite=overwrite)


def fetch_handshape_image_urls(session: requests.Session, origin_no: str) -> list[str]:
    """signContentsView.do의 '수형 사진' 섹션 이미지 URL들을 반환.

    주의: 이 이미지는 실제 사진이 아니라 손모양을 보여주는 손그림 일러스트(선화)다.
    MediaPipe/YOLO로는 검출이 잘 안 되므로 자동 채점에는 쓰지 말고,
    사람이 눈으로 참고하는 용도로만 사용한다.
    """
    t0 = time.perf_counter()
    resp = session.get(
        f"{BASE_URL}{CONTENT_VIEW_PATH}", params={"origin_no": origin_no},
        headers=HEADERS, timeout=15,
    )
    resp.raise_for_status()
    log.debug(f"[sldict] fetch_handshape_image_urls origin_no={origin_no}: {time.perf_counter() - t0:.3f}초")

    soup = BeautifulSoup(resp.text, "html.parser")
    urls: list[str] = []
    for dt in soup.find_all("dt"):
        if dt.get_text(strip=True) != "수형 사진":
            continue
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        for img in dd.find_all("img"):
            src = img.get("src")
            if src:
                urls.append(_to_https(src))
        break
    return urls


def download_handshape_images(
    session: requests.Session, origin_no: str, dest_dir: Path
) -> list[Path]:
    urls = fetch_handshape_image_urls(session, origin_no)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, url in enumerate(urls):
        dest_path = dest_dir / f"{origin_no}_{i}.jpg"
        if not dest_path.exists():
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(resp.content)
        paths.append(dest_path)
    return paths
