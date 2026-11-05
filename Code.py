import json
import math
import os
import queue
import re
import socket
import threading
import time
import unicodedata
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import quote

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, redirect, url_for, request
from paddleocr import PaddleOCR
from pymongo import MongoClient
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from waitress import serve


# ============================================================
# 1. CẤU HÌNH HỆ THỐNG & PHÂN TÁCH CHANNEL
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MONGO_HOST = os.environ.get("MONGO_HOST", "192.168.1.247")
MONGO_PORT = int(os.environ.get("MONGO_PORT", "27017"))
MONGO_USER = os.environ.get("MONGO_USER", "nap_readonly")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD", "8faFV38aGY0BN62a3Hym")
MONGO_DB_NAME = "nap-software"
MONGO_COLLECTION = "warehouse_stock_record"

MONGO_URI = (
    f"mongodb://{quote(MONGO_USER, safe='')}:{quote(MONGO_PASSWORD, safe='')}@"
    f"{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB_NAME}?authSource={MONGO_DB_NAME}"
)

CAMERA_IP = os.environ.get("CAMERA_IP", "192.168.1.140")
CAMERA_USER = os.environ.get("CAMERA_USER", "admin")
CAMERA_PASS = os.environ.get("CAMERA_PASS", "Admin123")
RTSP_PORT = 554

# ---- LUỒNG RTSP CHO OCR (Dùng Channel 101 - Main Stream sắc nét) ----
STREAM_CHANNEL_OCR = os.environ.get("STREAM_CHANNEL_OCR", "101")
RTSP_URL_OCR = (
    f"rtsp://{quote(CAMERA_USER, safe='')}:{quote(CAMERA_PASS, safe='')}@"
    f"{CAMERA_IP}:{RTSP_PORT}/Streaming/Channels/{STREAM_CHANNEL_OCR}"
)

# ---- LUỒNG RTSP CHO WEB PREVIEW (Dùng Channel 102 - Sub Stream nhẹ nhàng) ----
STREAM_CHANNEL_PREVIEW = os.environ.get("STREAM_CHANNEL_PREVIEW", "102")
RTSP_URL_STREAM = (
    f"rtsp://{quote(CAMERA_USER, safe='')}:{quote(CAMERA_PASS, safe='')}@"
    f"{CAMERA_IP}:{RTSP_PORT}/Streaming/Channels/{STREAM_CHANNEL_PREVIEW}"
)

# ---- CẤU HÌNH MEDIAMTX (LOW-LATENCY HLS) ----
MEDIAMTX_HOST = os.environ.get("MEDIAMTX_HOST", "127.0.0.1")
MEDIAMTX_PORT = int(os.environ.get("MEDIAMTX_PORT", "8888"))  # Cổng HLS của MediaMTX
MEDIAMTX_PUBLIC_URL = os.environ.get("MEDIAMTX_PUBLIC_URL", "").strip().rstrip("/")
MEDIAMTX_STREAM_PATH = "warehouse_cam"

HLS_LIVE_SYNC_SEGMENTS = 2
HLS_MAX_LIVE_LATENCY_SEGMENTS = 4

# ---- ĐIỀU KHIỂN ỐNG KÍNH HIKVISION QUA ISAPI (HTTP) ----
CAMERA_HTTP_PORT = int(os.environ.get("CAMERA_HTTP_PORT", "80"))
CAMERA_CHANNEL = os.environ.get("CAMERA_CHANNEL", "1")
CAM_BASE = f"http://{CAMERA_IP}:{CAMERA_HTTP_PORT}"
CAM_HTTP_TIMEOUT = 3.0
CAM_FOCUS_INVERT = os.environ.get("CAM_FOCUS_INVERT", "0") == "1"
CAM_ZOOM_ABS_MIN = int(os.environ.get("CAM_ZOOM_ABS_MIN", "10"))
CAM_ZOOM_ABS_MAX = int(os.environ.get("CAM_ZOOM_ABS_MAX", "43"))
CAM_ZOOM_FULL_SEC = float(os.environ.get("CAM_ZOOM_FULL_SEC", "6.0"))

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
WAITRESS_THREADS = int(os.environ.get("WAITRESS_THREADS", "16"))

# ---- CẤU HÌNH TỐI ƯU TỐC ĐỘ OCR & NHẬN DIỆN PHIÊN (ĐÃ TĂNG TỐC) ----
ROI_X1, ROI_X2 = 0.20, 0.80
ROI_Y1, ROI_Y2 = 0.20, 0.80
OCR_INTERVAL_SEC = 0.05       # Giảm thời gian giữa các lần quét ngầm
MIN_OCR_CONFIDENCE = 0.15
FORM_MIN_CONFIDENCE = 0.10
DETECTION_INTERVAL_SEC = 1.0 / 15.0
OBJECT_STABLE_FRAMES = 2
EMPTY_STABLE_FRAMES = 3
BACKGROUND_WARMUP_FRAMES = 10
MIN_OBJECT_AREA_RATIO = 0.005
REPLACEMENT_DIFF_RATIO = 0.08
REPLACEMENT_STABLE_FRAMES = 2
OCR_MIN_VOTES = 1
OCR_EARLY_STOP_ATTEMPTS = 1   # Dừng sớm để trả kết quả nhanh hơn
OCR_MAX_ATTEMPTS = 2          # Giảm số lần thử tối đa để tránh ì máy
OCR_MIN_SAMPLES = 1
OCR_STABLE_TEXT_SETS = 1
OCR_SINGLE_FRAME_FALLBACK_CONFIDENCE = 0.50
WEB_UI_LEASE_SEC = 5.0

# ---- CẤU HÌNH CAMERA ĐIỆN THOẠI ----
MOBILE_PRESENCE_THR = float(os.environ.get("MOBILE_PRESENCE_THR", "0.05"))  # tỉ lệ điểm ảnh có cạnh trong ROI
MOBILE_EDGE_DIFF = int(os.environ.get("MOBILE_EDGE_DIFF", "40"))
MOBILE_PRESENT_FRAMES = 2      # số frame liên tiếp có vật để bắt đầu quét
MOBILE_EMPTY_FRAMES = 3        # số frame liên tiếp trống để sẵn sàng phiên mới
MOBILE_CHANGE_RATIO = float(os.environ.get("MOBILE_CHANGE_RATIO", "0.30"))  # đổi sản phẩm không qua khoảng trống
MOBILE_OCR_ATTEMPTS = int(os.environ.get("MOBILE_OCR_ATTEMPTS", "2"))
MOBILE_WAIT_FRAME_SEC = 1.5


# ============================================================
# 2. KẾT NỐI MONGODB
# ============================================================
mongo_client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=5000,
)
db_mongo = mongo_client[MONGO_DB_NAME]
warehouse_col = db_mongo[MONGO_COLLECTION]


# ============================================================
# 3. KHỞI TẠO PADDLEOCR 3.x
# ============================================================
print("Đang khởi tạo PaddleOCR 3.x...")
ocr = PaddleOCR(
    use_angle_cls=True,
    ocr_version="PP-OCRv6",
    device="cpu",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)
print("PaddleOCR 3.x sẵn sàng.")


# ============================================================
# 4. TRẠNG THÁI VÀ LOCK HỆ THỐNG
# ============================================================
app = Flask("MainApp")

stop_event = threading.Event()
frame_lock = threading.Lock()
ocr_lock = threading.Lock()
recognition_lock = threading.Lock()
mongo_health_lock = threading.Lock()
cam_lock = threading.Lock()
zoom_move_lock = threading.Lock()
recognition_pause_event = threading.Event()
inventory_search_queue = queue.Queue()
web_ui_last_seen = 0.0

latest_frame = None
latest_frame_id = 0

camera_status = "Đang khởi động camera..."
last_ocr_text = "Đang chờ quét..."
mongo_health = {"ok": False, "message": "Chưa kiểm tra"}
recognition_snapshot = {
    "state": "WAIT_OBJECT",
    "session_id": None,
    "message": "Đang chờ sản phẩm...",
    "dets": [],
    "revision": 0,
    "success": False,
    "sharpness": None,
    "lookup_status": "idle",
    "lookup_text": "",
    "lookup_error": None,
    "matches": [],
    "lookup_revision": 0,
}

# ---- NGUỒN CAMERA: "ip" hoặc "mobile" ----
SOURCE = {"name": "ip"}
source_lock = threading.Lock()
mobile_cv = threading.Condition()
MF = {"frame": None, "seq": 0, "t": 0.0}   # chỉ giữ frame mới nhất từ điện thoại
M = {"state": "IDLE", "present": 0, "empty": 0, "changed": 0, "locked": None}


def get_source():
    with source_lock:
        return SOURCE["name"]


CAM = {
    "motion": 0, "motion_t": 0.0,
    "focus_mode": "UNKNOWN",
    "zoom_pct": 0.0,
    "zoom_abs": None,
}


# ============================================================
# 5. XỬ LÝ ẢNH & THUẬT TOÁN OCR PADDLEOCR 3.X
# ============================================================
def norm(value):
    text = str(value or "").upper().replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^A-Z0-9]", "", text)


def roi_box(frame):
    h, w = frame.shape[:2]
    return (
        int(w * ROI_X1), int(h * ROI_Y1),
        int(w * ROI_X2), int(h * ROI_Y2),
    )


def get_roi(frame):
    x1, y1, x2, y2 = roi_box(frame)
    return frame[y1:y2, x1:x2].copy(), (x1, y1)


def preprocess_roi(roi):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(
        clipLimit=2.5, tileGridSize=(8, 8)
    ).apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )
    soft = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.35, soft, -0.35, 0)


def collect_dets(result, ox, oy):
    dets = []
    if result is None:
        return dets

    res = result[0] if isinstance(result, (list, tuple)) and result else result
    if res is None:
        return dets

    if isinstance(res, list):
        for line in res:
            try:
                if not line or len(line) < 2:
                    continue
                box_pts = line[0]
                text_info = line[1]

                text = str(text_info[0]).strip()
                conf = float(text_info[1])

                if not text or conf < FORM_MIN_CONFIDENCE:
                    continue

                xs = [float(p[0]) for p in box_pts]
                ys = [float(p[1]) for p in box_pts]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                dets.append({
                    "text": text,
                    "conf": round(conf, 3),
                    "box": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
                })
            except Exception:
                continue
        return dets

    if isinstance(res, dict):
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("rec_boxes", None)
        polys = res.get("rec_polys", None)

        for i, text in enumerate(texts):
            try:
                conf = (
                    float(scores[i])
                    if scores is not None and i < len(scores)
                    else 0.2
                )
            except (IndexError, TypeError, ValueError):
                conf = 0.2

            text = str(text).strip()
            if not text or conf < FORM_MIN_CONFIDENCE:
                continue

            try:
                if boxes is not None and i < len(boxes):
                    x1, y1, x2, y2 = [float(v) for v in boxes[i]]
                elif polys is not None and i < len(polys):
                    xs = [float(p[0]) for p in polys[i]]
                    ys = [float(p[1]) for p in polys[i]]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                else:
                    continue
            except (IndexError, TypeError, ValueError):
                continue

            dets.append({
                "text": text,
                "conf": round(conf, 3),
                "box": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
            })
    return dets


def run_ocr(frame):
    roi, (ox, oy) = get_roi(frame)
    processed = preprocess_roi(roi)
    with ocr_lock:
        result = ocr.predict(
            processed,
            text_det_limit_side_len=960,  # Thu nhỏ giới hạn chiều cạnh để OCR chạy nhanh hơn
            text_det_limit_type="max",
        )
    return collect_dets(result, ox, oy)


def analyze_motion(mask, roi_shape):
    _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_area = max(100, int(roi_shape[0] * roi_shape[1] * MIN_OBJECT_AREA_RATIO))
    return any(cv2.contourArea(contour) >= min_area for contour in contours)


def roi_sharpness(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def stabilize_motion_roi(roi, previous_small):
    height, width = roi.shape[:2]
    motion_width = min(640, width)
    motion_height = max(1, round(height * motion_width / width))
    motion_roi = cv2.resize(
        roi, (motion_width, motion_height), interpolation=cv2.INTER_AREA
    )
    small_width = min(320, motion_width)
    small_height = max(1, round(motion_height * small_width / motion_width))
    gray = cv2.cvtColor(motion_roi, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (small_width, small_height),
                       interpolation=cv2.INTER_AREA).astype("float32")

    if previous_small is None or previous_small.shape != small.shape:
        return motion_roi, small, False

    window = cv2.createHanningWindow(
        (small_width, small_height), cv2.CV_32F
    )
    (shift_x, shift_y), response = cv2.phaseCorrelate(
        previous_small, small, window
    )
    if not math.isfinite(shift_x) or not math.isfinite(shift_y):
        return motion_roi, small, False

    if response >= 0.15 and max(abs(shift_x), abs(shift_y)) > 8:
        return motion_roi, small, True

    if response < 0.05 or max(abs(shift_x), abs(shift_y)) < 0.15:
        return motion_roi, small, False

    transform = cv2.getRotationMatrix2D(
        (motion_width / 2.0, motion_height / 2.0), 0.0, 1.0
    )
    transform[0, 2] = -shift_x * motion_width / small_width
    transform[1, 2] = -shift_y * motion_height / small_height
    stabilized = cv2.warpAffine(
        motion_roi, transform, (motion_width, motion_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    stabilized_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
    stabilized_small = cv2.resize(
        stabilized_gray, (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    ).astype("float32")
    return stabilized, stabilized_small, False


def roi_change_ratio(reference, current):
    if reference is None or reference.shape != current.shape:
        return 1.0
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    size = (160, 160)
    reference_small = cv2.resize(reference_gray, size)
    current_small = cv2.resize(current_gray, size)
    reference_small = cv2.GaussianBlur(reference_small, (5, 5), 0)
    current_small = cv2.GaussianBlur(current_small, (5, 5), 0)
    difference = cv2.absdiff(reference_small, current_small)
    _, changed = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(changed) / float(changed.size)


def presence_score(frame):
    """Tỉ lệ điểm ảnh có cạnh rõ trong ROI (thu nhỏ 96x72). Nền trơn -> thấp, nhãn có chữ -> cao."""
    x1, y1, x2, y2 = roi_box(frame)
    g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (96, 72), interpolation=cv2.INTER_AREA).astype(np.int16)
    gx = np.abs(g[:, 1:] - g[:, :-1])[:-1, :]
    gy = np.abs(g[1:, :] - g[:-1, :])[:, :-1]
    return float(((gx + gy) > MOBILE_EDGE_DIFF).mean())


def record_ocr_votes(dets, vote_counts, vote_confidence, vote_detections):
    frame_votes = {}
    for det in dets:
        key = norm(det["text"])
        if key and det["conf"] >= MIN_OCR_CONFIDENCE:
            previous = frame_votes.get(key)
            if previous is None or det["conf"] > previous["conf"]:
                frame_votes[key] = det

    for key, det in frame_votes.items():
        vote_counts[key] = vote_counts.get(key, 0) + 1
        vote_confidence[key] = (
            vote_confidence.get(key, 0.0) + det["conf"]
        )
        vote_detections.setdefault(key, []).append(det)

    if not vote_counts:
        return None
    return max(
        vote_counts,
        key=lambda key: (
            vote_counts[key],
            vote_confidence[key] / vote_counts[key],
        ),
    )


def stable_ocr_detections(vote_counts, vote_detections):
    return [
        max(detections, key=lambda det: det["conf"])
        for key, detections in vote_detections.items()
        if vote_counts.get(key, 0) >= OCR_MIN_VOTES and detections
    ]


def current_ocr_detections(vote_detections):
    return [
        max(detections, key=lambda det: det["conf"])
        for detections in vote_detections.values()
        if detections
    ]


def publish_recognition(state, message, session_id, dets, success=False,
                        sharpness=None):
    global last_ocr_text
    with recognition_lock:
        ui_active = time.time() - web_ui_last_seen <= WEB_UI_LEASE_SEC
        if not ui_active:
            dets = []
            if success:
                message = "Kết quả đã được xóa khi trang web đóng."
                success = False
                recognition_snapshot.update({
                    "lookup_status": "cleared",
                    "lookup_text": "",
                    "lookup_error": None,
                    "matches": [],
                    "lookup_revision": (
                        recognition_snapshot["lookup_revision"] + 1
                    ),
                })
        recognition_snapshot.update({
            "state": state,
            "session_id": session_id,
            "message": message,
            "dets": dets,
            "revision": recognition_snapshot["revision"] + 1,
            "success": success,
            "sharpness": sharpness,
        })
        last_ocr_text = message


def recognition_status():
    with recognition_lock:
        snapshot = dict(recognition_snapshot)
        snapshot["dets"] = [
            dict(det, box=list(det["box"]))
            for det in recognition_snapshot["dets"]
        ]
        snapshot["matches"] = [
            dict(match) for match in recognition_snapshot["matches"]
        ]
    snapshot["paused"] = recognition_pause_event.is_set()
    return snapshot


def web_ui_is_active():
    with recognition_lock:
        return time.time() - web_ui_last_seen <= WEB_UI_LEASE_SEC


def clear_recognition_cache():
    global last_ocr_text
    with recognition_lock:
        recognition_snapshot.update({
            "dets": [],
            "success": False,
            "sharpness": None,
            "lookup_status": "cleared",
            "lookup_text": "",
            "lookup_error": None,
            "matches": [],
            "revision": recognition_snapshot["revision"] + 1,
            "lookup_revision": recognition_snapshot["lookup_revision"] + 1,
        })
        last_ocr_text = "Đang chờ quét..."


def set_mongo_health(ok, message):
    with mongo_health_lock:
        mongo_health.update(ok=ok, message=message)


def get_mongo_health():
    with mongo_health_lock:
        return dict(mongo_health)


# ============================================================
# 6. TÌM SẢN PHẨM TRONG MONGODB
# ============================================================
def load_rows():
    return list(warehouse_col.find(
        {},
        {
            "_id": 0, "code": 1, "productName": 1,
            "remainingQuantity": 1, "locationName": 1, "updatedAt": 1,
            "status": 1, "unit": 1,
        },
    ))


def match_product(text, rows):
    target = norm(text)
    if len(target) < 2:
        return []

    exact_codes = [
        row for row in rows
        if target == norm(row.get("code", ""))
    ]
    if exact_codes:
        return [(1.0, row) for row in exact_codes]

    code_matches = []
    for row in rows:
        code = norm(row.get("code", ""))
        if not code:
            continue
        score = SequenceMatcher(None, target, code).ratio()
        if len(target) >= 3 and (target in code or code in target):
            score = max(score, 0.90)
        if score >= 0.80 and len(target) >= 6:
            code_matches.append((score, row))

    if code_matches:
        code_matches.sort(key=lambda item: item[0], reverse=True)
        return code_matches

    name_matches = []
    for row in rows:
        product_name = norm(row.get("productName", ""))
        if not product_name:
            continue
        score = SequenceMatcher(None, target, product_name).ratio()
        if target == product_name:
            score = 1.0
        elif target in product_name or product_name in target:
            score = max(score, 0.90)
        if score >= 0.65:
            name_matches.append((score, row))

    name_matches.sort(key=lambda item: item[0], reverse=True)
    return name_matches


def search_inventory_candidates(texts):
    rows = load_rows()
    best_matches = {}
    for text in texts:
        for score, row in match_product(text, rows):
            identity = row.get("code") or row.get("productName")
            previous = best_matches.get(identity)
            if previous is None or score > previous[0]:
                best_matches[identity] = (score, row, text)

    ranked = sorted(
        best_matches.values(), key=lambda item: item[0], reverse=True
    )
    results = []
    for score, row, matched_text in ranked[:10]:
        updated_at = row.get("updatedAt", "")
        if isinstance(updated_at, datetime):
            updated_at = updated_at.strftime("%Y-%m-%d %H:%M:%S")
        results.append({
            "product_name": row.get("productName", ""),
            "location": row.get("locationName", ""),
            "updated_at": updated_at,
            "quantity": row.get("remainingQuantity", 0),
            "score": int(round(score * 100)),
            "matched_text": matched_text,
        })
    set_mongo_health(True, "Kết nối OK")
    return results


def search_inventory(text):
    return search_inventory_candidates([text])


def begin_inventory_lookup(session_id, text):
    with recognition_lock:
        if (recognition_snapshot["session_id"] != session_id
                or time.time() - web_ui_last_seen > WEB_UI_LEASE_SEC):
            return None
        recognition_snapshot.update({
            "lookup_status": "searching",
            "lookup_text": text,
            "lookup_error": None,
        })
        recognition_snapshot["lookup_revision"] += 1
        return recognition_snapshot["lookup_revision"]


def finish_inventory_lookup(session_id, lookup_revision, candidates,
                            results=None, error=None):
    with recognition_lock:
        if (recognition_snapshot["session_id"] != session_id
                or recognition_snapshot["lookup_revision"] != lookup_revision
                or time.time() - web_ui_last_seen > WEB_UI_LEASE_SEC):
            return
        matched_dets = [
            dict(candidate["det"], box=list(candidate["det"]["box"]))
            for candidate in candidates
            if candidate.get("det") and candidate["det"].get("box")
        ]
        public_results = [
            {key: value for key, value in match.items()
             if key != "matched_text"}
            for match in results or []
        ]
        recognition_snapshot.update({
            "lookup_status": "error" if error else "complete",
            "lookup_error": error,
            "dets": matched_dets,
            "matches": public_results,
            "revision": recognition_snapshot["revision"] + 1,
            "lookup_revision": recognition_snapshot["lookup_revision"] + 1,
        })


def expire_inactive_web_ui():
    global web_ui_last_seen
    with recognition_lock:
        expired = (
            web_ui_last_seen > 0
            and time.time() - web_ui_last_seen > WEB_UI_LEASE_SEC
        )
        if expired:
            web_ui_last_seen = 0.0
    if expired:
        clear_recognition_cache()


# ============================================================
# 7. ĐIỀU KHIỂN ỐNG KÍNH HIKVISION (ISAPI)
# ============================================================
_http = requests.Session()
_http_lock = threading.Lock()
_AUTHS = {"digest": HTTPDigestAuth(CAMERA_USER, CAMERA_PASS),
          "basic": HTTPBasicAuth(CAMERA_USER, CAMERA_PASS)}
_auth_order = ["digest", "basic"]
XMLNS = 'version="2.0" xmlns="http://www.isapi.org/ver20/XMLSchema"'


def isapi(method, path, xml=None, timeout=CAM_HTTP_TIMEOUT):
    url = CAM_BASE + path
    headers = {"Content-Type": "application/xml"} if xml is not None else {}
    data = xml.encode("utf-8") if xml is not None else None
    last = (401, "Unauthorized")
    with _http_lock:
        for mode in list(_auth_order):
            try:
                r = _http.request(method, url, data=data, headers=headers,
                                  auth=_AUTHS[mode], timeout=timeout)
            except requests.RequestException as exc:
                return 0, str(exc)
            if r.status_code == 401:
                last = (401, r.text)
                continue
            if mode != _auth_order[0]:
                _auth_order.reverse()
            return r.status_code, r.text
    return last


def isapi_msg(code, text):
    if code == 0:
        return f"Không kết nối được camera qua HTTP ({CAM_BASE}): {str(text)[:120]}"
    if code == 401:
        return "Camera từ chối tài khoản/mật khẩu (kiểm tra CAMERA_USER / CAMERA_PASS)."
    m = re.search(r"<statusString>(.*?)</statusString>", text or "", re.S)
    sub = re.search(r"<subStatusCode>(.*?)</subStatusCode>", text or "", re.S)
    detail = " / ".join(x.group(1) for x in (m, sub) if x)
    return f"Camera trả về HTTP {code}" + (f": {detail}" if detail else "")


def cam_set_focus_style(style):
    path = f"/ISAPI/Image/channels/{CAMERA_CHANNEL}/focusConfiguration"
    code, text = isapi("GET", path)
    if code == 200 and "<focusStyle>" in text:
        body = re.sub(r"<\?xml.*?\?>", "", text, flags=re.S).strip()
        xml = re.sub(r"<focusStyle>.*?</focusStyle>",
                     f"<focusStyle>{style}</focusStyle>", body, flags=re.S)
    else:
        xml = (f"<FocusConfiguration {XMLNS}><focusStyle>{style}</focusStyle>"
               f"</FocusConfiguration>")
    code, text = isapi("PUT", path, xml)
    if 200 <= code < 300:
        CAM["focus_mode"] = style
    return code, text


def cam_focus_move(speed):
    speed = max(-100, min(100, int(speed)))
    if CAM_FOCUS_INVERT:
        speed = -speed
    with cam_lock:
        CAM["motion"], CAM["motion_t"] = speed, time.time()
    if speed != 0 and CAM["focus_mode"] != "MANUAL":
        cam_set_focus_style("MANUAL")
    xml = f"<FocusData {XMLNS}><focus>{speed}</focus></FocusData>"
    return isapi("PUT", f"/ISAPI/System/Video/inputs/channels/{CAMERA_CHANNEL}/focus", xml)


def _zoom_continuous(speed):
    xml = f"<PTZData {XMLNS}><pan>0</pan><tilt>0</tilt><zoom>{speed}</zoom></PTZData>"
    return isapi("PUT", f"/ISAPI/PTZCtrl/channels/{CAMERA_CHANNEL}/continuous", xml)


def cam_zoom_to(pct, home=False):
    pct = max(0.0, min(100.0, float(pct)))
    with zoom_move_lock:
        if home:
            CAM["zoom_pct"] = 100.0
        if CAM["zoom_abs"] is not False and not home:
            val = int(round(CAM_ZOOM_ABS_MIN +
                            (CAM_ZOOM_ABS_MAX - CAM_ZOOM_ABS_MIN) * pct / 100.0))
            xml = (f"<PTZData {XMLNS}><AbsoluteHigh><elevation>0</elevation>"
                   f"<azimuth>0</azimuth><absoluteZoom>{val}</absoluteZoom>"
                   f"</AbsoluteHigh></PTZData>")
            code, text = isapi("PUT", f"/ISAPI/PTZCtrl/channels/{CAMERA_CHANNEL}/absolute", xml)
            if 200 <= code < 300:
                CAM["zoom_abs"] = True
                CAM["zoom_pct"] = pct
                return code, text
            if code in (0, 401):
                return code, text
            CAM["zoom_abs"] = False

        delta = pct - CAM["zoom_pct"]
        if abs(delta) < 0.5:
            return 200, ""
        speed = 60 if delta > 0 else -60
        dur = abs(delta) / 100.0 * CAM_ZOOM_FULL_SEC
        code, text = _zoom_continuous(speed)
        if not (200 <= code < 300):
            return code, text
        end = time.time() + dur
        while time.time() < end and not stop_event.is_set():
            time.sleep(0.05)
        code, text = _zoom_continuous(0)
        CAM["zoom_pct"] = pct
        return code, text


def cam_watchdog():
    while not stop_event.is_set():
        time.sleep(0.3)
        with cam_lock:
            stale = CAM["motion"] != 0 and time.time() - CAM["motion_t"] > 1.2
            if stale:
                CAM["motion"] = 0
        if stale:
            try:
                isapi("PUT", f"/ISAPI/System/Video/inputs/channels/{CAMERA_CHANNEL}/focus",
                      f"<FocusData {XMLNS}><focus>0</focus></FocusData>")
            except Exception:
                pass


# ============================================================
# 8. CAMERA WORKER (ĐỌC LUỒNG CHANNEL 101 CHO OCR)
# ============================================================
def camera_worker():
    global latest_frame, latest_frame_id, camera_status
    fail = 0
    while not stop_event.is_set():
        cap = None
        try:
            camera_status = "Đang kết nối luồng OCR camera (Ch. 101)..."
            # Ép dùng TCP và giảm độ trễ tối đa của buffer xuống 0
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;0"
            cap = cv2.VideoCapture(RTSP_URL_OCR, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm bộ đệm phần cứng xuống 1 frame duy nhất

            if not cap.isOpened():
                fail += 1
                wait = min(3 * (2 ** (fail - 1)), 60)
                camera_status = f"Không kết nối được luồng OCR (thử lại sau {wait}s)"
                time.sleep(wait)
                continue

            fail = 0
            camera_status = "Đã kết nối camera (Ch. 101)"
            local_id = 0
            while not stop_event.is_set():
                # Xả sạch các frame cũ đang tồn đọng trong buffer của OpenCV
                for _ in range(3):
                    cap.grab()

                ok, frame = cap.read()
                if not ok or frame is None:
                    camera_status = "Mất kết nối luồng OCR, đang thử lại..."
                    break

                local_id += 1
                with frame_lock:
                    latest_frame = frame  # Ghi đè bằng frame mới nhất
                    latest_frame_id += 1

                time.sleep(0.005)

        except Exception as exc:
            camera_status = f"Lỗi camera: {exc}"
            print("[CAMERA ERROR]", exc)
        finally:
            if cap is not None:
                cap.release()

        if not stop_event.is_set():
            time.sleep(2)


def recognition_worker():
    """Nhận diện cho camera IP. Tự tạm nghỉ khi nguồn là điện thoại."""
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=24, detectShadows=False
    )
    warmup_frames = 0
    processed_frame_id = -1
    state = "WAIT_OBJECT"
    stable_count = 0
    empty_count = 0
    session_id = None
    last_ocr_at = 0.0
    ocr_attempts = 0
    vote_counts = {}
    vote_confidence = {}
    vote_detections = {}
    best_sharpness = -1.0
    best_dets = []
    motion_reference = None
    locked_roi_signature = None
    replacement_count = 0
    last_ui_lease_check = 0.0
    last_motion_at = 0.0
    need_reset = False

    while not stop_event.is_set():
        if get_source() != "ip":
            need_reset = True
            time.sleep(0.1)
            continue
        if need_reset:
            subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=24, detectShadows=False)
            warmup_frames = 0
            motion_reference = None
            state = "WAIT_OBJECT"
            stable_count = empty_count = replacement_count = 0
            session_id = None
            locked_roi_signature = None
            need_reset = False

        now = time.time()
        if now - last_ui_lease_check >= 1.0:
            expire_inactive_web_ui()
            last_ui_lease_check = now

        with frame_lock:
            frame_id = latest_frame_id
            frame = latest_frame

        if frame is None or frame_id == processed_frame_id:
            time.sleep(0.02)
            continue

        now = time.time()
        if now - last_motion_at < DETECTION_INTERVAL_SEC:
            time.sleep(min(DETECTION_INTERVAL_SEC - (now - last_motion_at), 0.02))
            continue
        processed_frame_id = frame_id
        last_motion_at = now

        try:
            roi, _ = get_roi(frame)
            if roi.size == 0:
                time.sleep(0.02)
                continue

            if warmup_frames < BACKGROUND_WARMUP_FRAMES:
                motion_roi, motion_reference, camera_moving = (
                    stabilize_motion_roi(roi, motion_reference)
                )
                if camera_moving:
                    subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=500, varThreshold=24, detectShadows=False
                    )
                    warmup_frames = 0
                    motion_reference = None
                    continue
                subtractor.apply(motion_roi, learningRate=0.05)
                warmup_frames += 1
                if warmup_frames == BACKGROUND_WARMUP_FRAMES:
                    publish_recognition("WAIT_OBJECT", "Đang chờ sản phẩm...", None, [])
                continue

            motion_roi, motion_reference, camera_moving = stabilize_motion_roi(
                roi, motion_reference
            )
            if camera_moving:
                subtractor.apply(motion_roi, learningRate=0.2)
                stable_count = 0
                replacement_count = 0
                continue

            learning_rate = 0.001 if state == "WAIT_OBJECT" else 0.0
            motion_mask = subtractor.apply(
                motion_roi, learningRate=learning_rate
            )
            object_present = analyze_motion(motion_mask, motion_roi.shape)

            if state == "WAIT_OBJECT":
                stable_count = stable_count + 1 if object_present else 0
                if stable_count >= OBJECT_STABLE_FRAMES:
                    state = "RECOGNIZING"
                    session_id = uuid.uuid4().hex
                    last_ocr_at = time.time()
                    ocr_attempts = 0
                    vote_counts = {}
                    vote_confidence = {}
                    vote_detections = {}
                    best_sharpness = -1.0
                    best_dets = []
                    stable_count = 0
                    locked_roi_signature = None
                    replacement_count = 0
                    publish_recognition(
                        state,
                        f"Vật thể ổn định; đang quét toàn bộ OCR · Session {session_id}",
                        session_id,
                        [],
                    )

            elif state == "RECOGNIZING":
                now = time.time()
                if not recognition_pause_event.is_set() and now - last_ocr_at >= OCR_INTERVAL_SEC:
                    last_ocr_at = now
                    ocr_attempts += 1
                    sharpness = roi_sharpness(roi)
                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_dets = []

                    try:
                        dets = run_ocr(frame)
                    except Exception as exc:
                        print(f"[{datetime.now():%H:%M:%S}] OCR ERROR: {exc}")
                        dets = []

                    if sharpness == best_sharpness:
                        best_dets = dets

                    record_ocr_votes(dets, vote_counts, vote_confidence, vote_detections)
                    live_dets = current_ocr_detections(vote_detections)

                    has_consensus = any(
                        votes >= OCR_MIN_VOTES
                        for votes in vote_counts.values()
                    )
                    should_finish = (
                        ocr_attempts >= OCR_MAX_ATTEMPTS
                        or (
                            ocr_attempts >= OCR_EARLY_STOP_ATTEMPTS
                            and has_consensus
                        )
                    )
                    if should_finish:
                        all_scanned_dets = stable_ocr_detections(
                            vote_counts, vote_detections
                        )
                        all_scanned_texts = [
                            det["text"] for det in all_scanned_dets
                        ]

                        if not all_scanned_texts and best_dets:
                            reliable_best_dets = [
                                det for det in best_dets
                                if det["conf"] >= OCR_SINGLE_FRAME_FALLBACK_CONFIDENCE
                            ]
                            all_scanned_dets = reliable_best_dets
                            all_scanned_texts = [
                                det["text"] for det in all_scanned_dets
                            ]

                        candidate_text = " | ".join(all_scanned_texts) if all_scanned_texts else ""

                        state = "WAIT_OBJECT_REMOVAL"
                        empty_count = 0
                        locked_roi_signature = motion_roi.copy()
                        replacement_count = 0

                        publish_recognition(
                            state,
                            (
                                f"Đã quét {ocr_attempts} mẫu; "
                                f"đang đối chiếu dữ liệu kho · "
                                f"nét ROI {best_sharpness:.0f} · Session {session_id}"
                                if all_scanned_texts else
                                f"Chưa đọc được chữ ổn định sau {ocr_attempts} mẫu; "
                                f"nét ROI {best_sharpness:.0f} · "
                                f"chờ lấy hoặc thay sản phẩm · Session {session_id}"
                            ),
                            session_id,
                            all_scanned_dets or best_dets,
                            success=bool(all_scanned_texts),
                            sharpness=best_sharpness,
                        )

                        if all_scanned_dets and web_ui_is_active():
                            lookup_revision = begin_inventory_lookup(session_id, candidate_text)
                            if lookup_revision is not None:
                                lookup_candidates = [
                                    {"text": det["text"], "det": det}
                                    for det in all_scanned_dets
                                ]
                                inventory_search_queue.put((session_id, lookup_revision, lookup_candidates))
                    else:
                        sharpness_label = (
                            f"độ nét ROI {sharpness:.0f}"
                        )
                        scan_detail = (
                            f"đã thấy {len(live_dets)} dòng chữ"
                            if live_dets else "chưa đọc được chữ"
                        )
                        publish_recognition(
                            state,
                            f"Đang quét OCR {ocr_attempts}/{OCR_MAX_ATTEMPTS} · "
                            f"{scan_detail} · {sharpness_label} · "
                            f"Session {session_id}",
                            session_id,
                            live_dets,
                        )

            elif state == "WAIT_OBJECT_REMOVAL":
                empty_count = empty_count + 1 if not object_present else 0
                if empty_count >= EMPTY_STABLE_FRAMES:
                    state = "WAIT_OBJECT"
                    stable_count = 0
                    session_id = None
                    locked_roi_signature = None
                    replacement_count = 0
                    publish_recognition(
                        state, "Đã sẵn sàng cho phiên tiếp theo...",
                        None, [],
                    )
                else:
                    changed_ratio = (
                        roi_change_ratio(locked_roi_signature, motion_roi)
                        if locked_roi_signature is not None else 0.0
                    )
                    replacement_count = (
                        replacement_count + 1
                        if object_present and changed_ratio >= REPLACEMENT_DIFF_RATIO
                        else 0
                    )
                    if replacement_count >= REPLACEMENT_STABLE_FRAMES:
                        state = "RECOGNIZING"
                        session_id = uuid.uuid4().hex
                        last_ocr_at = time.time()
                        ocr_attempts = 0
                        vote_counts = {}
                        vote_confidence = {}
                        vote_detections = {}
                        best_sharpness = -1.0
                        best_dets = []
                        locked_roi_signature = None
                        replacement_count = 0
                        publish_recognition(
                            state,
                            f"Phát hiện sản phẩm mới không có khoảng trống; "
                            f"đang quét · Session {session_id}",
                            session_id,
                            [],
                        )

        except Exception as exc:
            print(f"[{datetime.now():%H:%M:%S}] RECOGNITION ERROR: {exc}")
            publish_recognition(
                state,
                f"Lỗi xử lý camera: {exc}",
                session_id,
                recognition_status()["dets"],
            )
            time.sleep(0.1)

        time.sleep(0.01)


# ============================================================
# 8b. CAMERA ĐIỆN THOẠI: WORKER XỬ LÝ TRÊN SERVER
# ============================================================
def mobile_wait_new_frame(seq_after, timeout):
    end = time.time() + timeout
    with mobile_cv:
        while MF["frame"] is None or MF["seq"] == seq_after:
            left = end - time.time()
            if left <= 0 or stop_event.is_set():
                return None, seq_after
            mobile_cv.wait(left)
        return MF["frame"], MF["seq"]


def mobile_run_session(frame, seq):
    sid = uuid.uuid4().hex
    M["state"] = "SCANNING"
    vote_counts, vote_confidence, vote_detections = {}, {}, {}
    best_sharpness, best_dets = -1.0, []
    attempts = 0
    publish_recognition("RECOGNIZING",
                        f"Điện thoại: vật thể ổn định, đang quét OCR · Session {sid}",
                        sid, [])

    while frame is not None and attempts < MOBILE_OCR_ATTEMPTS and not stop_event.is_set():
        if get_source() != "mobile" or recognition_pause_event.is_set():
            M["state"] = "IDLE"
            return
        roi, _ = get_roi(frame)
        sharp = roi_sharpness(roi)
        try:
            dets = run_ocr(frame)
        except Exception as exc:
            print(f"[{datetime.now():%H:%M:%S}] MOBILE OCR ERROR: {exc}")
            dets = []
        attempts += 1
        if sharp > best_sharpness:
            best_sharpness, best_dets = sharp, dets
        record_ocr_votes(dets, vote_counts, vote_confidence, vote_detections)
        live = current_ocr_detections(vote_detections)
        publish_recognition(
            "RECOGNIZING",
            f"Đang quét OCR {attempts}/{MOBILE_OCR_ATTEMPTS} · đã thấy {len(live)} dòng chữ · "
            f"độ nét ROI {sharp:.0f} · Session {sid}",
            sid, live)
        if attempts >= MOBILE_OCR_ATTEMPTS:
            break
        frame, seq = mobile_wait_new_frame(seq, MOBILE_WAIT_FRAME_SEC)

    if get_source() != "mobile":
        return

    all_dets = stable_ocr_detections(vote_counts, vote_detections)
    texts = [d["text"] for d in all_dets]
    if not texts and best_dets:
        all_dets = [d for d in best_dets if d["conf"] >= OCR_SINGLE_FRAME_FALLBACK_CONFIDENCE]
        texts = [d["text"] for d in all_dets]

    with mobile_cv:
        last = MF["frame"]
    if last is not None:
        M["locked"] = get_roi(last)[0]
    M.update(state="DONE", empty=0, changed=0, present=0)

    publish_recognition(
        "WAIT_OBJECT_REMOVAL",
        (f"Đã quét {attempts} mẫu; đang đối chiếu dữ liệu kho · nét ROI {best_sharpness:.0f} · Session {sid}"
         if texts else
         f"Chưa đọc được chữ ổn định sau {attempts} mẫu; nét ROI {best_sharpness:.0f} · "
         f"chờ lấy hoặc thay sản phẩm · Session {sid}"),
        sid, all_dets or best_dets, success=bool(texts), sharpness=best_sharpness)

    if all_dets and web_ui_is_active():
        rev = begin_inventory_lookup(sid, " | ".join(texts))
        if rev is not None:
            inventory_search_queue.put(
                (sid, rev, [{"text": d["text"], "det": d} for d in all_dets]))


def mobile_process_frame(frame, seq):
    roi, _ = get_roi(frame)
    if roi.size == 0:
        return
    present = presence_score(frame) >= MOBILE_PRESENCE_THR

    if M["state"] == "IDLE":
        M["present"] = M["present"] + 1 if present else 0
        if M["present"] >= MOBILE_PRESENT_FRAMES:
            M["present"] = 0
            mobile_run_session(frame, seq)

    elif M["state"] == "DONE":
        if not present:
            M["empty"] += 1
            M["changed"] = 0
            if M["empty"] >= MOBILE_EMPTY_FRAMES:
                M.update(state="IDLE", empty=0, present=0, locked=None)
                publish_recognition("WAIT_OBJECT", "Đã sẵn sàng cho phiên tiếp theo...", None, [])
        else:
            M["empty"] = 0
            locked = M["locked"]
            ratio = roi_change_ratio(locked, roi) if locked is not None else 0.0
            M["changed"] = M["changed"] + 1 if ratio >= MOBILE_CHANGE_RATIO else 0
            if M["changed"] >= 2:      # đổi sản phẩm không có khoảng trống
                M["changed"] = 0
                mobile_run_session(frame, seq)


def mobile_worker():
    last_seq = 0
    while not stop_event.is_set():
        with mobile_cv:
            while not stop_event.is_set() and (
                    MF["frame"] is None or MF["seq"] == last_seq):
                mobile_cv.wait(0.5)
            if stop_event.is_set():
                break
            frame, last_seq = MF["frame"], MF["seq"]
        if get_source() != "mobile" or recognition_pause_event.is_set():
            continue
        try:
            mobile_process_frame(frame, last_seq)
        except Exception as exc:
            print(f"[{datetime.now():%H:%M:%S}] MOBILE ERROR: {exc}")
            M["state"] = "IDLE"


def inventory_search_worker():
    while not stop_event.is_set():
        try:
            session_id, lookup_revision, candidates = (
                inventory_search_queue.get(timeout=0.25)
            )
        except queue.Empty:
            continue

        try:
            results = search_inventory_candidates([
                candidate["text"] for candidate in candidates
            ])
            finish_inventory_lookup(
                session_id,
                lookup_revision,
                candidates,
                results=results,
            )
            print(
                f"[MONGODB SEARCH] Session {session_id}: "
                f"{len(results)} kết quả cho {len(candidates)} nội dung OCR"
            )
        except Exception as exc:
            set_mongo_health(False, f"Lỗi truy vấn MongoDB: {exc}")
            finish_inventory_lookup(
                session_id,
                lookup_revision,
                candidates,
                error=str(exc),
            )
            print(f"[MONGODB SEARCH ERROR] Session {session_id}: {exc}")
        finally:
            inventory_search_queue.task_done()


# ============================================================
# 9. GIAO DIỆN WEB
# ============================================================
PICK_HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hệ thống nhận diện sản phẩm</title>
<script src="https://cdn.jsdelivr.net/npm/hls.js@1.4.12/dist/hls.min.js"></script>
<style>
:root{
  --bg:#12151a;--panel:#1b2028;--panel2:#232a35;--line:#2f3846;
  --text:#e8edf4;--muted:#8d99ab;--accent:#19d3a2;--accent2:#ffb02e;
  --info:#4cc9f0;--danger:#ff5d6c;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font-family:"Segoe UI",Arial,sans-serif;padding:14px;
max-width:1500px;margin-inline:auto}
.top{display:flex;justify-content:space-between;align-items:center;
flex-wrap:wrap;gap:12px;margin-bottom:14px}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:42px;height:42px;border-radius:10px;
background:linear-gradient(135deg,var(--accent),var(--info));
display:grid;place-items:center;color:#06231c;font-weight:800;font-size:18px}
.brand h1{font-size:19px;margin:0;letter-spacing:.4px}
.brand p{margin:2px 0 0;font-size:12px;color:var(--muted)}
.pills{display:flex;gap:8px;flex-wrap:wrap}
.pill{background:var(--panel);border:1px solid var(--line);border-radius:999px;
padding:6px 13px;font-size:13px;display:flex;align-items:center;gap:7px}
.pill::before{content:"";width:9px;height:9px;border-radius:50%;background:var(--muted)}
.pill.ok::before{background:var(--accent);box-shadow:0 0 8px var(--accent)}
.pill.bad::before{background:var(--danger);box-shadow:0 0 8px var(--danger)}
.pill.warn::before{background:var(--accent2)}
.grid{display:grid;grid-template-columns:minmax(0,2.4fr) minmax(320px,1fr);gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
.card h3{margin:0 0 12px;font-size:13px;letter-spacing:1px;color:var(--accent);
text-transform:uppercase;display:flex;justify-content:space-between;align-items:center}
.camera-wrap{position:relative;background:#000;border-radius:10px;overflow:hidden;aspect-ratio:16/9}
#hls_stream,#phone_cam{position:absolute;inset:0;width:100%;height:100%;
object-fit:contain;display:block}
#roi_overlay{position:absolute;border:3px solid #00ff66;pointer-events:none;
box-shadow:0 0 5px rgba(0,0,0,.8);z-index:5;display:none}
.badge{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.65);
padding:5px 12px;border-radius:999px;font-size:12px;font-weight:700;line-height:1.3;z-index:10}
.side{display:flex;flex-direction:column;gap:14px}
.ctl{margin-bottom:14px}
.ctl:last-child{margin-bottom:0}
.ctl .lbl{display:flex;justify-content:space-between;font-size:13px;color:var(--muted);margin-bottom:7px}
.ctl .lbl b{color:var(--text)}
input[type=range]{width:100%;accent-color:var(--accent);height:26px}
.row{display:flex;gap:8px;flex-wrap:wrap}
.row .grow{flex:1}
button{background:var(--panel2);color:var(--text);border:1px solid var(--line);
border-radius:8px;padding:10px 12px;font-size:14px;font-weight:600;cursor:pointer;
touch-action:manipulation;user-select:none}
button:hover{border-color:var(--accent)}
button:active{background:#2d3646}
button.pri{background:var(--accent);color:#06231c;border-color:var(--accent)}
button.pri:hover{filter:brightness(1.08)}
button.sm{padding:6px 10px;font-size:13px}
select{background:var(--panel2);color:var(--text);border:1px solid var(--line);
border-radius:8px;padding:7px 8px;font-size:13px}
.msg{font-size:12px;color:var(--accent2);min-height:16px;margin-top:8px}
.recognition-notice{margin:0 0 12px;padding:14px 16px;border:1px solid #8a3842;
border-left:5px solid var(--danger);border-radius:8px;background:#351f25;
color:#ffd9dd;font-size:18px;font-weight:700;line-height:1.45}
.recognition-notice[hidden]{display:none}
.chips{display:flex;flex-wrap:wrap;gap:7px;max-height:170px;overflow:auto}
.chip{background:var(--panel2);border:1px solid var(--line);border-radius:999px;
padding:5px 11px;font-size:13px;cursor:pointer}
.chip small{color:var(--muted);margin-left:5px}
.chip:hover,.chip.sel{border-color:var(--accent2);color:var(--accent2)}
.hint{font-size:12px;color:var(--muted);margin-top:8px;line-height:1.5}
.result{grid-column:1/-1}
.best{display:none;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:14px}
.best.show{display:grid}
.kpi{background:var(--panel2);border-left:4px solid var(--accent);border-radius:8px;padding:10px 14px}
.kpi span{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}
.kpi b{display:block;font-size:19px;margin-top:4px;word-break:break-word}
.kpi.loc{border-color:var(--danger)}.kpi.loc b{color:#ff8f9a}
.kpi.qty{border-color:var(--accent2)}.kpi.qty b{color:#ffb02e}
.kpi.nm{border-color:var(--info)}.kpi.nm b{color:#4cc9f0}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:620px}
th,td{border-bottom:1px solid var(--line);padding:10px;text-align:left;font-size:14px}
th{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.7px}
.empty{text-align:center;color:var(--muted);padding:22px}
#selected{color:var(--accent);font-size:13px;margin-bottom:10px}
#selected.warning{color:var(--danger);font-weight:700}
#src_ip.pri,#src_mobile.pri{background:var(--accent);color:#06231c;border-color:var(--accent)}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header class="top">
  <div class="brand">
    <div class="logo">OCR</div>
    <div>
      <h1>HỆ THỐNG NHẬN DIỆN SẢN PHẨM</h1>
      <p>Hikvision DS-2CD2683G3-LIZS2UY · MediaMTX Low-Latency HLS (Ch. 102) · Camera điện thoại · MongoDB</p>
    </div>
  </div>
  <div class="pills">
    <span class="pill" id="p_cam">Camera: ...</span>
    <span class="pill" id="p_preview">MediaMTX: đang kết nối...</span>
    <span class="pill" id="p_db">Kho: ...</span>
    <span class="pill" id="p_ocr">Nhận diện: ...</span>
    <span class="pill" id="p_focus">Nét: --</span>
  </div>
</header>

<main class="grid">
  <section class="card">
    <h3>Hình ảnh trực tiếp <span id="ocr_text" style="color:var(--muted);text-transform:none;letter-spacing:0;font-size:12px"></span></h3>
    <div class="row" style="margin-bottom:10px">
      <button class="sm grow pri" id="src_ip">📷 CAMERA IP</button>
      <button class="sm grow" id="src_mobile">📱 CAMERA ĐIỆN THOẠI</button>
      <button class="sm" id="btn_rescan" style="display:none">QUÉT LẠI</button>
    </div>
    <div class="camera-wrap">
      <video id="hls_stream" autoplay playsinline muted></video>
      <video id="phone_cam" autoplay playsinline muted style="display:none"></video>
      <div id="roi_overlay" aria-hidden="true"></div>
      <div class="badge" id="badge">ĐANG NHẬN DIỆN</div>
    </div>
  </section>

  <aside class="side">
    <section class="card" id="lens_card">
      <h3>Điều khiển ống kính</h3>

      <div class="ctl">
        <div class="lbl"><span>Zoom (Wide ⟷ Tele)</span><b id="zoom_val">0%</b></div>
        <input type="range" id="zoom_s" min="0" max="100" step="1" value="0">
        <div class="row" style="margin-top:8px">
          <button class="sm grow" id="z_minus">− THU NHỎ</button>
          <button class="sm grow" id="z_plus">PHÓNG TO +</button>
          <button class="sm" id="z_home" title="Chạy hết về Wide để đồng bộ lại">VỀ WIDE</button>
        </div>
      </div>

      <div class="ctl">
        <div class="lbl"><span>Lấy nét</span>
          <span>Bước:
            <select id="f_speed">
              <option value="20">Mịn</option>
              <option value="40" selected>Vừa</option>
              <option value="70">Nhanh</option>
            </select>
          </span>
        </div>
        <div class="row">
          <button class="grow" id="f_minus">NÉT −</button>
          <button class="grow" id="f_plus">NÉT +</button>
          <button class="grow pri" id="f_auto">AUTO NÉT</button>
        </div>
        <div class="hint">Giữ nút NÉT −/+ để chỉnh liên tục, thả tay là dừng. Bấm AUTO để camera tự bắt nét.</div>
      </div>
      <div class="msg" id="cam_msg"></div>
    </section>

    <section class="card">
      <h3>Chữ nhận diện được
        <button class="sm" id="btn_pause">TẠM DỪNG</button></h3>
      <div class="recognition-notice" id="recognition_notice"
           role="status" aria-live="polite" hidden></div>
      <div class="chips" id="chips"><span class="hint">Chưa có dữ liệu...</span></div>
      <div class="hint">Các dòng chữ OCR sẽ hiện tại đây trong lúc quét; sau đó hệ thống tự đối chiếu với kho.</div>
    </section>
  </aside>

  <section class="card result">
    <h3>Kết quả tra cứu kho</h3>
    <div id="selected">Chưa chọn nội dung OCR</div>
    <div class="best" id="best">
      <div class="kpi nm"><span>Sản phẩm</span><b id="b_name">--</b></div>
      <div class="kpi loc"><span>Vị trí kệ</span><b id="b_loc">--</b></div>
      <div class="kpi qty"><span>Tồn kho</span><b id="b_qty">--</b></div>
      <div class="kpi"><span>Cập nhật</span><b id="b_upd" style="font-size:15px">--</b></div>
    </div>
    <div class="tablewrap">
      <table>
        <thead><tr>
          <th>Tên sản phẩm</th><th>Vị trí kệ</th>
          <th>Ngày cập nhật</th><th>Số lượng còn</th><th>Độ khớp</th>
        </tr></thead>
        <tbody id="results">
          <tr><td colspan="5" class="empty">Đang chờ nhận diện...</td></tr>
        </tbody>
      </table>
    </div>
  </section>
</main>

<script>
const $=id=>document.getElementById(id);
let detections=[],busy=false,selectedIndex=-1;
let paused=false,pauseUntil=0,lastRevision=-1,lookupRevision=-1;
let serverPaused=false,pauseTimer=null;
let recognitionNoticeText="";

const configuredMediaMtxUrl=__MEDIAMTX_PUBLIC_URL__;
const mediaMtxBaseUrl=configuredMediaMtxUrl||
  (window.location.protocol+"//"+window.location.hostname+":__MEDIAMTX_PORT__");
const hlsUrl=mediaMtxBaseUrl+"/__MEDIAMTX_STREAM_PATH__/index.m3u8";
const videoElement=$("hls_stream");
const roiOverlay=$("roi_overlay");

let currentSource="ip",activeVideo=videoElement,mediaStream=null,mobileGen=0,wakeLock=null;
const phoneVideo=$("phone_cam");
const MOBILE_FRAME_W=1280,MOBILE_JPEG_Q=0.8,MOBILE_IDLE_MS=400,MOBILE_SCAN_MS=250;
let hlsPlayer=null;

function updateRoiOverlay(){
  const v=activeVideo;
  const videoWidth=v.videoWidth,videoHeight=v.videoHeight;
  const container=v.parentElement;
  if(!videoWidth||!videoHeight||!container)return;
  const scale=Math.min(container.clientWidth/videoWidth,container.clientHeight/videoHeight);
  const dw=videoWidth*scale,dh=videoHeight*scale;
  roiOverlay.style.left=((container.clientWidth-dw)/2+dw*0.2)+"px";
  roiOverlay.style.top=((container.clientHeight-dh)/2+dh*0.2)+"px";
  roiOverlay.style.width=(dw*0.6)+"px";
  roiOverlay.style.height=(dh*0.6)+"px";
  roiOverlay.style.display="block";
}
videoElement.addEventListener("loadedmetadata",updateRoiOverlay);
phoneVideo.addEventListener("loadedmetadata",updateRoiOverlay);
phoneVideo.addEventListener("playing",updateRoiOverlay);
window.addEventListener("resize",updateRoiOverlay);

function setPreviewStatus(text,ok){
  const pill=$("p_preview");
  pill.textContent=text;
  pill.className="pill "+(ok?"ok":"bad");
}
videoElement.addEventListener("playing",()=>{
  if(currentSource!=="ip")return;
  const resolution=videoElement.videoWidth?
    " · "+videoElement.videoWidth+"×"+videoElement.videoHeight:"";
  setPreviewStatus("MediaMTX: phát trực tiếp"+resolution,true);
  updateRoiOverlay();
});
videoElement.addEventListener("waiting",()=>{
  if(currentSource!=="ip")return;
  setPreviewStatus("MediaMTX: đang chờ frame...",false);
});
videoElement.addEventListener("error",()=>{
  if(currentSource!=="ip")return;
  setPreviewStatus("MediaMTX: lỗi phát video",false);
});

if(window.Hls&&Hls.isSupported()){
  const hls=new Hls({
    lowLatencyMode: true,
    liveSyncDurationCount: 1,
    liveMaxLatencyDurationCount: 2,
    maxLiveSyncPlaybackRate: 1.1,
    maxBufferLength: 1,
    backBufferLength: 0,
    enableWorker: true,
  });
  hlsPlayer=hls;
  hls.loadSource(hlsUrl);
  hls.attachMedia(videoElement);
  hls.on(Hls.Events.MANIFEST_PARSED,()=>{
    videoElement.play().catch(error=>{
      setPreviewStatus("MediaMTX: không tự phát được video",false);
      console.error("HLS autoplay failed:",error);
    });
  });
  hls.on(Hls.Events.ERROR,(_event,data)=>{
    if(!data.fatal)return;
    setPreviewStatus("MediaMTX: đang kết nối lại...",false);
    if(data.type===Hls.ErrorTypes.NETWORK_ERROR){
      hls.startLoad();
    }else if(data.type===Hls.ErrorTypes.MEDIA_ERROR){
      hls.recoverMediaError();
    }else{
      hls.destroy();
    }
  });
}else if(videoElement.canPlayType("application/vnd.apple.mpegurl")){
  videoElement.src=hlsUrl;
  videoElement.play().catch(error=>{
    setPreviewStatus("MediaMTX: không tự phát được video",false);
    console.error("Native HLS autoplay failed:",error);
  });
}else{
  setPreviewStatus("Trình duyệt không hỗ trợ phát HLS",false);
}

async function heartbeatWebUI(){
  try{
    const r=await fetch("/api/pick/heartbeat",{method:"POST",cache:"no-store"});
    if(!r.ok)throw new Error("Không gia hạn được phiên web.");
  }catch(e){console.error("Web UI heartbeat failed:",e);}
}
heartbeatWebUI();setInterval(heartbeatWebUI,1500);
window.addEventListener("pagehide",()=>{
  const body=new Blob([],{type:"text/plain"});
  if(!navigator.sendBeacon("/api/pick/clear",body)){
    fetch("/api/pick/clear",{method:"POST",body,keepalive:true});
  }
});

function clearDetections(){
  detections=[];selectedIndex=-1;
  renderChips();
}
async function syncRecognitionPause(){
  const shouldPause=paused||Date.now()<pauseUntil;
  if(shouldPause===serverPaused)return;
  serverPaused=shouldPause;
  try{
    const r=await fetch("/api/pick/control",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({paused:shouldPause})});
    if(!r.ok)throw new Error("Không cập nhật được trạng thái OCR.");
    if(!shouldPause){lastRevision=-1;scanOCR();}
  }catch(e){
    serverPaused=!shouldPause;
    $("ocr_text").textContent="Lỗi điều khiển OCR: "+e;
  }
}
function pauseScan(ms){
  pauseUntil=Math.max(pauseUntil,Date.now()+ms);clearDetections();
  syncRecognitionPause();
  clearTimeout(pauseTimer);
  pauseTimer=setTimeout(syncRecognitionPause,Math.max(0,pauseUntil-Date.now())+100);
}

function renderChips(){
  const box=$("chips");box.innerHTML="";
  if(!detections.length){
    box.innerHTML='<span class="hint">Chưa nhận diện được chữ nào...</span>';return;}
  detections.forEach((d,i)=>{
    const c=document.createElement("span");
    c.className="chip"+(i===selectedIndex?" sel":"");
    c.textContent=d.text;
    const s=document.createElement("small");s.textContent=Number(d.conf).toFixed(2);
    c.appendChild(s);c.onclick=()=>selectDet(i);box.appendChild(c);
  });
}

function renderRecognitionNotice(data){
  const notice=$("recognition_notice");
  const shouldShow=data.state==="WAIT_OBJECT_REMOVAL"&&!data.success&&
    data.lookup_status==="idle";
  const text=shouldShow?(data.ocr||"Không đọc được nội dung ổn định; chờ lấy sản phẩm khỏi ROI."):"";
  if(text===recognitionNoticeText)return;
  recognitionNoticeText=text;
  notice.textContent=text;
  notice.hidden=!shouldShow;
}

async function scanOCR(){
  if(busy||paused||Date.now()<pauseUntil)return;
  busy=true;
  try{
    const r=await fetch("/api/pick/capture",{method:"POST",cache:"no-store"});
    const data=await r.json();
    if(!r.ok){$("ocr_text").textContent=data.message||"Không lấy được hình ảnh.";return;}
    if(Date.now()<pauseUntil)return;
    renderRecognitionNotice(data);
    const revisionChanged=data.revision!==lastRevision;
    if(revisionChanged){
      lastRevision=data.revision;
      detections=data.dets||[];selectedIndex=-1;
      renderChips();
    }
    if(data.lookup_revision!==lookupRevision){
      lookupRevision=data.lookup_revision;
      if(data.lookup_status==="complete"){
        renderResults(data.matches||[]);
      }else if(data.lookup_status==="error"){
        $("selected").textContent="Lỗi tra cứu MongoDB: "+(data.lookup_error||"Không rõ lỗi");
        $("selected").classList.add("warning");
        $("results").innerHTML='<tr><td colspan="5" class="empty">Không truy vấn được kho. Kiểm tra trạng thái MongoDB phía trên.</td></tr>';
        setBest(null);
      }else if(data.lookup_status==="idle"&&data.state==="WAIT_OBJECT_REMOVAL"&&!data.success){
        $("selected").textContent="Không đọc được nội dung sản phẩm ổn định.";
        $("selected").classList.add("warning");
        $("results").innerHTML='<tr><td colspan="5" class="empty">Thử đặt sản phẩm ổn định trong vùng quét.</td></tr>';
        setBest(null);
      }else if(data.lookup_status==="cleared"){
        $("selected").textContent="Kết quả phiên trước đã được xóa.";
        $("selected").classList.remove("warning");
        $("results").innerHTML='<tr><td colspan="5" class="empty">Đang chờ phiên nhận diện mới...</td></tr>';
        setBest(null);
      }
    }
    $("ocr_text").textContent=data.ocr||"Đang chờ sản phẩm...";
    $("badge").textContent=data.paused?"ĐÃ TẠM DỪNG":
      data.state==="RECOGNIZING"?"ĐANG NHẬN DIỆN":
      data.state==="WAIT_OBJECT_REMOVAL"?"CHỜ LẤY SẢN PHẨM":"ĐANG CHỜ SẢN PHẨM";
  }catch(e){$("ocr_text").textContent="Lỗi OCR: "+e;}
  finally{busy=false;}
}
scanOCR();setInterval(scanOCR,300);

$("btn_pause").onclick=()=>{
  paused=!paused;
  $("btn_pause").textContent=paused?"TIẾP TỤC":"TẠM DỪNG";
  $("badge").textContent=paused?"ĐÃ TẠM DỪNG":"ĐANG NHẬN DIỆN";
  if(paused)clearDetections();
  syncRecognitionPause();
  if(!paused){lastRevision=-1;scanOCR();}
};

function setBest(p){
  const b=$("best");
  if(!p){b.classList.remove("show");return;}
  $("b_name").textContent=p.product_name||"--";
  $("b_loc").textContent=p.location||"--";
  $("b_qty").textContent=(p.quantity==null?"--":p.quantity);
  $("b_upd").textContent=p.updated_at||"--";
  b.classList.add("show");
}
function renderResults(rows){
  const tb=$("results");tb.innerHTML="";
  if(!rows.length){
    $("selected").textContent="CẢNH BÁO: Sản phẩm chưa có trong kho.";
    $("selected").classList.add("warning");
    tb.innerHTML='<tr><td colspan="5" class="empty">Sản phẩm chưa có trong kho.</td></tr>';
    setBest(null);return;}
  $("selected").textContent="Đã tìm thấy sản phẩm trong kho.";
  $("selected").classList.remove("warning");
  setBest(rows[0]);
  rows.forEach(p=>{
    const tr=document.createElement("tr");
    [p.product_name,p.location,p.updated_at,p.quantity,(p.score!=null?p.score+"%":"")].forEach(v=>{
      const td=document.createElement("td");td.textContent=v??"";tr.appendChild(td);});
    tb.appendChild(tr);
  });
}
async function selectDet(i){
  selectedIndex=i;renderChips();
  const text=detections[i].text;
  $("selected").textContent="Đang tra cứu: "+text;
  $("results").innerHTML='<tr><td colspan="5" class="empty">Đang tìm trong kho...</td></tr>';
  try{
    const r=await fetch("/api/pick/search",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({text})});
    const data=await r.json();
    if(!r.ok){$("selected").textContent=data.message||"Lỗi tra cứu";return;}
    renderResults(data.results||[]);
  }catch(e){$("selected").textContent="Lỗi tra cứu: "+e;}
}

function camMsg(t){$("cam_msg").textContent=t||"";}

async function zoomSend(pct,home){
  pauseScan(3500);camMsg("Đang zoom...");
  try{
    const r=await fetch("/api/cam_zoom_to",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({pct:pct,home:!!home})});
    const d=await r.json();camMsg(d.ok?"":("Camera: "+(d.message||"lỗi")));
    pauseScan(1500);
  }catch(e){camMsg("Camera: "+e);}
}
const zs=$("zoom_s");
zs.addEventListener("input",()=>{$("zoom_val").textContent=zs.value+"%";});
zs.addEventListener("change",()=>zoomSend(+zs.value,false));
function zoomStep(d){
  zs.value=Math.max(0,Math.min(100,(+zs.value)+d));
  $("zoom_val").textContent=zs.value+"%";zoomSend(+zs.value,false);}
$("z_minus").onclick=()=>zoomStep(-10);
$("z_plus").onclick=()=>zoomStep(10);
$("z_home").onclick=()=>{zs.value=0;$("zoom_val").textContent="0%";zoomSend(0,true);};

const FOCUS={t:null,v:0,start:0};
async function focusSend(speed){
  try{
    const r=await fetch("/api/cam/focus",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({speed:speed})});
    const d=await r.json();camMsg(d.ok?"":("Camera: "+(d.message||"lỗi")));
  }catch(e){camMsg("Camera: "+e);}
}
function focusStart(sign){
  FOCUS.v=sign*(+$("f_speed").value);pauseScan(2500);
  if(!FOCUS.t){FOCUS.start=Date.now();focusSend(FOCUS.v);
    FOCUS.t=setInterval(()=>{pauseScan(2500);focusSend(FOCUS.v);},350);}
}
function focusEnd(){
  if(!FOCUS.t)return;
  clearInterval(FOCUS.t);FOCUS.t=null;FOCUS.v=0;
  const wait=Math.max(0,250-(Date.now()-FOCUS.start));
  setTimeout(()=>{focusSend(0);pauseScan(1500);},wait);
}
function bindHold(id,sign){
  const b=$(id);
  b.addEventListener("pointerdown",e=>{e.preventDefault();focusStart(sign);});
  ["pointerup","pointerleave","pointercancel"].forEach(ev=>b.addEventListener(ev,focusEnd));
}
bindHold("f_minus",-1);bindHold("f_plus",1);
window.addEventListener("blur",focusEnd);

$("f_auto").onclick=async()=>{
  pauseScan(3000);camMsg("Đang bật lấy nét tự động...");
  try{
    const r=await fetch("/api/cam_focus_mode",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({mode:"auto"})});
    const d=await r.json();camMsg(d.message||(d.ok?"Đã bật AUTO NÉT.":"Lỗi"));
    pauseScan(2000);health();
  }catch(e){camMsg("Camera: "+e);}
};

function setPill(id,text,cls){const e=$(id);e.textContent=text;e.className="pill "+cls;}
async function health(){
  try{
    const r=await fetch("/api/health",{cache:"no-store"});const d=await r.json();
    const okCam=(d.camera||"").startsWith("Đã kết nối");
    const mob=currentSource==="mobile";
    setPill("p_cam",mob?"Camera: điện thoại":"Camera: "+(d.camera||"--"),mob||okCam?"ok":"bad");
    setPill("p_preview",mob?"Video: điện thoại → server":"Video: "+(d.preview_message||"chưa rõ trạng thái"),
      mob||d.preview_ok?"ok":"bad");
    setPill("p_db","Kho: "+(d.database_message||"Chưa kiểm tra"),
      d.database_ok?"ok":"bad");
    const ocrState=d.recognition_state;
    const ocrLabel=paused||d.recognition_paused?"Nhận diện: tạm dừng":
      ocrState==="RECOGNIZING"?"Đang OCR":
      ocrState==="WAIT_OBJECT_REMOVAL"?"Chờ lấy sản phẩm":"Chờ sản phẩm";
    setPill("p_ocr",ocrLabel,paused||d.recognition_paused?"warn":"ok");
    const fm=d.focus_mode;
    setPill("p_focus","Nét: "+(fm==="AUTO"?"tự động":fm==="MANUAL"?"thủ công":"--"),fm==="AUTO"?"ok":(fm==="MANUAL"?"warn":""));
  }catch(e){setPill("p_cam","Mất kết nối máy chủ","bad");}
}
health();setInterval(health,4000);

// ============ CAMERA ĐIỆN THOẠI ============
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const fcv=document.createElement("canvas"),fctx=fcv.getContext("2d");

function setSourceUI(src){
  const mob=src==="mobile";
  $("src_ip").classList.toggle("pri",!mob);
  $("src_mobile").classList.toggle("pri",mob);
  videoElement.style.display=mob?"none":"block";
  phoneVideo.style.display=mob?"block":"none";
  $("btn_rescan").style.display=mob?"":"none";
  $("lens_card").style.display=mob?"none":"";
  roiOverlay.style.display="none";
  activeVideo=mob?phoneVideo:videoElement;
  updateRoiOverlay();
}

async function postSource(src){
  const r=await fetch("/api/source",{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify({source:src})});
  if(!r.ok)throw new Error("Server từ chối đổi nguồn");
}

function stopPhoneCamera(){
  mobileGen++;
  if(mediaStream){mediaStream.getTracks().forEach(t=>t.stop());mediaStream=null;}
  phoneVideo.srcObject=null;
  if(wakeLock){wakeLock.release().catch(()=>{});wakeLock=null;}
}

async function switchSource(src){
  if(src===currentSource)return;
  if(src==="mobile"){
    if(!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia)){
      alert("Trình duyệt chặn camera. Hãy mở trang bằng HTTPS (xem hướng dẫn USE_HTTPS).");return;}
    try{
      mediaStream=await navigator.mediaDevices.getUserMedia({
        video:{facingMode:{ideal:"environment"},width:{ideal:1920},height:{ideal:1080}},audio:false});
    }catch(e){alert("Không mở được camera điện thoại: "+e);return;}
    try{await postSource("mobile");}catch(e){stopPhoneCamera();camMsg(String(e));return;}
    currentSource="mobile";
    if(hlsPlayer)hlsPlayer.stopLoad();
    videoElement.pause();
    phoneVideo.srcObject=mediaStream;
    setSourceUI("mobile");
    setPreviewStatus("Video: điện thoại → server",true);
    try{if(navigator.wakeLock)wakeLock=await navigator.wakeLock.request("screen");}catch(e){}
    mobileLoop(++mobileGen);
  }else{
    stopPhoneCamera();
    try{await postSource("ip");}catch(e){camMsg(String(e));}
    currentSource="ip";
    if(hlsPlayer)hlsPlayer.startLoad(-1);
    videoElement.play().catch(()=>{});
    setSourceUI("ip");
  }
  lastRevision=-1;lookupRevision=-1;clearDetections();
  $("results").innerHTML='<tr><td colspan="5" class="empty">Đang chờ nhận diện...</td></tr>';
  $("selected").textContent="Chưa chọn nội dung OCR";
  $("selected").classList.remove("warning");
  setBest(null);scanOCR();
}

function phoneFrameBlob(){
  const v=phoneVideo;
  if(!v.videoWidth)return Promise.resolve(null);
  const w=Math.min(MOBILE_FRAME_W,v.videoWidth),h=Math.round(v.videoHeight*w/v.videoWidth);
  if(fcv.width!==w)fcv.width=w;
  if(fcv.height!==h)fcv.height=h;
  fctx.drawImage(v,0,0,w,h);
  return new Promise(r=>fcv.toBlob(r,"image/jpeg",MOBILE_JPEG_Q));
}

// Một vòng lặp tuần tự: không bao giờ có 2 request chồng nhau
async function mobileLoop(gen){
  while(mobileGen===gen){
    let wait=MOBILE_IDLE_MS;
    try{
      if(!paused){
        const blob=await phoneFrameBlob();
        if(blob){
          const ctl=new AbortController(),to=setTimeout(()=>ctl.abort(),4000);
          try{
            const r=await fetch("/api/mobile_frame",{method:"POST",cache:"no-store",
              headers:{"Content-Type":"image/jpeg"},body:blob,signal:ctl.signal});
            const d=await r.json();
            wait=d.state==="SCANNING"?MOBILE_SCAN_MS:MOBILE_IDLE_MS;
          }finally{clearTimeout(to);}
        }
      }
    }catch(e){wait=1000;}
    await sleep(wait);
  }
}

document.addEventListener("visibilitychange",async()=>{
  if(document.visibilityState==="visible"&&currentSource==="mobile"&&navigator.wakeLock){
    try{wakeLock=await navigator.wakeLock.request("screen");}catch(e){}
  }
});
$("src_ip").onclick=()=>switchSource("ip");
$("src_mobile").onclick=()=>switchSource("mobile");
$("btn_rescan").onclick=async()=>{
  await fetch("/api/mobile/rescan",{method:"POST"});
  lastRevision=-1;lookupRevision=-1;scanOCR();
};
</script>
</body>
</html>
"""


# ============================================================
# 10. FLASK ROUTES
# ============================================================
@app.route("/")
def index():
    return redirect(url_for("pick_page"))


@app.route("/pick")
def pick_page():
    return (
        PICK_HTML
        .replace("__MEDIAMTX_PUBLIC_URL__", json.dumps(MEDIAMTX_PUBLIC_URL))
        .replace("__MEDIAMTX_PORT__", str(MEDIAMTX_PORT))
        .replace("__MEDIAMTX_STREAM_PATH__", MEDIAMTX_STREAM_PATH)
    )


@app.route("/api/pick/capture", methods=["POST"])
def api_pick_capture():
    source = get_source()
    if source == "mobile":
        with mobile_cv:
            frame = MF["frame"]
    else:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            return jsonify(
                error="no_frame",
                message="Chưa có khung hình từ camera.",
            ), 503

    snapshot = recognition_status()
    h, w = frame.shape[:2] if frame is not None else (0, 0)
    return jsonify(
        w=w,
        h=h,
        dets=snapshot["dets"],
        ocr=snapshot["message"],
        state=snapshot["state"],
        session_id=snapshot["session_id"],
        revision=snapshot["revision"],
        success=snapshot["success"],
        paused=snapshot["paused"],
        sharpness=snapshot["sharpness"],
        lookup_status=snapshot["lookup_status"],
        lookup_error=snapshot["lookup_error"],
        matches=snapshot["matches"],
        lookup_revision=snapshot["lookup_revision"],
        source=source,
    )


@app.route("/api/pick/control", methods=["POST"])
def api_pick_control():
    data = request.get_json(silent=True) or {}
    paused = data.get("paused")
    if not isinstance(paused, bool):
        return jsonify(
            error="invalid_pause_state",
            message="Trạng thái tạm dừng phải là true hoặc false.",
        ), 400

    if paused:
        recognition_pause_event.set()
    else:
        recognition_pause_event.clear()
    return jsonify(ok=True, paused=recognition_pause_event.is_set())


@app.route("/api/pick/heartbeat", methods=["POST"])
def api_pick_heartbeat():
    global web_ui_last_seen
    with recognition_lock:
        web_ui_last_seen = time.time()
    return jsonify(ok=True)


@app.route("/api/pick/clear", methods=["POST"])
def api_pick_clear():
    global web_ui_last_seen
    with recognition_lock:
        web_ui_last_seen = 0.0
    clear_recognition_cache()
    return jsonify(ok=True)


@app.route("/api/source", methods=["POST"])
def api_source():
    src = "mobile" if (request.get_json(silent=True) or {}).get("source") == "mobile" else "ip"
    with source_lock:
        SOURCE["name"] = src
    with mobile_cv:
        M.update(state="IDLE", present=0, empty=0, changed=0, locked=None)
        MF.update(frame=None)
    clear_recognition_cache()
    publish_recognition("WAIT_OBJECT", "Đang chờ sản phẩm...", None, [])
    return jsonify(ok=True, source=src)


@app.route("/api/mobile_frame", methods=["POST"])
def api_mobile_frame():
    if get_source() != "mobile":
        return jsonify(ok=False, state="STOPPED"), 409
    raw = request.get_data(cache=False)
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
    if frame is None:
        return jsonify(ok=False, error="ảnh không hợp lệ"), 400
    with mobile_cv:
        MF.update(frame=frame, seq=MF["seq"] + 1, t=time.time())
        mobile_cv.notify_all()
    return jsonify(ok=True, state=M["state"])


@app.route("/api/mobile/rescan", methods=["POST"])
def api_mobile_rescan():
    with mobile_cv:
        M.update(state="IDLE", present=0, empty=0, changed=0, locked=None)
    publish_recognition("WAIT_OBJECT", "Đang chờ sản phẩm...", None, [])
    return jsonify(ok=True)


@app.route("/api/pick/search", methods=["POST"])
def api_pick_search():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if len(norm(text)) < 2:
        return jsonify(results=[])

    try:
        results = search_inventory(text)
        return jsonify(results=[
            {key: value for key, value in result.items()
             if key != "matched_text"}
            for result in results
        ])

    except Exception as exc:
        set_mongo_health(False, f"Lỗi truy vấn MongoDB: {exc}")
        print("[MONGODB SEARCH ERROR]", exc)
        return jsonify(
            message=f"Lỗi truy vấn MongoDB: {exc}",
            results=[],
        ), 500


@app.route("/api/cam/focus", methods=["POST"])
def api_cam_focus():
    try:
        speed = int((request.get_json(silent=True) or {}).get("speed", 0))
    except (TypeError, ValueError):
        speed = 0
    code, text = cam_focus_move(speed)
    ok = 200 <= code < 300
    return jsonify(ok=ok, status=code, message="" if ok else isapi_msg(code, text))


@app.route("/api/cam_zoom_to", methods=["POST"])
def api_cam_zoom_to():
    d = request.get_json(silent=True) or {}
    try:
        pct = float(d.get("pct", 0))
    except (TypeError, ValueError):
        pct = 0.0
    code, text = cam_zoom_to(pct, home=bool(d.get("home")))
    ok = 200 <= code < 300
    return jsonify(ok=ok, status=code,
                   mode="absolute" if CAM["zoom_abs"] else "timed",
                   message="" if ok else isapi_msg(code, text))


@app.route("/api/cam_focus_mode", methods=["POST"])
def api_cam_focus_mode():
    mode = (request.get_json(silent=True) or {}).get("mode")
    if mode not in ("auto", "manual"):
        return jsonify(ok=False, message="Chế độ không hợp lệ."), 400
    code, text = cam_set_focus_style("AUTO" if mode == "auto" else "MANUAL")
    ok = 200 <= code < 300
    return jsonify(ok=ok, message=("Đã bật lấy nét tự động." if mode == "auto"
                                   else "Đã chuyển lấy nét thủ công.") if ok
                   else isapi_msg(code, text))


@app.route("/api/cam/probe")
def api_cam_probe():
    out = {}
    for path in ("/ISAPI/System/deviceInfo",
                 f"/ISAPI/Image/channels/{CAMERA_CHANNEL}/focusConfiguration",
                 f"/ISAPI/PTZCtrl/channels/{CAMERA_CHANNEL}/capabilities",
                 f"/ISAPI/System/Video/inputs/channels/{CAMERA_CHANNEL}/focus/capabilities"):
        code, text = isapi("GET", path)
        out[path] = {"status": code, "body": (text or "")[:600]}
    return jsonify(out)


@app.route("/api/health")
def api_health():
    snapshot = recognition_status()
    db_status = get_mongo_health()
    media_mtx_host = (
        "127.0.0.1" if MEDIAMTX_HOST in ("", "0.0.0.0", "::")
        else MEDIAMTX_HOST
    )
    try:
        with socket.create_connection(
                (media_mtx_host, MEDIAMTX_PORT), timeout=0.5):
            mediamtx_ok = True
    except OSError:
        mediamtx_ok = False

    with frame_lock:
        camera_frame_ready = latest_frame is not None
    preview_ok = mediamtx_ok and camera_frame_ready
    if preview_ok:
        preview_message = f"MediaMTX LL-HLS · cổng {MEDIAMTX_PORT}"
    elif not mediamtx_ok:
        preview_message = (
            f"Không kết nối MediaMTX tại {media_mtx_host}:{MEDIAMTX_PORT}"
        )
    else:
        preview_message = "MediaMTX sẵn sàng; đang chờ frame camera"
    return jsonify(
        camera=camera_status,
        preview_ok=preview_ok,
        preview_message=preview_message,
        ocr=snapshot["message"],
        recognition_state=snapshot["state"],
        recognition_paused=snapshot["paused"],
        session_id=snapshot["session_id"],
        database_ok=db_status["ok"],
        database_message=db_status["message"],
        database_name=MONGO_DB_NAME,
        collection_name=MONGO_COLLECTION,
        focus_mode=CAM["focus_mode"],
        zoom_pct=CAM["zoom_pct"],
        source=get_source(),
        running=not stop_event.is_set(),
    )


# ============================================================
# 11. KHỞI CHẠY ỨNG DỤNG
# ============================================================
def get_lan_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


if __name__ == "__main__":
    if not CAMERA_PASS:
        print("CẢNH BÁO: CAMERA_PASS đang trống - camera sẽ trả về 401 Unauthorized.")
    try:
        mongo_client.admin.command("ping")
        document_count = warehouse_col.estimated_document_count()
        set_mongo_health(
            True, f"Kết nối OK · {document_count} bản ghi"
        )
        print(
            f"MongoDB kết nối thành công: {MONGO_DB_NAME}."
            f"{MONGO_COLLECTION} ({document_count} bản ghi)."
        )
    except Exception as exc:
        set_mongo_health(False, f"Lỗi kết nối MongoDB: {exc}")
        raise RuntimeError(f"Không kết nối được MongoDB: {exc}") from exc

    # Khởi chạy các worker ngầm
    threading.Thread(target=camera_worker, daemon=True, name="camera-worker").start()
    threading.Thread(target=recognition_worker, daemon=True, name="recognition-worker").start()
    threading.Thread(target=mobile_worker, daemon=True, name="mobile-worker").start()
    threading.Thread(target=inventory_search_worker, daemon=True, name="inventory-search-worker").start()
    threading.Thread(target=cam_watchdog, daemon=True, name="cam-watchdog").start()

    lan_ip = get_lan_ip()
    use_https = os.environ.get("USE_HTTPS", "0") == "1"
    scheme = "https" if use_https else "http"
    print("=" * 60)
    print(" HỆ THỐNG NHẬN DIỆN SẢN PHẨM (IP CAMERA + ĐIỆN THOẠI)")
    print(f" Giao diện chính : {scheme}://127.0.0.1:{WEB_PORT}/pick")
    print(f" Mạng LAN (Web)  : {scheme}://{lan_ip}:{WEB_PORT}/pick")
    print(
        f" MediaMTX HLS    : http://{MEDIAMTX_HOST}:{MEDIAMTX_PORT}/"
        f"{MEDIAMTX_STREAM_PATH}/index.m3u8"
    )
    print("=" * 60)

    try:
        if use_https:
            cert = os.path.join(BASE_DIR, "cert.pem")
            key = os.path.join(BASE_DIR, "key.pem")
            ctx = (cert, key) if os.path.exists(cert) and os.path.exists(key) else "adhoc"
            app.run(host=WEB_HOST, port=WEB_PORT, threaded=True,
                    ssl_context=ctx, use_reloader=False)
        else:
            serve(
                app,
                host=WEB_HOST,
                port=WEB_PORT,
                threads=WAITRESS_THREADS,
            )
    finally:
        stop_event.set()
        with mobile_cv:
            mobile_cv.notify_all()
        mongo_client.close()