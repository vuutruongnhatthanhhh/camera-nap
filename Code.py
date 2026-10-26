import base64
import csv
import html
import os
import re
import socket
import threading
import time
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from functools import wraps
from urllib.parse import quote

import cv2
from flask import (Flask, Response, jsonify, redirect, request,
                   send_from_directory, session, url_for)
from paddleocr import PaddleOCR
from pymongo import MongoClient
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:
    zbar_decode = None

# ============================================================
# 1. CẤU HÌNH & MONGODB
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ocr_manufacturer_results.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "product_images")
os.makedirs(IMAGE_DIR, exist_ok=True)


MONGO_URI = "mongodb://trantiendat:123456@192.168.1.59:27017/nap-software?authSource=admin&serverSelectionTimeoutMS=5000"
client = MongoClient(MONGO_URI)
db_mongo = client["nap-software"]
warehouse_col = db_mongo["warehouse"]

CAMERA_IP = os.environ.get("CAMERA_IP", "192.168.1.140")
CAMERA_USER = os.environ.get("CAMERA_USER", "admin")
CAMERA_PASS = os.environ.get("CAMERA_PASS", "Admin123")
RTSP_URL = (f"rtsp://{quote(CAMERA_USER, safe='')}:{quote(CAMERA_PASS, safe='')}"
            f"@{CAMERA_IP}:554/Streaming/Channels/101")

WEB_HOST = "0.0.0.0"
WEB_PORT = int(os.environ.get("WEB_PORT", "5000"))
WEB_USER = os.environ.get("WEB_USER", "admin")
WEB_PASS = os.environ.get("WEB_PASS", "Admin123")
WEB_PASS_HASH = generate_password_hash(WEB_PASS)

ROI_X1, ROI_X2, ROI_Y1, ROI_Y2 = 0.20, 0.80, 0.20, 0.80
OCR_INTERVAL_SEC = 0.25
COOLDOWN_SEC = 2.0
UNSEEN_TIMEOUT_SEC = 3.0      # mã lạ liên tục 3s thì chụp & mở form nhập liệu
IGNORE_SEC = 30.0             # sau khi bấm "Bỏ qua", không hỏi lại mã đó trong 30s
MIN_OCR_CONFIDENCE = 0.50
FORM_MIN_CONFIDENCE = 0.30    # ngưỡng thấp hơn để người dùng còn chọn trong form
DUP_WARN_RATIO = 0.85

STREAM_WIDTH, STREAM_FPS, JPEG_QUALITY = 960, 12, 70

print("Đang khởi tạo PaddleOCR...")
ocr = PaddleOCR(use_textline_orientation=False, device="cpu",
                 enable_mkldnn=False,  # mkldnn gây lỗi PIR trên PaddleOCR 3.x/CPU
                 use_doc_orientation_classify=False, use_doc_unwarping=False,  # bỏ 2 bước không cần cho nhãn sản phẩm
                 text_det_limit_side_len=480,
                 # model "mobile" nhẹ hơn nhiều so với "medium" mặc định -
                 # giảm thời gian quét từ ~6s xuống ~1.2s mà độ chính xác vẫn
                 # tốt với nhãn sản phẩm, đổi lại tự tải model lần chạy đầu
                 text_detection_model_name="PP-OCRv5_mobile_det",
                 text_recognition_model_name="PP-OCRv5_mobile_rec")
print("PaddleOCR sẵn sàng.")

# ============================================================
# 3. TRẠNG THÁI DÙNG CHUNG
# ============================================================
frame_lock = threading.Lock()
state_lock = threading.Lock()
jpeg_lock = threading.Lock()
ocr_lock = threading.Lock()   # PaddleOCR không an toàn khi gọi đồng thời từ nhiều luồng
stop_event = threading.Event()
scan_enabled = threading.Event()

latest_frame = None
latest_jpeg = None
ignore_until = {}      # norm(code) -> thời điểm hết hạn bỏ qua

S = {
    "camera": "Đang kết nối camera...",
    "code": None, "info": None, "ocr": "Chờ đọc...",
    "last_sku": None, "last_time": 0.0,
    "pending": None,   # dữ liệu chụp chờ nhập liệu
}

# ============================================================
# 4. XỬ LÝ ẢNH & TÌM KIẾM
# ============================================================
def norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())

def roi_box(frame):
    h, w = frame.shape[:2]
    return int(w * ROI_X1), int(h * ROI_Y1), int(w * ROI_X2), int(h * ROI_Y2)

def get_roi(frame):
    x1, y1, x2, y2 = roi_box(frame)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # bỏ upscale 2x - model "mobile" + text_det_limit_side_len đã tự co giãn
    # ảnh hợp lý, upscale thêm chỉ làm chậm mà không tăng độ chính xác
    den = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    blur = cv2.GaussianBlur(den, (0, 0), 2.0)
    sharp = cv2.addWeighted(den, 1.5, blur, -0.5, 0)
    out = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(sharp)
    # PaddleOCR 3.x đòi ảnh 3 kênh (BGR) cho các bước tiền xử lý nội bộ của pipeline
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

NOISE = {"POWERSUPPLY", "POWER", "SUPPLY", "MODEL", "INPUT", "OUTPUT", "MADEIN",
         "WARNING", "CAUTION", "V", "A", "AC", "DC", "HZ", "CE", "ROHS"}

def is_noise_text(text):
    t = text.upper()
    return t in NOISE or len(t) < 2

def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def collect_dets(result, ox, oy, scale=1.0):
    # PaddleOCR >=3.x: predict() trả về list các OCRResult (dict-like) với
    # rec_texts / rec_scores / rec_boxes (hoặc rec_polys), không còn dạng
    # [[points, (text, conf)], ...] như bản cũ.
    dets = []
    if not result:
        return dets
    res = result[0]
    try:
        texts = list(res["rec_texts"])
        scores = list(res["rec_scores"])
    except (KeyError, TypeError, IndexError):
        return dets
    boxes = None
    try:
        b = list(res["rec_boxes"])
        boxes = b if len(b) == len(texts) else None
    except (KeyError, TypeError):
        boxes = None
    polys = None
    if boxes is None:
        try:
            polys = list(res["rec_polys"])
        except (KeyError, TypeError):
            polys = None
    for i, text in enumerate(texts):
        try:
            conf = float(scores[i])
        except (IndexError, TypeError, ValueError):
            continue
        text = str(text).strip()
        if conf < FORM_MIN_CONFIDENCE or not text:
            continue
        try:
            if boxes is not None:
                x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            elif polys is not None:
                xs = [float(p[0]) for p in polys[i]]
                ys = [float(p[1]) for p in polys[i]]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                continue
        except (IndexError, TypeError, ValueError):
            continue
        dets.append({"text": text, "conf": round(conf, 3),
                     "box": [x1 / scale + ox, y1 / scale + oy,
                             x2 / scale + ox, y2 / scale + oy]})
    return dets

def load_rows():
    return list(warehouse_col.find({}, {"_id": 0}))

def match_product(text, rows):
    c = norm(text)
    if len(c) < 2:
        return None
    best, top = None, 0.0
    for r in rows:
        sku, brand, series = norm(r.get("sku")), norm(r.get("brand")), norm(r.get("series"))
        keys = [k for k in (sku, series, brand) if k]
        if c in keys:
            score = 1.0
        elif len(c) >= 3 and any(c in k or k in c for k in keys if len(k) >= 3):
            score = 0.90
        else:
            score = max([SequenceMatcher(None, c, k).ratio() for k in (sku, series) if k] or [0])
        if score > top:
            top = score
            best = {"sku": r["sku"], "product_name": r["product_name"],
                    "location": r["location"], "quantity": r["quantity"],
                    "status": r.get("status", "OK"), "found": True,
                    "confidence_msg": f"Khớp (Score: {score*100:.0f}%)"}
    return best if top >= 0.65 else None

def save_result(code, conf, info):
    new_file = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "scanned_text", "product_name", "location",
                        "status", "confidence", "camera_ip"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), code,
                    info["product_name"], info["location"],
                    "FOUND" if info["found"] else "NOT_FOUND", f"{conf:.3f}", CAMERA_IP])

# ============================================================
# 5. LUỒNG CAMERA / OCR / STREAM
# ============================================================
def camera_worker():
    global latest_frame
    while not stop_event.is_set():
        cap = None
        try:
            with state_lock:
                S["camera"] = "Đang kết nối camera..."
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                time.sleep(3)
                continue
            with state_lock:
                S["camera"] = "Đã kết nối camera"
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                with frame_lock:
                    latest_frame = frame
        except Exception:
            pass
        finally:
            if cap:
                cap.release()
        time.sleep(2)

def stream_worker():
    global latest_jpeg
    delay = 1.0 / STREAM_FPS
    while not stop_event.is_set():
        with frame_lock:
            f = None if latest_frame is None else latest_frame.copy()
        if f is None:
            time.sleep(0.2)
            continue
        x1, y1, x2, y2 = roi_box(f)
        cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
        h, w = f.shape[:2]
        if w > STREAM_WIDTH:
            f = cv2.resize(f, (STREAM_WIDTH, int(h * STREAM_WIDTH / w)))
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with jpeg_lock:
                latest_jpeg = buf.tobytes()
        time.sleep(delay)

def confirm(info, conf, now):
    if info["sku"] == S["last_sku"] and (now - S["last_time"]) < COOLDOWN_SEC + 0.4:
        return
    with state_lock:
        S.update(code=info["sku"], info=info, last_sku=info["sku"], last_time=now)
    save_result(info["sku"], conf, info)
    print(f"[{datetime.now():%H:%M:%S}] CHỐT KHO: {info['product_name']} | {info['location']}")

def new_session(now):
    return {"start": now, "votes": {}, "best_frame": None, "best_dets": [], "best_score": -1.0}

def session_add(sess, frame, dets, cands):
    for text, conf in cands:
        v = sess["votes"].setdefault(norm(text), {"text": text, "n": 0, "conf": 0.0})
        v["n"] += 1
        v["conf"] += conf
    score = sharpness(frame)
    if score > sess["best_score"]:
        sess.update(best_score=score, best_frame=frame.copy(), best_dets=dets)

def finalize_session(sess, now):
    ranked = sorted(sess["votes"].values(),
                    key=lambda v: (v["n"], v["conf"] / v["n"]), reverse=True)
    cands = [v["text"] for v in ranked[:8]]
    if not cands or sess["best_frame"] is None:
        return
    if all(ignore_until.get(norm(c), 0) > now for c in cands[:2]):
        return

    frame = sess["best_frame"]
    barcodes = []
    if zbar_decode:
        try:
            barcodes = [b.data.decode("utf-8", "ignore") for b in zbar_decode(frame)]
        except Exception:
            pass

    pid = uuid.uuid4().hex[:10]
    filename = f"{norm(cands[0]) or 'NEW'}_{datetime.now():%Y%m%d_%H%M%S}_{pid}.jpg"
    cv2.imwrite(os.path.join(IMAGE_DIR, filename), frame)
    h, w = frame.shape[:2]
    with state_lock:
        S["pending"] = {"id": pid, "file": filename, "w": w, "h": h,
                        "dets": sess["best_dets"], "candidates": barcodes + [c for c in cands if c not in barcodes],
                        "barcodes": barcodes}
    print(f"[{datetime.now():%H:%M:%S}] MÃ LẠ - chờ nhập liệu trên web: {cands[0]}")

def ocr_worker():
    scan_enabled.set()
    sess = None
    while not stop_event.is_set():
        if not scan_enabled.is_set() or S["pending"] is not None:
            sess = None
            time.sleep(0.1)
            continue
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.1)
            continue

        now = time.time()
        if S["code"] is not None:
            if now - S["last_time"] < COOLDOWN_SEC:
                time.sleep(0.1)
                continue
            with state_lock:
                S["code"], S["info"] = None, None
            sess = None

        try:
            roi, (ox, oy, _, _) = get_roi(frame)
            with ocr_lock:
                result = ocr.predict(preprocess_roi(roi))
            dets = collect_dets(result, ox, oy)
            rows = load_rows()

            matched, best_conf, cands = None, 0.0, []
            for d in dets:
                cleaned = re.sub(r"\s+", "", d["text"])
                if d["conf"] < MIN_OCR_CONFIDENCE or is_noise_text(cleaned):
                    continue
                if not (2 <= len(cleaned) <= 15 and any(ch.isalpha() for ch in cleaned)):
                    continue
                cands.append((cleaned, d["conf"]))
                m = match_product(cleaned, rows)
                if m:
                    matched, best_conf = m, d["conf"]
                    break

            with state_lock:
                S["ocr"] = " | ".join(f"{t} ({c:.2f})" for t, c in cands[:3]) or "Đang quét..."

            if matched:
                sess = None
                confirm(matched, best_conf, now)
            elif cands:
                if sess is None:
                    sess = new_session(now)
                session_add(sess, frame, dets, cands)
                if now - sess["start"] >= UNSEEN_TIMEOUT_SEC:
                    finalize_session(sess, now)
                    sess = None
            else:
                sess = None
        except Exception as exc:
            with state_lock:
                S["ocr"] = f"Lỗi OCR: {exc}"
        time.sleep(OCR_INTERVAL_SEC)

# ============================================================
# 6. WEB (FLASK)
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get("WEB_SECRET", os.urandom(24).hex())

def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

def login_required(fn):
    # Đã tắt yêu cầu đăng nhập theo yêu cầu - ứng dụng chỉ chạy trong mạng
    # LAN nội bộ (đã giới hạn bằng firewall ở dải 192.168.1.0/24), không cần
    # xác thực thêm ở tầng ứng dụng nữa.
    return fn

CSS = """<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:Arial,sans-serif;background:#202020;color:#fff;margin:0;padding:12px}
a{color:#ffaa00}.box{background:#111;border:2px groove #444;padding:12px;margin-top:10px;max-width:960px;border-radius:8px}
.code{color:#00ff88;font:bold 20px Consolas,monospace}.prod{color:#0ff;font-size:20px;font-weight:bold}
.loc{color:#f55;font-size:20px;font-weight:bold}.small{color:#aaa;font-size:13px}
img,canvas{max-width:100%;border-radius:8px;background:#000}
button,input,select{padding:9px;border-radius:6px;border:0;font-size:15px}
button{background:#00aa66;color:#fff;cursor:pointer;margin:3px}button.gray{background:#555}button.red{background:#c0392b}
input{width:100%;box-sizing:border-box;background:#fff;color:#000}
.row{display:flex;gap:8px;align-items:center;margin:5px 0}.row label{width:110px;flex:none}
.chip{display:inline-block;background:#333;border:1px solid #666;padding:4px 9px;margin:3px;border-radius:14px;cursor:pointer}
.banner{background:#7a4b00;border:2px solid #ffaa00;padding:12px;border-radius:8px;margin-top:10px;max-width:960px}
table{border-collapse:collapse;width:100%;max-width:1100px}td,th{border:1px solid #444;padding:6px;text-align:left}
.pend{color:#ffaa00}
</style>"""

LOGIN_HTML = CSS + """<title>Đăng nhập</title>
<form method="post" style="max-width:290px;margin:12vh auto;background:#111;padding:26px;border-radius:10px">
<h3>Truy xuất kho</h3><div style="color:#f66">%%ERR%%</div>
<input type="hidden" name="next" value="%%NEXT%%">
<input name="username" placeholder="Tài khoản" autofocus style="margin:6px 0">
<input name="password" type="password" placeholder="Mật khẩu" style="margin:6px 0">
<button style="width:100%">Đăng nhập</button></form>"""

INDEX_HTML = CSS + """<title>Nhận diện & truy xuất kho</title>
<h3>NHẬN DIỆN SẢN PHẨM & TRUY XUẤT KHO
<a href="/products" style="font-size:13px">[Danh sách kho]</a>
<a href="/logout" style="font-size:13px">[Đăng xuất]</a></h3>
<div id="banner" class="banner" style="display:none">⚠ Phát hiện sản phẩm mới chưa có trong kho.
<a href="/new"><button>NHẬP LIỆU NGAY</button></a></div>
<img src="/video_feed" style="width:100%;max-width:960px;margin-top:8px">
<div class="box">
<div class="small" id="status">...</div><div class="code" id="code">--</div>
<div class="prod" id="prod">--</div><div class="loc" id="loc">--</div>
<div class="small" id="ocr">--</div>
<button onclick="scan('start')">BẮT ĐẦU QUÉT</button><button class="gray" onclick="scan('stop')">DỪNG QUÉT</button>
</div>
<script>
const $=id=>document.getElementById(id);
async function scan(a){await fetch('/api/scan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:a})});tick()}
async function tick(){try{const r=await fetch('/api/status');if(r.status===401){location='/login';return}
const d=await r.json();
$('status').textContent='Camera: '+d.camera+' | '+(d.scanning?'ĐANG QUÉT':'ĐÃ DỪNG');
$('code').textContent=d.code?'MÃ ĐÃ CHỐT: '+d.code:'Đang chờ đưa nhãn vào khung...';
$('prod').textContent=d.code?'📦 '+d.product+(d.status==='PENDING'?' [chưa đủ thông tin]':''):'📦 --';
$('loc').textContent=d.code?'📍 '+d.location:'📍 --';
$('ocr').textContent='OCR đọc được: '+d.ocr;
$('banner').style.display=d.pending?'block':'none';}catch(e){}}
setInterval(tick,500);tick();
</script>"""

NEW_HTML = CSS + """<title>Thêm sản phẩm mới</title>
<h3>THÊM SẢN PHẨM MỚI <a href="/" style="font-size:13px">[Về trang chính]</a></h3>
<div class="small">Bấm vào ô cần điền, rồi bấm vào chữ (khung xanh) trên ảnh để điền tự động. Bấm nhiều chữ sẽ ghép lại.</div>
<canvas id="cv"></canvas>
<div class="box"><div class="small">Gợi ý (bấm để điền vào ô đang chọn):</div><div id="chips"></div>
<div id="form"></div>
<button onclick="save(false)">LƯU</button><button class="gray" onclick="skip()">BỎ QUA</button>
<div id="msg" class="small"></div></div>
<datalist id="dl_brand"></datalist><datalist id="dl_loc"></datalist>
<script>
const $=id=>document.getElementById(id);
const FIELDS=[['sku','SKU / Mã',''],['brand','Hãng','dl_brand'],['series','Series',''],['name','Tên sản phẩm',''],['location','Vị trí kệ','dl_loc'],['qty','Số lượng','']];
let P=null,focusId='sku',scale=1,img=new Image(),hit=new Set();
FIELDS.forEach(([k,t,dl])=>{const r=document.createElement('div');r.className='row';
r.innerHTML=`<label>${t}</label><input id="f_${k}" ${dl?`list="${dl}"`:''} autocomplete="off"><button class="gray" tabindex="-1">✕</button>`;
r.querySelector('button').onclick=()=>{$('f_'+k).value='';};$('form').appendChild(r);$('f_'+k).addEventListener('focus',()=>focusId=k);});$('f_qty').value='0';
function fill(text){const el=$('f_'+focusId);const v=el.value.trim();
el.value=(v&&!(focusId==='qty'))?v+' '+text:text;}
function draw(){const cv=$('cv'),ctx=cv.getContext('2d');
scale=Math.min(900/P.w,1);cv.width=P.w*scale;cv.height=P.h*scale;
ctx.drawImage(img,0,0,cv.width,cv.height);ctx.lineWidth=2;
P.dets.forEach((d,i)=>{const[b0,b1,b2,b3]=d.box.map(v=>v*scale);
ctx.strokeStyle=hit.has(i)?'#ffaa00':'#00ff88';ctx.strokeRect(b0,b1,b2-b0,b3-b1);});}
$('cv').addEventListener('click',e=>{const r=$('cv').getBoundingClientRect();
const x=(e.clientX-r.left)*($('cv').width/r.width)/scale,y=(e.clientY-r.top)*($('cv').height/r.height)/scale;
let best=-1,area=1e18;P.dets.forEach((d,i)=>{const[a,b,c,f]=d.box;
if(x>=a&&x<=c&&y>=b&&y<=f&&(c-a)*(f-b)<area){area=(c-a)*(f-b);best=i}});
if(best>=0){fill(P.dets[best].text);hit.add(best);draw();}});
async function init(){const r=await fetch('/api/pending');if(r.status===401){location='/login';return}
P=await r.json();if(!P.id){document.body.innerHTML='<p>Không có sản phẩm nào đang chờ. <a href="/">Về trang chính</a></p>';return}
P.brands.forEach(b=>$('dl_brand').appendChild(new Option(b)));P.locations.forEach(l=>$('dl_loc').appendChild(new Option(l)));
P.candidates.forEach(c=>{const s=document.createElement('span');s.className='chip';s.textContent=c;s.onclick=()=>fill(c);$('chips').appendChild(s)});
if(P.candidates.length)$('f_sku').value=P.candidates[0];
img.onload=draw;img.src=P.image;}
async function save(force){const body={id:P.id,force:force};
FIELDS.forEach(([k])=>body[k]=$('f_'+k).value.trim());
const r=await fetch('/api/products',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
const d=await r.json();
if(r.ok){alert(d.message);location='/';return}
if(d.needs_confirm){if(confirm(d.message+'\\nVẫn muốn lưu thành sản phẩm mới?'))save(true);return}
$('msg').textContent=d.message||'Lỗi';}
async function skip(){await fetch('/api/pending/skip',{method:'POST'});location='/';}
init();
</script>"""

PICK_HTML = CSS + """<title>Quét tự động</title>
<h3>QUÉT <a href="/" style="font-size:13px">[Về trang chính]</a></h3>
<div class="small" id="hint">Đưa nhãn vào khung hình - hệ thống sẽ tự chụp &amp; quét sau vài giây (hoặc bấm "QUÉT NGAY" để quét luôn). Khi nhận diện được chữ sẽ tự gửi toàn bộ về ô tìm kiếm, không cần bấm gì thêm.</div>
<img id="live" src="/video_feed" style="width:100%;max-width:900px;border-radius:8px;background:#000">
<canvas id="cv" style="display:none"></canvas>
<div class="box">
<div class="row" id="selRow" style="display:none"><label>Đã chọn</label><input id="sel" readonly></div>
<button onclick="startScan()">QUÉT NGAY</button>
<button class="gray" id="clearBtn" style="display:none" onclick="clearSel()">XOÁ</button>
<button id="sendBtn" style="display:none" onclick="sendSelected()">GỬI</button>
<div id="msg" class="small">Sẽ tự quét sau 3 giây...</div>
</div>
<script>
const $=id=>document.getElementById(id);
const AUTO_DELAY_MS=3000;
const START_DELAY_MS=3000;
// mode=pick: bấm chọn từng chữ trên ảnh rồi bấm GỬI (dùng khi cần điền vào
// 1 ô cụ thể, VD "Tên hàng"). Mặc định (không có mode): tự gửi toàn bộ chữ
// nhận diện được ngay khi quét xong, không cần thao tác gì thêm.
const PICK_MODE=new URLSearchParams(location.search).get('mode')==='pick';
if(PICK_MODE){
$('hint').textContent='Đưa nhãn vào khung hình - hệ thống tự chụp & quét sau vài giây. Bấm vào chữ (khung xanh) trên ảnh để chọn - có thể bấm nhiều chữ, rồi bấm "GỬI".';
$('selRow').style.display='';$('clearBtn').style.display='';$('sendBtn').style.display='';
}
let img=new Image(),autoTimer=null,busy=false,sent=false,dets=[],selected=[],hit=new Set(),scale=1,curW=0,curH=0;
function draw(){const cv=$('cv'),ctx=cv.getContext('2d');
scale=Math.min(900/curW,1);cv.width=curW*scale;cv.height=curH*scale;
ctx.drawImage(img,0,0,cv.width,cv.height);ctx.lineWidth=2;
dets.forEach((d,i)=>{const[b0,b1,b2,b3]=d.box.map(v=>v*scale);
ctx.strokeStyle=hit.has(i)?'#ffaa00':'#00ff88';ctx.strokeRect(b0,b1,b2-b0,b3-b1);});}
function stopAuto(){if(autoTimer){clearTimeout(autoTimer);autoTimer=null;}}
function scheduleAuto(delay){if(sent)return;stopAuto();autoTimer=setTimeout(autoCapture,delay);}
function startScan(){$('msg').textContent='Đang quét...';scheduleAuto(0);}
(function countdownThenStart(){
let left=Math.ceil(START_DELAY_MS/1000);
const tick=setInterval(()=>{if(sent){clearInterval(tick);return}
left--;if(left>0)$('msg').textContent='Sẽ tự quét sau '+left+' giây...';},1000);
setTimeout(()=>{clearInterval(tick);if(!sent)startScan();},START_DELAY_MS);
})();
async function doCapture(){
try{const r=await fetch('/api/pick/capture',{method:'POST'});
if(r.status===401){location='/login';return null}
const d=await r.json();
if(!r.ok){$('msg').textContent=d.message||'Không lấy được khung hình.';return null}
return d;
}catch(e){$('msg').textContent='Lỗi: '+e;return null}}
function sendAll(text){sent=true;stopAuto();
const origin=document.referrer?new URL(document.referrer).origin:'*';
if(window.opener){window.opener.postMessage({type:'ocr-pick-text',text:text},origin);window.close();}
else{$('msg').textContent='Không tìm thấy cửa sổ gốc để gửi. Kết quả: '+text;}}
function clearSel(){selected=[];hit=new Set();$('sel').value='';if(dets.length)draw();}
function sendSelected(){const text=$('sel').value.trim();
if(!text){$('msg').textContent='Chưa chọn chữ nào.';return}
sendAll(text);}
$('cv').addEventListener('click',e=>{if(!PICK_MODE||!dets.length)return;
const r=$('cv').getBoundingClientRect();
const x=(e.clientX-r.left)*($('cv').width/r.width)/scale,y=(e.clientY-r.top)*($('cv').height/r.height)/scale;
let best=-1,area=1e18;dets.forEach((d,i)=>{const[a,b,c,f]=d.box;
if(x>=a&&x<=c&&y>=b&&y<=f&&(c-a)*(f-b)<area){area=(c-a)*(f-b);best=i}});
if(best>=0){selected.push(dets[best].text);hit.add(best);$('sel').value=selected.join(' ');draw();}});
async function autoCapture(){if(busy||sent)return;busy=true;
$('msg').textContent='Đang quét...';
const d=await doCapture();busy=false;
if(!d){scheduleAuto(AUTO_DELAY_MS);return}
dets=d.dets;hit=new Set();curW=d.w;curH=d.h;
img.onload=()=>{$('live').style.display='none';$('cv').style.display='';draw();};img.src=d.image;
if(!dets.length){$('msg').textContent='Không nhận diện được chữ nào, đang quét lại...';scheduleAuto(AUTO_DELAY_MS);return}
if(PICK_MODE){
$('msg').textContent='Tìm thấy '+dets.length+' chữ. Bấm vào chữ muốn chọn (có thể chọn nhiều), rồi bấm GỬI.';
return;}
// sắp xếp theo thứ tự đọc: trên->dưới (theo hàng), trong hàng thì trái->phải
const ordered=dets.slice().sort((a,b)=>{
const ay=(a.box[1]+a.box[3])/2,by=(b.box[1]+b.box[3])/2;
if(Math.abs(ay-by)>15)return ay-by;
return a.box[0]-b.box[0];});
const text=ordered.map(x=>x.text).join(' ');
$('msg').textContent='Tìm thấy: '+text+' — đang gửi về tìm kiếm...';
sendAll(text);}
</script>"""

PRODUCTS_HTML = CSS + """<title>Danh sách kho</title>
<h3>DANH SÁCH KHO <a href="/" style="font-size:13px">[Về trang chính]</a></h3>
<table id="t"><thead><tr><th>SKU</th><th>Hãng</th><th>Series</th><th>Tên</th><th>Vị trí</th><th>SL</th><th>Trạng thái</th><th>Ảnh</th></tr></thead><tbody></tbody></table>
<script>
(async()=>{const r=await fetch('/api/products');if(r.status===401){location='/login';return}
const d=await r.json();const tb=document.querySelector('tbody');
d.forEach(p=>{const tr=document.createElement('tr');
[p.sku,p.brand,p.series,p.product_name,p.location,p.quantity,p.status==='PENDING'?'Chưa đủ thông tin':'Đầy đủ'].forEach((v,i)=>{
const td=document.createElement('td');td.textContent=v??'';if(i===6&&p.status==='PENDING')td.className='pend';tr.appendChild(td)});
const td=document.createElement('td');if(p.image_path){const a=document.createElement('a');a.href='/captures/'+p.image_path;a.target='_blank';a.textContent='xem';td.appendChild(a)}
tr.appendChild(td);tb.appendChild(tr)})})();
</script>"""

def is_safe_next(url):
    return bool(url) and url.startswith("/") and not url.startswith("//")

@app.route("/login", methods=["GET", "POST"])
def login():
    err = ""
    next_url = request.values.get("next", "")
    if not is_safe_next(next_url):
        next_url = ""
    if request.method == "POST":
        if (request.form.get("username") == WEB_USER and
                check_password_hash(WEB_PASS_HASH, request.form.get("password", ""))):
            session["auth"] = True
            return redirect(next_url if is_safe_next(next_url) else url_for("index"))
        err = "Sai tài khoản hoặc mật khẩu"
        time.sleep(1)
    return LOGIN_HTML.replace("%%ERR%%", html.escape(err)).replace("%%NEXT%%", html.escape(next_url))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def index():
    return INDEX_HTML

@app.route("/new")
@login_required
def new_page():
    return NEW_HTML

@app.route("/pick")
@login_required
def pick_page():
    return PICK_HTML

@app.route("/products")
@login_required
def products_page():
    return PRODUCTS_HTML

@app.route("/captures/<path:filename>")
@login_required
def captures(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route("/video_feed")
@login_required
def video_feed():
    def gen():
        while not stop_event.is_set():
            with jpeg_lock:
                buf = latest_jpeg
            if buf:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
            time.sleep(1.0 / STREAM_FPS)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
@login_required
def api_status():
    with state_lock:
        info = S["info"]
        return jsonify(camera=S["camera"], scanning=scan_enabled.is_set(),
                    code=S["code"], ocr=S["ocr"],
                    product=info["product_name"] if info else "",
                    location=info["location"] if info else "",
                    status=info.get("status", "OK") if info else "",
                    found=bool(info and info["found"]),
                    pending=S["pending"] is not None)

@app.route("/api/scan", methods=["POST"])
@login_required
def api_scan():
    action = (request.get_json(silent=True) or {}).get("action")
    if action == "start":
        with state_lock:
            S["code"], S["info"] = None, None
        scan_enabled.set()
    elif action == "stop":
        scan_enabled.clear()
    return jsonify(scanning=scan_enabled.is_set())

@app.route("/api/pending")
@login_required
def api_pending():
    with state_lock:
        p = S["pending"]
    if not p:
        return jsonify(id=None)
    
    brands = warehouse_col.distinct("brand", {"brand": {"$ne": ""}})
    locs = warehouse_col.distinct("location", {"location": {"$ne": ""}})
    
    return jsonify(id=p["id"], image=url_for("captures", filename=p["file"]), w=p["w"], h=p["h"],
                    dets=p["dets"], candidates=p["candidates"], barcodes=p["barcodes"],
                    brands=sorted(brands), locations=sorted(locs))

@app.route("/api/pick/capture", methods=["POST"])
@login_required
def api_pick_capture():
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    if frame is None:
        return jsonify(error="no_frame", message="Chưa có khung hình từ camera."), 503
    try:
        roi, (ox, oy, _, _) = get_roi(frame)
        with ocr_lock:
            result = ocr.predict(preprocess_roi(roi))
        dets = collect_dets(result, ox, oy)
        h, w = frame.shape[:2]
        # trả ảnh trực tiếp (data URI), không ghi ra product_images - ảnh
        # này chỉ dùng hiển thị tạm trên trình duyệt rồi bỏ, không ai dùng
        # lại nên không cần lưu, tránh đầy ổ đĩa theo thời gian
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return jsonify(error="encode_failed", message="Không mã hoá được ảnh."), 500
        image_data_uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
        return jsonify(image=image_data_uri, w=w, h=h, dets=dets)
    except Exception as exc:
        print(f"[{datetime.now():%H:%M:%S}] Lỗi /api/pick/capture: {exc}")
        return jsonify(error="capture_failed", message=f"Lỗi khi chụp/nhận diện: {exc}"), 500

@app.route("/api/pending/skip", methods=["POST"])
@login_required
def api_pending_skip():
    with state_lock:
        p = S["pending"]
        S["pending"] = None
    if p:
        until = time.time() + IGNORE_SEC
        for c in p["candidates"][:5]:
            ignore_until[norm(c)] = until
    return jsonify(ok=True)

@app.route("/api/products", methods=["GET"])
@login_required
def api_products_list():
    rows = list(warehouse_col.find({}, {"_id": 0}).sort("_id", -1))
    return jsonify(rows)

@app.route("/api/products", methods=["POST"])
@login_required
def api_products_add():
    d = request.get_json(silent=True) or {}
    sku = str(d.get("sku", "")).strip().upper()
    if not sku:
        return jsonify(message="SKU là bắt buộc."), 400
    try:
        qty = int(str(d.get("qty", "0")).strip() or 0)
    except ValueError:
        return jsonify(message="Số lượng phải là số nguyên."), 400

    brand = str(d.get("brand", "")).strip().upper()
    series = str(d.get("series", "")).strip().upper()
    name = str(d.get("name", "")).strip()
    loc_in = str(d.get("location", "")).strip()

    existing = warehouse_col.find_one({"sku": sku})
    if existing:
        return jsonify(message=f"SKU {sku} đã tồn tại trong kho."), 409

    if not d.get("force"):
        all_skus = [r["sku"] for r in warehouse_col.find({}, {"sku": 1, "_id": 0})]
        similar = [s for s in all_skus if SequenceMatcher(None, norm(sku), norm(s)).ratio() >= DUP_WARN_RATIO]
        if similar:
            return jsonify(needs_confirm=True,
                           message="Mã này rất giống: " + ", ".join(similar) + " (có thể OCR đọc sai)."), 409

    with state_lock:
        p = S["pending"]
    image_path = p["file"] if p and p["id"] == d.get("id") else ""
    status = "OK" if (name and loc_in) else "PENDING"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = {
        "sku": sku,
        "brand": brand,
        "series": series,
        "product_name": name or "(Chưa đặt tên)",
        "location": loc_in or "CHUA_CO",
        "quantity": qty,
        "updated_at": now,
        "image_path": image_path,
        "status": status
    }

    try:
        warehouse_col.insert_one(doc)
    except Exception as e:
        return jsonify(message=f"Lỗi lưu database: {e}"), 409

    with state_lock:
        if p and p["id"] == d.get("id"):
            S["pending"] = None
        S["last_sku"], S["last_time"] = sku, time.time()
        
    msg = "Đã lưu." if status == "OK" else "Đã lưu (chưa đủ thông tin, trạng thái PENDING)."
    return jsonify(message=msg)

# ============================================================
# 7. KHỞI CHẠY
# ============================================================
if __name__ == "__main__":
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=stream_worker, daemon=True).start()
    threading.Thread(target=ocr_worker, daemon=True).start()
    if WEB_PASS == "Admin123":
        print("CẢNH BÁO: đang dùng mật khẩu web mặc định, hãy đặt biến môi trường WEB_PASS.")
    print(f"Web server: http://{get_lan_ip()}:{WEB_PORT}  (tài khoản: {WEB_USER})")
    try:
        app.run(host=WEB_HOST, port=WEB_PORT, threaded=True, use_reloader=False)
    finally:
        stop_event.set()
        scan_enabled.clear()