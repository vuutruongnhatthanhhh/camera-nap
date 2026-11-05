#!/usr/bin/env bash
# Chạy MediaMTX bằng Docker trên Ubuntu, tự lấy thư mục chứa script này làm config mount
# => không phụ thuộc đường dẫn máy/project, chạy được ở bất kỳ server nào clone repo
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Không tìm thấy $ENV_FILE. Copy .env.example thành .env và điền thông tin camera." >&2
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

RTSP_SOURCE="rtsp://${CAMERA_USER}:${CAMERA_PASS}@${CAMERA_IP}:554/Streaming/Channels/102"

sed "s#__CAMERA_SOURCE__#${RTSP_SOURCE}#" "$DIR/config.yml" > "$DIR/config.local.yml"

docker rm -f mediamtx >/dev/null 2>&1 || true

docker run -d --name mediamtx \
    --restart unless-stopped \
    -p 8889:8889 -p 8888:8888 -p 8554:8554 \
    -v "$DIR:/config" \
    bluenviron/mediamtx /config/config.local.yml

echo "MediaMTX started. Camera: $CAMERA_IP (config: $DIR/config.local.yml)"
