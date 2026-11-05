# Chạy MediaMTX bằng Docker, tự lấy thư mục chứa script này làm config mount
# => không phụ thuộc đường dẫn máy/project, chạy được ở bất kỳ máy nào clone repo

$configDir = $PSScriptRoot

docker rm -f mediamtx 2>$null | Out-Null

docker run -d --name mediamtx `
    -p 8889:8889 -p 8888:8888 -p 8554:8554 `
    -v "${configDir}:/config" `
    bluenviron/mediamtx /config/config.yml

Write-Host "MediaMTX started. Config: $configDir\config.yml"
