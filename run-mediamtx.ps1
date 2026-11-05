# Chạy MediaMTX bằng Docker, tự lấy thư mục chứa script này làm config mount
# => không phụ thuộc đường dẫn máy/project, chạy được ở bất kỳ máy nào clone repo

# Đổi IP/user/pass camera thì chỉ cần sửa trong file .env (cạnh script này) - không cần sửa gì khác
$envFile = Join-Path $PSScriptRoot ".env"
$envVars = @{}
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=][^=]*)\s*=\s*(.*)\s*$') {
            $envVars[$Matches[1].Trim()] = $Matches[2].Trim()
        }
    }
} else {
    throw "Không tìm thấy $envFile. Copy .env.example thành .env và điền thông tin camera."
}

$CameraIp   = $envVars['CAMERA_IP']
$CameraUser = $envVars['CAMERA_USER']
$CameraPass = $envVars['CAMERA_PASS']
$RtspSource = "rtsp://${CameraUser}:${CameraPass}@${CameraIp}:554/Streaming/Channels/102"

# Sinh config.local.yml từ config.yml (template) bằng cách thay __CAMERA_SOURCE__ bằng URL thật
$configDir = $PSScriptRoot
$templatePath = Join-Path $configDir "config.yml"
$renderedPath = Join-Path $configDir "config.local.yml"
(Get-Content $templatePath -Raw) -replace '__CAMERA_SOURCE__', $RtspSource | Set-Content -Path $renderedPath -NoNewline

docker rm -f mediamtx 2>$null | Out-Null

docker run -d --name mediamtx `
    --restart unless-stopped `
    -p 8889:8889 -p 8888:8888 -p 8554:8554 `
    -v "${configDir}:/config" `
    bluenviron/mediamtx /config/config.local.yml

Write-Host "MediaMTX started. Camera: $CameraIp (config: $renderedPath)"
