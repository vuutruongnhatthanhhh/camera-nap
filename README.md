window
.venv\Scripts\activate.bat
python ./Code.py

ubuntu
source .venv/bin/activate # nếu đang không ở trong venv
python Code.py

áp dụng code mới trên server
git pull origin main
sudo systemctl restart camera-ocr

## Luồng phát hiện và nhận diện

Worker camera và MediaMTX cùng dùng luồng chính 101. Web phát trực tiếp từ MediaMTX bằng HLS low-latency, không dùng MJPEG qua web server; player giữ khoảng cách 2 segment so với live edge, tối đa 4 segment trễ, tự tăng tốc/đọc lại khi bị chậm. Thanh tua bị ẩn. Khung ROI màu xanh được vẽ riêng theo đúng kích thước video; không vẽ box của từng kết quả OCR. Cấu hình MediaMTX giữ 7 segment (mức tối thiểu cho low-latency), 1 giây mỗi segment và part 200 ms. Worker nhận diện theo dõi ROI bằng background subtraction, threshold, morphology và contour, đồng thời căn chỉnh dịch chuyển toàn cảnh nhỏ để giảm ảnh hưởng rung camera. OCR chỉ bắt đầu sau khi vật thể hiện diện ổn định ít nhất 3 frame.

Mỗi phiên có Session ID riêng, dùng model PP-OCRv6 hỗ trợ tiếng Việt, lấy mẫu OCR cách nhau 0,2 giây và thử tối đa 6 mẫu. Kết quả được chốt sớm sau ít nhất 3 mẫu nếu chữ có ít nhất 2 phiếu; nếu chưa ổn định, worker tiếp tục lấy mẫu. Hệ thống chọn ảnh ROI sắc nét nhất bằng Laplacian Variance. Ảnh OCR giữ màu, cân bằng tương phản cục bộ và làm nét nhẹ; detector văn bản dùng giới hạn cạnh 1280 pixel.

Sau khi quét, hệ thống chuyển sang `WAIT_OBJECT_REMOVAL` và giữ nguyên kết quả cho đến khi ROI trống liên tục ít nhất 5 frame. Nếu sản phẩm mới được đặt đè/ngay sau sản phẩm cũ, hệ thống so sánh ảnh ROI và yêu cầu thay đổi ổn định 3 frame để mở phiên mới. Chuỗi OCR và dữ liệu MongoDB được chuẩn hóa giống nhau; hệ thống ưu tiên so mã `code`, sau đó so `productName`. Kết quả tra cứu hiện tại được giữ nguyên trong khi phiên mới đang chạy. Khi tab đóng, trang yêu cầu xóa cache; nếu không nhận heartbeat trong 5 giây, server tự xóa cache.
