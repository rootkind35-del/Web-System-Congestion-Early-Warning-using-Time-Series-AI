# Web System Congestion Early Warning using Time-Series AI

Dự án này là mã nguồn chính thức cho đề tài: **DỰ ĐOÁN NGHẼN HỆ THỐNG WEB BẰNG MÔ HÌNH TRÍ TUỆ NHÂN TẠO DỰA TRÊN CHUỖI THỜI GIAN**.

Dự án sử dụng kiến trúc học sâu kép (**BiLSTM-Attention** và **SG-TCN-LSTM**) để dự báo sớm mức tải CPU của hệ thống máy chủ trước 5, 10 và 15 phút. Kết hợp với thuật toán phát hiện trôi dạt dữ liệu (Page-Hinkley) và Ngưỡng động trung bình trượt (EMA), hệ thống có khả năng đưa ra cảnh báo sớm (Early Warning) chính xác, cho phép hệ thống tự động scale tài nguyên (Auto-scaling) để chống sập web trong các sự kiện siêu lưu lượng như Mega Sale 11.11.

---

## Tính năng Nổi bật
*   **Mô phỏng 1.5 triệu Data points**: Dữ liệu được tổng hợp từ log truy cập NASA, Wikipedia và giả lập mua sắm E-commerce 3 năm.
*   **Dự báo 3 mốc thời gian cùng lúc**: Sử dụng Multi-horizon Prediction để xuất kết quả T+5, T+10, T+15 chỉ trong 1 lần chạy (Latency ~2.5ms).
*   **Siêu nhẹ (Ultra-lightweight)**: Mô hình chạy ở chế độ FP16, kích thước chỉ ~265KB, tốn < 10MB VRAM GPU.
*   **Giao diện Glassmorphism**: Dashboard mô phỏng Real-time siêu đẹp chạy qua FastAPI và Server-Sent Events (SSE).

---

## Cài đặt (Installation)

Yêu cầu máy tính có cài đặt Python 3.9+ (Khuyến nghị dùng GPU NVIDIA để có tốc độ Inference tốt nhất).

```bash
# 1. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## Hướng dẫn Sử dụng (Usage)

### 1. Chuẩn bị Dữ liệu (Data Preparation)
Sinh tập dữ liệu gốc 1.5 triệu dòng (mất khoảng 1-2 phút tùy CPU).
```bash
python src/data/make_dataset.py
```
Tiền xử lý, áp dụng bộ lọc Savitzky-Golay và chuẩn hóa dữ liệu ra file `.npz`.
```bash
python src/features/build_features.py
```

### 2. Huấn luyện Mô hình (Training)
Chạy quá trình học (Train) trên GPU. Mặc định sẽ train mô hình `bilstm_attention` với 120 Epochs.
```bash
python src/train.py --model bilstm_attention
```
Lịch sử đồ thị Loss sẽ được tự động lưu vào `models/history_bilstm_attention.csv`.

### 3. Trực quan hóa (Visualization)
Chạy script để tự động vẽ 5 biểu đồ chất lượng cao (Chuẩn IEEE) và lưu vào `docs/figures/`.
```bash
python src/visualization/visualize.py
```

### 4. Bật Giao diện Mô phỏng (Web Dashboard)
Mở máy chủ FastAPI để chạy mô phỏng thời gian thực sự kiện siêu bão 11.11.
```bash
python src/app.py
```
Sau đó, mở trình duyệt và truy cập: **[http://localhost:8000/dashboard](http://localhost:8000/dashboard)**

---

## Cấu trúc Thư mục

```text
project
 ┣ data           # Contains raw and processed data
 ┣ docs           # Research documents, manuscript drafts (.md) and figures
 ┣ models         # Pre-trained model weights (.pth)
 ┣ src
 ┃ ┣ data         # Data generation scripts (make_dataset, event_injector)
 ┃ ┣ features     # Preprocessing scripts (build_features, sg_filter)
 ┃ ┣ models       # PyTorch architectures (BiLSTM-Attention, SG-TCN-LSTM)
 ┃ ┣ utils        # Baseline algorithms (Thresholding EMA, Concept Drift)
 ┃ ┣ visualization# Scientific plotting scripts
 ┃ ┣ app.py       # FastAPI server for Web Dashboard
 ┃ ┣ simulate.py  # Real-time data stream generator
 ┃ ┣ train.py     # Training entry point
 ┃ ┗ profile_model.py # Hardware profiling script
 ┣ web            # Static frontend (HTML/CSS/JS) for Web Dashboard
 ┣ requirements.txt
 ┗ README.md
```

---
*Developed for Scientific Research on Web Congestion Prediction.*
