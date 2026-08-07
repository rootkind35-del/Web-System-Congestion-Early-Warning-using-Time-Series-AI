# Web System Congestion Early Warning using Time-Series AI

Dự án này là mã nguồn chính thức cho đề tài: **DỰ ĐOÁN NGHẼN HỆ THỐNG WEB BẰNG MÔ HÌNH TRÍ TUỆ NHÂN TẠO DỰA TRÊN CHUỖI THỜI GIAN**.

Hệ thống sử dụng kiến trúc học sâu tiên tiến (**BiLSTM-Attention** và baseline **SG-TCN-LSTM**) để dự báo sớm tải CPU trước 5, 10 và 15 phút. Hệ thống tích hợp bộ lọc nhân quả Savitzky-Golay (Causal FIR Filter) để triệt tiêu hoàn toàn rò rỉ dữ liệu, cơ chế cảnh báo ngưỡng động EMA kết hợp bộ lọc trễ hệ thống (Latency SLO constraint) để lọc báo động giả, và mô hình phát hiện trôi dạt dữ liệu (Page-Hinkley Concept Drift) phục vụ việc kích hoạt retraining tự động.

---

## Tính năng Nổi bật & Chỉ số Đạt được

*   **Độ chính xác & Độ ổn định cao**: BiLSTM-Attention đạt sai số MAE cực thấp (**2.34%** CPU tại T+5, **2.45%** tại T+10 và **2.66%** tại T+15) với độ lệch chuẩn nhỏ hơn **2.6 lần** so với TCN-LSTM baseline.
*   **Bộ cảnh báo chống nhiễu (Proposed SLO Alert)**: Kết hợp dự báo CPU và trễ hệ thống thực tế ($Latency > 100$ ms) để đạt **Precision 91.77%**, **Recall 81.71%**, **F1-score 86.45%** và giảm **83.5%** báo động giả (chỉ còn 133 cảnh báo nhầm so với 810 của ngưỡng tĩnh).
*   **Phản ứng nhanh với Concept Drift**: Thuật toán Page-Hinkley phát hiện sự thay đổi phân phối tải chỉ sau **1 bước** (độ trễ 1 phút), kích hoạt học lại và phục hồi MAE từ **15.02%** về **2.58%** CPU.
*   **Đo đạc hiệu năng thực tế (RTX 4060 GPU)**:
    *   *Độ trễ xử lý (Latency)*: Tiền xử lý (1.27 ms) + Model Inference (1.86 ms) + Hậu xử lý (0.02 ms) = **3.15 ms** cho toàn bộ pipeline.
    *   *Bộ nhớ*: Dung lượng file trọng số FP16 siêu nhẹ (**265 KB**). VRAM GPU tiêu thụ thực tế chỉ **9.65 MB** (Peak 42.28 MB), lý tưởng để nhúng vào Edge Servers hoặc chạy tích hợp trên Pods Kubernetes.

---

## Cài đặt (Installation)

```bash
# 1. Clone repository và cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## Hướng dẫn Sử dụng (Usage Workflow)

### 1. Chuẩn bị Dữ liệu & Khử rò rỉ (Data Preprocessing)
Sinh tập dữ liệu telemetry 3 năm (1.5 triệu dòng, sampling 1 phút) từ log NASA, Wikipedia và E-commerce:
```bash
python src/data/make_dataset.py
```
Tiền xử lý, áp dụng bộ lọc Savitzky-Golay nhân quả (FIR) và chia Train/Val/Test độc lập (fit scaler trên Train set):
```bash
python src/features/build_features.py
```

### 2. Huấn luyện Mô hình lặp lại 5 lần (Training)
Huấn luyện mô hình và lưu lịch sử loss (lặp lại 5 lần để đánh giá Mean ± Std, batch size 1024, 120 epochs):
```bash
# Huấn luyện mô hình đề xuất
python src/train.py --model bilstm_attention

# Huấn luyện mô hình baseline
python src/train.py --model sg_tcn_lstm
python src/train.py --model standard_lstm
```
Các file trọng số tốt nhất được lưu tại `models/best_[model_name].pth`.

### 3. Đánh giá Toàn diện Pipeline (Evaluation)
Chạy script đánh giá để tính các chỉ số của baseline truyền thống, thiết lập mô phỏng Page-Hinkley Drift, chạy ablation study cảnh báo và xuất kết quả ra file CSV:
```bash
python src/evaluate_pipeline.py
```
Biểu đồ drift được tự động lưu vào `docs/figures/concept_drift_analysis.png`.

### 4. Đo đạc phần cứng chi tiết (Hardware Profiling)
Đo latency cô lập của từng phân đoạn và tài nguyên RAM/VRAM GPU thực tế:
```bash
python src/profile_model.py
```

### 5. Cập nhật Biểu đồ trực quan (Visualization)
Tự động sinh các hình vẽ kết quả dự báo và lưu vào `docs/figures/`:
```bash
python src/visualization/visualize.py
```

### 6. Giao diện Dashboard (Web UI)
Bật FastAPI server để theo dõi Dashboard thời gian thực:
```bash
python src/app.py
```
Mở trình duyệt và truy cập: **[http://localhost:8000/dashboard](http://localhost:8000/dashboard)**

---

## Cấu trúc Thư mục

```text
project
 ┣ data           # Dữ liệu raw và processed
 ┣ docs           # Bản thảo bản viết (.md) và hình ảnh kết quả (.png)
 ┣ models         # File trọng số (.pth) và log lịch sử (.csv, .json)
 ┣ src
 ┃ ┣ data         # Script sinh và chèn sự kiện đột biến tải
 ┃ ┣ features     # Script tiền xử lý và bộ lọc SG nhân quả
 ┃ ┣ models       # Kiến trúc mạng PyTorch (baselines, BiLSTM-Attention)
 ┃ ┣ utils        # Thuật toán phụ trợ (Dynamic Threshold, PH Drift, Latency SLO)
 ┃ ┣ visualization# Vẽ các biểu đồ khoa học
 ┃ ┣ app.py       # Máy chủ giao diện Web Dashboard
 ┃ ┣ simulate.py  # Tạo luồng dữ liệu stream real-time
 ┃ ┣ train.py     # Script huấn luyện
 ┃ ┣ profile_model.py       # Đo đạc hiệu năng tài nguyên hệ thống
 ┃ ┗ evaluate_pipeline.py   # Chạy toàn bộ pipeline đánh giá baseline & drift
 ┣ web            # Giao diện tĩnh HTML/CSS/JS
 ┣ requirements.txt
 ┗ README.md
 ```

---
*Developed for Scientific Research on Web Congestion Prediction.*
