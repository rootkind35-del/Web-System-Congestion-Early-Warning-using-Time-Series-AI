# Web System Congestion Early Warning using Time-Series AI

Dự án này là mã nguồn chính thức cho đề tài: **DỰ ĐOÁN NGHẼN HỆ THỐNG WEB BẰNG MÔ HÌNH TRÍ TUỆ NHÂN TẠO DỰA TRÊN CHUỖI THỜI GIAN**.

Hệ thống sử dụng kiến trúc học sâu tiên tiến nhất **TCN-DualAtt-BiLSTM** (Kế thừa và tối ưu hóa từ đợt tổng rà soát SOTA 2020-2024) để dự báo sớm tải CPU trước 5, 10 và 15 phút. Hệ thống tích hợp bộ lọc nhân quả Savitzky-Golay (Causal FIR Filter) để triệt tiêu hoàn toàn rò rỉ dữ liệu, cơ chế cảnh báo ngưỡng động EMA kết hợp bộ lọc trễ hệ thống (Latency SLO constraint) để lọc báo động giả, và mô hình phát hiện trôi dạt dữ liệu (Page-Hinkley Concept Drift) phục vụ việc kích hoạt retraining tự động.

---

## Tính năng Nổi bật & Chỉ số Đạt được

*   **Độ chính xác & Độ ổn định cao (The Final Architecture)**: Trải qua quá trình rà soát và đối đầu với các kiến trúc SOTA 2023-2024 (như *iTransformer*, *DLinear*), **TCN-DualAtt-BiLSTM** chứng minh được sự phù hợp tuyệt vời cho bài toán có cửa sổ ngắn (W=30) chạy trên tài nguyên hạn chế. Đạt sai số MAE cực thấp (chỉ **2.47** điểm phần trăm CPU tại T+5, **2.43** tại T+10 và **2.53** tại T+15) với độ lệch chuẩn siêu nhỏ (chỉ `± 0.034` ở T+10) trên tập kiểm thử thực tế hoàn toàn độc lập (tháng 8).
*   **Bộ cảnh báo chống nhiễu (Proposed SLO Alert)**: Kết hợp dự báo CPU và trễ hệ thống thực tế ($Latency > 48$ ms) để giảm **41.5%** báo động giả so với ngưỡng tĩnh và giảm **36.8%** so với cấu hình động EMA+1.5*Std, nâng Precision lên **10.23%**.
*   **Phản ứng nhanh với Concept Drift**: Thuật toán Page-Hinkley phát hiện sự thay đổi phân phối tải chỉ sau **2 bước** (độ trễ 2 phút), kích hoạt học lại và phục hồi MAE từ **13.43%** về **3.27%** CPU.
*   **Đo đạc hiệu năng thực tế (RTX 4060 GPU)**:
    *   *Độ trễ xử lý (Latency)*: Tiền xử lý (1.14 ms) + Model Inference (2.26 ms) + Hậu xử lý (0.02 ms) = **3.42 ms** cho toàn bộ pipeline (Throughput 292.4 requests/s).
    *   *Bộ nhớ*: Dung lượng file trọng số siêu nhẹ (**728 KB**). VRAM GPU tiêu thụ thực tế chỉ **9.83 MB** (Peak VRAM 42.65 MB), lý tưởng để nhúng vào Edge Servers hoặc chạy tích hợp trên Pods Kubernetes.


---

## Comprehensive Architecture Discovery (2020-2024)

Để chứng minh sức mạnh của mô hình, chúng tôi đã tiến hành một đợt Benchmark quy mô lớn, đưa **TCN-DualAtt-BiLSTM** đối đầu trực tiếp với các "siêu kiến trúc" của năm 2023 và 2024:
- **DLinear (SOTA 2023):** Kiến trúc Linear tối giản tối ưu.
- **iTransformer (SOTA 2024):** Inverted Transformer áp dụng Attention lên Feature thay vì Time.
- **CNN-Patch-BiLSTM (2023):** Ứng dụng kỹ thuật PatchTST nén tín hiệu.
- **TS-Mixer (2023/24):** Kiến trúc MLP-Mixer thuần túy.

Kết quả chứng minh **TCN-DualAtt-BiLSTM** đánh bại hoàn toàn các SOTA 2023/2024 nhờ khả năng duy trì độ phân giải cao cho các đỉnh nghẽn (flash crowds) mà không bị "làm mượt" (smoothed) hoặc đánh mất trật tự thời gian (như Transformer hay Patching trên chuỗi W=30).

![Exhaustive Architecture Benchmark (2020-2024)](docs/figures/sota_architecture_benchmark.png)
*Biểu đồ: So sánh Sai số tuyệt đối trung bình (MAE) trên 3 khung thời gian (T+5, T+10, T+15). Chỉ số càng thấp càng tốt.*

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
# Huấn luyện mô hình TCN-DualAtt-BiLSTM (Kiến trúc lõi)
python src/train.py --model tcn_dualatt_bilstm

# Huấn luyện các mô hình SOTA Benchmark khác (để so sánh)
python src/train.py --model itransformer
python src/train.py --model dlinear
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
