# Dự báo Nghẽn Hệ thống Web Bằng Mô Hình Học Sâu Dựa Trên Chuỗi Thời Gian
**(Web System Congestion Prediction using Time-Series Deep Learning)**

## 1. Abstract
(Tóm tắt nội dung nghiên cứu. *Sẽ hoàn thiện sau khi có kết quả thực nghiệm.*)

## 2. Introduction
Trong thời đại điện toán đám mây và kiến trúc microservices (Cloud-native), các hệ thống web thường xuyên phải đối mặt với các đỉnh tải (spikes) không thể lường trước. Các hệ thống giám sát cảnh báo truyền thống (như Prometheus Alertmanager) thường sử dụng ngưỡng tĩnh (Static Thresholds), dẫn đến tình trạng báo động giả (False Alarms) khi tài nguyên cao nhưng ứng dụng vẫn phản hồi tốt, hoặc bỏ sót lỗi khi hệ thống suy thoái chậm (Slow Degradation).

Nghiên cứu này đề xuất một kiến trúc cảnh báo sớm (Early Warning System) đột phá, tích hợp 3 thành phần lõi:
1. Cơ chế Ngưỡng Nghẽn Động (Dynamic Congestion Threshold) kết hợp giữa tài nguyên phần cứng và Service Level Objective (SLO).
2. Mô hình học sâu dự báo đa chân trời (Multi-Horizon Forecasting) sử dụng kiến trúc lai TCN-LSTM.
3. Kỹ thuật nén mô hình Float16 (FP16) trên nền tảng NVIDIA CUDA để giảm tối đa độ trễ suy luận (Inference Overhead).

## 3. Related Work
(Đánh giá các nghiên cứu liên quan về LSTM, Transformer, và RL trong Cloud-native. Tham khảo từ 12 bài báo đã đọc. *Sẽ hoàn thiện chi tiết sau.*)

## 4. Thực nghiệm và Kết quả (Experiments and Results)

### 4.1. Tập dữ liệu Đa năm (Multi-year Fusion Dataset)
Chúng tôi tổng hợp 1.576.800 mẫu dữ liệu kết hợp từ Wikipedia, NASA HTTP Logs và sự kiện Mega Sales (Shopee). Biểu đồ dưới đây minh họa tính chu kỳ và các điểm bùng nổ (Flash Crowds) được hệ thống học hỏi:

![Hình 1: Tổng quan Siêu tập dữ liệu (Dataset Overview)](file:///C:/Users/dhp01/OneDrive/Máy tính/New folder/docs/figures/fig2_dataset_overview.png)
*Hình 1: Trực quan hóa dữ liệu mô phỏng trong khoảng 2 tuần, làm nổi bật sự bùng nổ lưu lượng vào ngày 11.11.*

### 4.2. Môi trường Thực nghiệm
- Cấu hình phần cứng: NVIDIA RTX 4060 8GB, quá trình suy luận được nén với định dạng **FP16 (Half-precision)** qua công nghệ Tensor Cores.
- Cấu trúc tập huấn luyện: 1.1 triệu cửa sổ trượt (Sliding Windows).

### 4.3. Đánh giá Khả năng Hội tụ (Convergence)

![Hình 2: Biểu đồ đường cong hội tụ (Learning Curves)](file:///C:/Users/dhp01/OneDrive/Máy tính/New folder/docs/figures/fig1_learning_curves.png)
*Hình 2: So sánh Validation Loss (MSE) giữa SG-TCN-LSTM và BiLSTM-Attention sau 120 epochs.*

Chúng tôi tiến hành so sánh 2 kiến trúc mạng Neural qua đợt huấn luyện cường độ cao (120 epochs):

| Kiến trúc | Validation Loss Tốt nhất (MSE) | Thời gian Suy luận (Inference Time FP16) | Ghi chú |
| :--- | :--- | :--- | :--- |
| **SG-TCN-LSTM** | **0.000234** | ~238.0 ms | Bắt đặc trưng bằng bộ lọc SG và Tích chập TCN rất mạnh, nhưng tốc độ chậm hơn một chút. |
| **BiLSTM-Attention** | **0.000229** | ~189.2 ms | Khả năng chú ý (Self-Attention) giúp mô hình đạt Loss tốt nhất, dự báo rất nhanh và ổn định. |

### 4.4. Đánh giá Khả năng Dự báo Thực tế (Inference Accuracy)

Để kiểm chứng hiệu năng, chúng tôi lấy một đoạn dữ liệu kiểm thử (Test Set) không nằm trong tập huấn luyện để tiến hành chạy dự báo thời gian thực với mô hình BiLSTM-Attention. Các hình dưới đây so sánh tải CPU thực tế (đường màu đen) và tải CPU do mô hình dự báo (đường nét đứt màu cam) cho 3 chân trời thời gian:

![Hình 3: Dự báo T+5 Phút](file:///C:/Users/dhp01/OneDrive/Máy tính/New folder/docs/figures/fig3_prediction_T5.png)
*Hình 3: Kết quả dự báo tải CPU trước 5 phút (T+5).*

![Hình 4: Dự báo T+10 Phút](file:///C:/Users/dhp01/OneDrive/Máy tính/New folder/docs/figures/fig3_prediction_T10.png)
*Hình 4: Kết quả dự báo tải CPU trước 10 phút (T+10).*

![Hình 5: Dự báo T+15 Phút](file:///C:/Users/dhp01/OneDrive/Máy tính/New folder/docs/figures/fig3_prediction_T15.png)
*Hình 5: Kết quả dự báo tải CPU trước 15 phút (T+15). Có thể thấy dù dự báo xa 15 phút, mô hình vẫn bắt kịp xu hướng nhưng có độ trễ nhẹ so với T+5.*

## 5. Methodology
### 5.1. Định nghĩa Ngưỡng Nghẽn Động (Dynamic Congestion Threshold)
Sự "Nghẽn" thực sự xảy ra khi và chỉ khi hệ thống thỏa mãn đồng thời hai điều kiện: (1) Trải nghiệm người dùng suy giảm (Latency > SLO) và (2) Tài nguyên tiêu thụ vượt ngưỡng bất thường so với quá khứ gần (Dựa trên toán tử Exponential Moving Average).

### 5.2. Kiến trúc Mạng Nơ-ron TCN-LSTM (Multi-Horizon)
Mô hình kết hợp Temporal Convolutional Network (TCN) để trích xuất nhanh đặc trưng cục bộ (đỉnh tải) và Long Short-Term Memory (LSTM) để học xu hướng dài hạn. Tầng đầu ra được thiết kế để dự báo cùng lúc 3 mốc thời gian: 5, 10, và 15 phút trong tương lai.

### 4.3. Quản trị Concept Drift (Trôi dạt khái niệm)
Môi trường web có tính thay đổi cao. Khi hành vi người dùng hoặc bản cập nhật phần mềm làm lệch phân phối dữ liệu (Concept Drift), nghiên cứu áp dụng thuật toán Page-Hinkley để phát hiện và kích hoạt cơ chế Online Retraining trên một cửa sổ dữ liệu trượt (Sliding Window).

### 4.4. Tối ưu Suy luận (Inference Optimization)
Áp dụng cơ chế Automatic Mixed Precision (AMP) nén trọng số mô hình từ FP32 xuống FP16.

## 5. Experimental Setup
### 5.1. Dataset
(Mô tả tập dữ liệu sử dụng: Nguồn, số lượng mẫu, các trường tính năng. *Chờ cấu hình.*)

### 5.2. Hardware & Frameworks
Thực nghiệm được triển khai trên GPU NVIDIA RTX 4060 8GB VRAM, sử dụng PyTorch 2.x và CUDA Toolkit 12.x.

### 5.3. Evaluation Metrics
Các mô hình được đánh giá dựa trên:
- **Độ chính xác dự báo (Forecasting):** RMSE, MAE.
- **Độ chuẩn xác phân loại nghẽn (Classification):** Precision, Recall, F1-Score.

## 6. Results and Discussion
### 6.1. Hiệu suất mô hình (Forecasting Performance)
(So sánh TCN-LSTM với Baseline. *Chờ thực nghiệm.*)

### 6.2. Phân tích Ngưỡng Động vs Ngưỡng Tĩnh
(Chứng minh việc giảm thiểu False Alarms. *Chờ thực nghiệm.*)

### 6.3. Hiệu năng Suy luận (Inference Benchmark)
(Đo đạc ms/batch trên FP32 và FP16. *Chờ thực nghiệm.*)

## 7. Conclusion
(Kết luận đóng góp của nghiên cứu. *Sẽ hoàn thiện sau.*)

## References
(Danh mục tài liệu tham khảo.)
