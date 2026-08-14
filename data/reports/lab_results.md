# Bảng Kết Quả Thực Nghiệm Công Bằng (Automated Experiment Lab)

| Kiến trúc Mô hình | Chân trời | MSE | MAE | RMSE | R2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **StandardLSTM** | T+5 | 0.0174 | 0.0964 | 0.1321 | 0.9445 |
| **StandardLSTM** | T+10 | 0.0221 | 0.1120 | 0.1486 | 0.9297 |
| **StandardLSTM** | T+15 | 0.0269 | 0.1276 | 0.1640 | 0.9144 |
| **SG-TCN-LSTM** | T+5 | 0.0137 | 0.0852 | 0.1170 | 0.9564 |
| **SG-TCN-LSTM** | T+10 | 0.0214 | 0.1100 | 0.1461 | 0.9320 |
| **SG-TCN-LSTM** | T+15 | 0.0245 | 0.1209 | 0.1566 | 0.9220 |
| **BiLSTM-Attention** | T+5 | 0.0187 | 0.0962 | 0.1367 | 0.9405 |
| **BiLSTM-Attention** | T+10 | 0.0244 | 0.1126 | 0.1563 | 0.9222 |
| **BiLSTM-Attention** | T+15 | 0.0292 | 0.1297 | 0.1708 | 0.9072 |
| **TCN-DualAtt-BiLSTM** | T+5 | 0.0128 | 0.0825 | 0.1130 | 0.9593 |
| **TCN-DualAtt-BiLSTM** | T+10 | 0.0213 | 0.1103 | 0.1460 | 0.9322 |
| **TCN-DualAtt-BiLSTM** | T+15 | 0.0243 | 0.1194 | 0.1559 | 0.9227 |
