import numpy as np
from scipy.signal import savgol_coeffs, lfilter

def apply_sg_filter(data_array: np.ndarray, window_length: int = 15, polyorder: int = 3):
    """
    Áp dụng bộ lọc Savitzky-Golay (SG) NHÂN QUẢ (Causal FIR Filter).
    Chỉ sử dụng dữ liệu quá khứ và hiện tại để làm mượt, tránh rò rỉ dữ liệu tương lai.
    
    :param data_array: Mảng 1D hoặc 2D chứa dữ liệu chuỗi thời gian (N, features).
    :param window_length: Độ dài cửa sổ lọc (phải là số lẻ).
    :param polyorder: Bậc đa thức dùng để nội suy (phải nhỏ hơn window_length).
    :return: Mảng dữ liệu đã làm mượt.
    """
    if window_length > len(data_array):
        window_length = len(data_array) if len(data_array) % 2 != 0 else len(data_array) - 1
        
    if window_length <= polyorder:
        return data_array
        
    # Tính toán các hệ số SG cho vị trí biên phải (pos = window_length - 1)
    coeffs = savgol_coeffs(window_length=window_length, polyorder=polyorder, pos=window_length - 1)
    # Lật ngược hệ số vì lfilter áp dụng b[0]*x[n] + b[1]*x[n-1] + ...
    b = coeffs[::-1]
    
    smoothed_data = np.zeros_like(data_array)
    
    if len(data_array.shape) == 2:
        for i in range(data_array.shape[1]):
            # lfilter thực hiện lọc nhân quả (causal filtering)
            smoothed_data[:, i] = lfilter(b, 1, data_array[:, i])
            # Xử lý các điểm khởi đầu (n < window_length) bằng cách giữ nguyên hoặc padding
            # Vì lfilter sẽ bị lệch pha hoặc bằng 0 tại những điểm đầu
            smoothed_data[:window_length, i] = data_array[:window_length, i]
    else:
        smoothed_data = lfilter(b, 1, data_array)
        smoothed_data[:window_length] = data_array[:window_length]
        
    return smoothed_data
