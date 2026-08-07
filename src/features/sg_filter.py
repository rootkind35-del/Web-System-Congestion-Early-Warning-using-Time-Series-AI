import numpy as np
from scipy.signal import savgol_filter

def apply_sg_filter(data_array: np.ndarray, window_length: int = 11, polyorder: int = 3):
    """
    Áp dụng bộ lọc Savitzky-Golay (SG) để khử nhiễu tần số cao của hệ điều hành.
    Làm mượt dữ liệu nhưng bảo toàn các đỉnh (spikes) quan trọng.
    
    :param data_array: Mảng 1D hoặc 2D chứa dữ liệu chuỗi thời gian (N, features).
    :param window_length: Độ dài cửa sổ lọc (phải là số lẻ).
    :param polyorder: Bậc đa thức dùng để nội suy (phải nhỏ hơn window_length).
    :return: Mảng dữ liệu đã làm mượt.
    """
    # Nếu window_length không hợp lệ so với kích thước dữ liệu
    if window_length > len(data_array):
        window_length = len(data_array) if len(data_array) % 2 != 0 else len(data_array) - 1
        
    if window_length <= polyorder:
        return data_array # Không lọc nếu window quá nhỏ
        
    smoothed_data = np.zeros_like(data_array)
    
    # Nếu là mảng 2D (N, features)
    if len(data_array.shape) == 2:
        for i in range(data_array.shape[1]):
            smoothed_data[:, i] = savgol_filter(data_array[:, i], window_length=window_length, polyorder=polyorder)
    else:
        # Nếu là mảng 1D
        smoothed_data = savgol_filter(data_array, window_length=window_length, polyorder=polyorder)
        
    return smoothed_data
