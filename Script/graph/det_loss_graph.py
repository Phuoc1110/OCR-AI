import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os

log_file_path = '../Train_log/train_det_50.log'

# Sử dụng dictionary để cộng dồn loss và đếm số lượng log của mỗi epoch
epoch_data = defaultdict(lambda: {'total': 0.0, 'shrink': 0.0, 'threshold': 0.0, 'binary': 0.0, 'count': 0})

with open(log_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if "epoch:" in line and "loss:" in line and "lr:" in line:
            # Lấy số epoch (PaddleOCR thường log dưới dạng: epoch: [1/100])
            epoch_match = re.search(r'epoch:\s*\[?(\d+)', line)
            loss_match = re.search(r'loss:\s*([\d.]+)', line)
            shrink_match = re.search(r'loss_shrink_maps:\s*([\d.]+)', line)
            threshold_match = re.search(r'loss_threshold_maps:\s*([\d.]+)', line)
            binary_match = re.search(r'loss_binary_maps:\s*([\d.]+)', line)
            
            if epoch_match and loss_match:
                ep = int(epoch_match.group(1))
                
                # Cộng dồn loss vào epoch tương ứng
                epoch_data[ep]['total'] += float(loss_match.group(1))
                epoch_data[ep]['shrink'] += float(shrink_match.group(1)) if shrink_match else 0
                epoch_data[ep]['threshold'] += float(threshold_match.group(1)) if threshold_match else 0
                epoch_data[ep]['binary'] += float(binary_match.group(1)) if binary_match else 0
                epoch_data[ep]['count'] += 1

if not epoch_data:
    print("Không tìm thấy dữ liệu loss trong file log.")
    exit()

# Tính giá trị trung bình cho mỗi epoch
epochs = sorted(epoch_data.keys())
total_losses = [epoch_data[ep]['total'] / epoch_data[ep]['count'] for ep in epochs]
shrink_losses = [epoch_data[ep]['shrink'] / epoch_data[ep]['count'] for ep in epochs]
threshold_losses = [epoch_data[ep]['threshold'] / epoch_data[ep]['count'] for ep in epochs]
binary_losses = [epoch_data[ep]['binary'] / epoch_data[ep]['count'] for ep in epochs]

plt.figure(figsize=(12, 8))

# Biểu đồ Total Loss
plt.plot(epochs, total_losses, label='Total Loss', color='red', linewidth=2, marker='o')
plt.plot(epochs, shrink_losses, label='Shrink Maps Loss', color='blue', linewidth=1.5, marker='x', linestyle='--')
plt.plot(epochs, threshold_losses, label='Threshold Maps Loss', color='green', linewidth=1.5, marker='s', linestyle='--')
plt.plot(epochs, binary_losses, label='Binary Maps Loss', color='purple', linewidth=1.5, marker='^', linestyle='--')

plt.title('Biểu đồ Hàm mất mát (Loss) của Model Detection theo Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right')

plt.tight_layout()

# Lưu biểu đồ thành file ảnh
output_img = 'det_loss_graph.png'
plt.savefig(output_img, dpi=300)
print(f"Đã vẽ và lưu biểu đồ tại: {os.path.abspath(output_img)}")
