import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os

log_file_path = '../Train_log/train_rec_32.log'

# Sử dụng dictionary để cộng dồn loss và đếm số lượng log của mỗi epoch
epoch_data = defaultdict(lambda: {'total': 0.0, 'ctc': 0.0, 'sar': 0.0, 'count': 0})

with open(log_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if "ppocr INFO: epoch:" in line and "loss:" in line:
            # Lấy số epoch
            epoch_match = re.search(r'epoch:\s*\[?(\d+)', line)
            loss_match = re.search(r'loss:\s*([\d.]+)', line)
            ctc_match = re.search(r'CTCLoss:\s*([\d.]+)', line)
            sar_match = re.search(r'SARLoss:\s*([\d.]+)', line)
            
            if epoch_match and loss_match:
                ep = int(epoch_match.group(1))
                
                # Cộng dồn loss vào epoch tương ứng
                epoch_data[ep]['total'] += float(loss_match.group(1))
                epoch_data[ep]['ctc'] += float(ctc_match.group(1)) if ctc_match else 0
                epoch_data[ep]['sar'] += float(sar_match.group(1)) if sar_match else 0
                epoch_data[ep]['count'] += 1

# Tính giá trị trung bình cho mỗi epoch
epochs = sorted(epoch_data.keys())
total_losses = [epoch_data[ep]['total'] / epoch_data[ep]['count'] for ep in epochs]
ctc_losses = [epoch_data[ep]['ctc'] / epoch_data[ep]['count'] for ep in epochs]
sar_losses = [epoch_data[ep]['sar'] / epoch_data[ep]['count'] for ep in epochs]

plt.figure(figsize=(16, 6))

# Biểu đồ Total Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, total_losses, label='Total Loss', color='red', alpha=0.8, linewidth=2, marker='o', markersize=4)
plt.title('Tổng Loss trung bình (Total Loss) theo Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss (Average)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Biểu đồ Các thành phần Loss của Model Rec
plt.subplot(1, 2, 2)
plt.plot(epochs, ctc_losses, label='CTC Loss', color='blue', alpha=0.7, marker='s', markersize=4)
plt.plot(epochs, sar_losses, label='SAR Loss', color='green', alpha=0.7, marker='^', markersize=4)
plt.title('Các thành phần Loss của Mô hình Nhận dạng (Rec) theo Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss Component Value (Average)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()

output_img = 'rec_loss_graph.png'
plt.savefig(output_img, dpi=300)
print(f"Đã vẽ và lưu biểu đồ tại: {os.path.abspath(output_img)}")
plt.show()