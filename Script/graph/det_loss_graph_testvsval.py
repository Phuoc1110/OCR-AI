import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os

log_file_path = '../Train_log/train_det_50.log'

train_data = defaultdict(lambda: {'total': 0.0, 'count': 0})
val_data = {}
current_epoch = 0

with open(log_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Bắt Train Loss
        if "epoch:" in line and "loss:" in line and "lr:" in line:
            epoch_match = re.search(r'epoch:\s*\[?(\d+)', line)
            loss_match = re.search(r'loss:\s*([\d.]+)', line)
            if epoch_match and loss_match:
                current_epoch = int(epoch_match.group(1))
                train_data[current_epoch]['total'] += float(loss_match.group(1))
                train_data[current_epoch]['count'] += 1
                
        # Bắt Validation Metrics (Lấy Hmean làm đại diện)
        if "hmean:" in line and "precision:" in line and "best metric" not in line:
            hmean_match = re.search(r'hmean:\s*([\d.]+)', line)
            if hmean_match and current_epoch > 0:
                val_data[current_epoch] = float(hmean_match.group(1))

if not train_data:
    print("Không tìm thấy dữ liệu loss trong file log.")
    exit()

# Xử lý tính trung bình cho Train Data
train_epochs = sorted(train_data.keys())
total_losses = [train_data[ep]['total'] / train_data[ep]['count'] for ep in train_epochs]

# Lấy dữ liệu Validation Data
val_epochs = sorted(val_data.keys())
val_hmeans = [val_data[ep] for ep in val_epochs]

# --------- BẮT ĐẦU VẼ BIỂU ĐỒ ---------
fig, ax1 = plt.subplots(figsize=(12, 6))

# Trục Y thứ nhất (Bên trái) cho Train Loss
color1 = 'tab:red'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Train Loss (Total Loss)', color=color1, fontsize=12)
ax1.plot(train_epochs, total_losses, color=color1, label='Train Loss', marker='o', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle=':', alpha=0.7)

# Trục Y thứ hai (Bên phải) cho Validation Hmean
if val_epochs:
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Validation Hmean (F-score)', color=color2, fontsize=12)  
    ax2.plot(val_epochs, val_hmeans, color=color2, label='Validation Hmean', marker='s', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Gom 2 legend lại với nhau
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
else:
    ax1.legend(loc='upper right')
    ax1.text(0.5, 0.5, 'Chưa có dữ liệu Validation trong file log', transform=ax1.transAxes, 
             ha='center', va='center', fontsize=14, color='gray', alpha=0.5)

plt.title('So sánh quá trình hội tụ: Train Loss vs Validation Hmean', fontsize=14, fontweight='bold')
fig.tight_layout()

# Lưu biểu đồ
output_img = 'det_compare_train_val.png'
plt.savefig(output_img, dpi=300)
print(f"Đã vẽ và lưu biểu đồ so sánh tại: {os.path.abspath(output_img)}")