import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os

log_file_path = '../Train_log/train_rec_32.log'

# Lưu trữ dữ liệu Train
train_data = defaultdict(lambda: {'acc': 0.0, 'count': 0})

# Mảng lưu trữ dữ liệu Validation
val_data = {}

current_epoch = 0 

with open(log_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # 1. BẮT DỮ LIỆU TRAIN
        if "epoch:" in line and "acc:" in line and "loss:" in line:
            epoch_match = re.search(r'epoch:\s*\[?(\d+)', line)
            acc_match = re.search(r'acc:\s*([\d.eE+-]+)', line)
            
            if epoch_match and acc_match:
                current_epoch = int(epoch_match.group(1))
                train_data[current_epoch]['acc'] += float(acc_match.group(1))
                train_data[current_epoch]['count'] += 1
                
        # 2. BẮT DỮ LIỆU VALIDATION (Lấy từ cur metric)
        if "cur metric" in line and "acc:" in line:
            acc_match = re.search(r'acc:\s*([\d.eE+-]+)', line)
            if acc_match and current_epoch > 0:
                # Ghi đè dòng eval cuối cùng của epoch
                val_data[current_epoch] = float(acc_match.group(1))

# Xử lý tính trung bình cho Train Data
train_epochs = sorted(train_data.keys())
train_accuracies = [train_data[ep]['acc'] / train_data[ep]['count'] for ep in train_epochs]

# Lấy dữ liệu Validation Data
val_epochs = sorted(val_data.keys())
val_accuracies = [val_data[ep] for ep in val_epochs]

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))

plt.plot(train_epochs, train_accuracies, label='Train Accuracy', color='tab:orange', marker='o', markersize=4, linestyle='-', alpha=0.8, linewidth=2)
if val_epochs:
    plt.plot(val_epochs, val_accuracies, label='Validation Accuracy', color='tab:blue', marker='s', markersize=4, linestyle='-', alpha=0.8, linewidth=2)

plt.title('So sánh Accuracy: Train vs Validation', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()

# Lưu ảnh
output_img = 'rec_train_val_acc_graph.png'
plt.savefig(output_img, dpi=300)
print(f"Đã vẽ và lưu biểu đồ tại: {os.path.abspath(output_img)}")
plt.show()