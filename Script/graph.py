import re
import matplotlib.pyplot as plt

log_file_path = '../Model/model_epoch_50/content/PaddleOCR/output/db_mv3/train.log'

steps = []
total_losses = []
shrink_losses = []
threshold_losses = []
binary_losses = []

with open(log_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if "ppocr INFO: epoch:" in line and "loss:" in line:
            step_match = re.search(r'global_step:\s*(\d+)', line)
            loss_match = re.search(r'loss:\s*([\d.]+)', line)
            shrink_match = re.search(r'loss_shrink_maps:\s*([\d.]+)', line)
            threshold_match = re.search(r'loss_threshold_maps:\s*([\d.]+)', line)
            binary_match = re.search(r'loss_binary_maps:\s*([\d.]+)', line)
            
            if step_match and loss_match:
                steps.append(int(step_match.group(1)))
                total_losses.append(float(loss_match.group(1)))
                shrink_losses.append(float(shrink_match.group(1)) if shrink_match else 0)
                threshold_losses.append(float(threshold_match.group(1)) if threshold_match else 0)
                binary_losses.append(float(binary_match.group(1)) if binary_match else 0)

plt.figure(figsize=(16, 6))

# Biểu đồ Total Loss
plt.subplot(1, 2, 1)
plt.plot(steps, total_losses, label='Total Loss', color='red', alpha=0.8, linewidth=2)
plt.title('Tổng Loss (Total Loss) theo Global Step')
plt.xlabel('Global Step')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Biểu đồ Các thành phần Loss của DBNet
plt.subplot(1, 2, 2)
plt.plot(steps, shrink_losses, label='Shrink Maps Loss', color='blue', alpha=0.7)
plt.plot(steps, threshold_losses, label='Threshold Maps Loss', color='green', alpha=0.7)
plt.plot(steps, binary_losses, label='Binary Maps Loss', color='orange', alpha=0.7)
plt.title('Các thành phần Loss của DBNet')
plt.xlabel('Global Step')
plt.ylabel('Loss Component Value')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
