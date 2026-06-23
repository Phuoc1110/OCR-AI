import math
import random
import datetime

# Tên file xuất ra
output_file = 'fake_train_50.log'

# Thời gian bắt đầu giả lập
current_time = datetime.datetime(2026, 6, 7, 4, 51, 26)

total_epochs = 50
steps_per_epoch = 65
global_step = 0

best_hmean = 0.0
best_precision = 0.0
best_recall = 0.0
best_epoch = 0

with open(output_file, 'w', encoding='utf-8') as f:
    for epoch in range(1, total_epochs + 1):
        # 1. TẠO LOG QUÁ TRÌNH TRAIN (3 dòng đại diện cho mỗi epoch)
        for step_idx in [20, 40, 65]:
            global_step += (step_idx if step_idx == 20 else 20)
            current_time += datetime.timedelta(seconds=random.randint(15, 35))
            time_str = current_time.strftime("%Y/%m/%d %H:%M:%S")
            
            # Tính toán Loss giảm dần (Hàm mũ giảm)
            # Epoch 1 loss ~ 6.5, Epoch 50 loss ~ 1.5
            base_loss = 1.2 + 6.0 * math.exp(-0.065 * epoch)
            noise = random.uniform(-0.1, 0.1)
            loss = max(1.0, base_loss + noise)
            
            # Chia nhỏ loss ra 3 thành phần (tỷ lệ tương đối của DBNet)
            shrink_loss = loss * random.uniform(0.55, 0.60)
            threshold_loss = loss * random.uniform(0.20, 0.25)
            binary_loss = loss - shrink_loss - threshold_loss
            
            log_line = (f"[{time_str}] ppocr INFO: epoch: [{epoch}/50], global_step: {global_step}, "
                        f"lr: 0.001000, loss: {loss:.6f}, loss_shrink_maps: {shrink_loss:.6f}, "
                        f"loss_threshold_maps: {threshold_loss:.6f}, loss_binary_maps: {binary_loss:.6f}, "
                        f"loss_cbn: 0.000000, avg_reader_cost: {random.uniform(1.5, 2.5):.5f} s, "
                        f"avg_batch_cost: {random.uniform(2.5, 3.5):.5f} s, avg_samples: 16.0, "
                        f"ips: {random.uniform(4.5, 6.0):.5f} samples/s, eta: 1:00:00, "
                        f"max_mem_reserved: 7788 MB, max_mem_allocated: 6951 MB\n")
            f.write(log_line)
            
        # 2. TẠO LOG VALIDATION / EVAL CUỐI MỖI EPOCH
        current_time += datetime.timedelta(seconds=random.randint(40, 90))
        time_str = current_time.strftime("%Y/%m/%d %H:%M:%S")
        
        # Tính toán Metrics tăng dần (Đường cong Logarit / Cận trên)
        # Precision tiến về ~0.63, Recall tiến về ~0.55
        p = 0.64 - 0.60 * math.exp(-0.07 * epoch) + random.uniform(-0.01, 0.01)
        r = 0.56 - 0.54 * math.exp(-0.06 * epoch) + random.uniform(-0.01, 0.01)
        
        # Đảm bảo giá trị không vượt quá giới hạn
        p = min(max(p, 0.01), 0.99)
        r = min(max(r, 0.01), 0.99)
        
        # Công thức tính Hmean (F1-score)
        h = 2 * p * r / (p + r + 1e-6)
        
        # Cập nhật Best Metric
        if h > best_hmean:
            best_hmean = h
            best_precision = p
            best_recall = r
            best_epoch = epoch
            
        eval_line_1 = (f"[{time_str}] ppocr INFO: cur metric, precision: {p:.16f}, "
                       f"recall: {r:.16f}, hmean: {h:.16f}, fps: {random.uniform(19.5, 21.5):.16f}\n")
        eval_line_2 = (f"[{time_str}] ppocr INFO: best metric, hmean: {best_hmean:.16f}, "
                       f"is_float16: False, precision: {best_precision:.16f}, "
                       f"recall: {best_recall:.16f}, fps: 20.065827401848782, best_epoch: {best_epoch}\n")
                       
        f.write(eval_line_1)
        f.write(eval_line_2)

print(f"✅ Đã tạo thành công file log giả: {output_file}")
print(f"Chỉ số cuối cùng -> Precision: {p:.2f}, Recall: {r:.2f}, Hmean: {h:.2f}")