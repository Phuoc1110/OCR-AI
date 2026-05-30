import os
from app import app, initialize_ocr

if __name__ == "__main__":
    # Khởi tạo mô hình AI trước khi chạy server
    initialize_ocr()
    
    # Môi trường cloud (như Render) sẽ tự động gán biến PORT
    port = int(os.environ.get("PORT", 5001))
    print(f"[OCR Server Deploy] Starting OCR server on 0.0.0.0:{port}...", flush=True)
    
    # Quan trọng: host="0.0.0.0" để cho phép các request từ bên ngoài truy cập vào server
    app.run(host="0.0.0.0", port=port, debug=False)
