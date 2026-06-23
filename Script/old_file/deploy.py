import os
from app import app, initialize_ocr

if __name__ == "__main__":
    initialize_ocr()
    port = int(os.environ.get("PORT", 5001))
    
    try:
        from pyngrok import ngrok
        # Tạo đường hầm (tunnel) để đưa localhost ra public internet
        public_url = ngrok.connect(port).public_url
        print("\n" + "="*60)
        print("🚀 ĐÃ KẾT NỐI INTERNET THÀNH CÔNG!")
        print(f"👉 COPY LINK NÀY ĐỂ DÁN VÀO RENDER: {public_url}")
        print("="*60 + "\n")
    except ImportError:
        print("\n" + "="*60)
        print("⚠️ CHƯA CÀI ĐẶT PYNGROK")
        print("Hãy mở terminal mới và chạy lệnh: pip install pyngrok")
        print("Sau đó chạy lại file này để có link public!")
        print("="*60 + "\n")

    app.run(host="0.0.0.0", port=port, debug=False)
