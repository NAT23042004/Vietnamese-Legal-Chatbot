import time
import requests

CHATBOT_URL = "http://localhost:8000/rag-agent"

questions = [
"Các sản phẩm, hàng hóa nhóm 2 được miễn chứng nhận hợp quy, công bố hợp quy cần đáp ứng những yêu cầu nào?",
"Các loại hình kiểm tra của đăng kiểm đối với tàu quân sự là gì? "
]

request_bodies = [{"text": q} for q in questions]

start_time = time.perf_counter()
outputs = [requests.post(CHATBOT_URL, json=data) for data in request_bodies]
end_time = time.perf_counter()

print(f"Run time: {end_time - start_time} seconds")