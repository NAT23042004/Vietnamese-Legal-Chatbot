import asyncio
import time
import httpx

CHATBOT_URL = "http://localhost:8000/rag-agent"

async def make_async_post(url, data):
    timeout = httpx.Timeout(timeout=120)
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, timeout=timeout)
        return response

async def make_bulk_requests(url, data):
    tasks = [make_async_post(url, payload) for payload in data]
    responses = await asyncio.gather(*tasks)
    outputs = [r.json()["output"] for r in responses]
    return outputs

questions = [
"Các sản phẩm, hàng hóa nhóm 2 được miễn chứng nhận hợp quy, công bố hợp quy cần đáp ứng những yêu cầu nào?",
"Các loại hình kiểm tra của đăng kiểm đối với tàu quân sự là gì? "
]


request_bodies = [{"text": q} for q in questions]

start_time = time.perf_counter()
outputs = asyncio.run(make_bulk_requests(CHATBOT_URL, request_bodies))
end_time = time.perf_counter()

print(f"Run time: {end_time - start_time} seconds")