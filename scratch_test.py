import requests
import time

url = "http://127.0.0.1:8000/api/query"

# Warmup run
print("Run 1 (gpu_rag):")
t0 = time.time()
res1 = requests.get(url, params={"q": "What is the address of the British Prime Minister?", "model": "gpu_rag", "dataset": "trivia_qa"}).json()
print(f"Latency: {res1['metrics']['total_ms']} ms")

# Run 2
print("\nRun 2 (cpu_rag):")
t0 = time.time()
res2 = requests.get(url, params={"q": "What is the address of the British Prime Minister?", "model": "cpu_rag", "dataset": "trivia_qa"}).json()
print(f"Latency: {res2['metrics']['total_ms']} ms")

# Run 3
print("\nRun 3 (gpu_rag):")
t0 = time.time()
res3 = requests.get(url, params={"q": "What is the address of the British Prime Minister?", "model": "gpu_rag", "dataset": "trivia_qa"}).json()
print(f"Latency: {res3['metrics']['total_ms']} ms")
