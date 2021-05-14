import requests
from transformers import AutoTokenizer
import time
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
n_requests = 5
cpu_url = "http://127.0.0.1:8080/predict"
gpu_url = "http://127.0.0.1:8080/predict"


list_cpu_times = []
list_gpu_times = []

list_sequence_length = [8, 16, 32, 64, 128, 256, 512]
list_ticks = [i for i in range(len(list_sequence_length))]

for power in list_sequence_length:
    string = "ana " * (power - 2)
    cpu_total_time = 0
    gpu_total_time = 0

    for i in range(n_requests):
        start = time.time()
        request = requests.post(
            cpu_url,
            json={"data": string}
        )
        end = time.time()

        cpu_total_time = cpu_total_time + (end - start)

        start = time.time()
        request = requests.post(
            gpu_url,
            json={"data": string}
        )
        end = time.time()

        gpu_total_time = gpu_total_time + (end - start)

    print(f"response time for sequence of size {len(tokenizer.encode(string))}: "
          f"CPU - {cpu_total_time / n_requests}"
          f"GPU - {gpu_total_time / n_requests}")

    list_cpu_times.append(cpu_total_time / n_requests)
    list_gpu_times.append(gpu_total_time / n_requests)

plt.plot(list_ticks, list_cpu_times, label="CPU latency")
plt.plot(list_ticks, list_gpu_times, label="GPU latency")
plt.xticks(list_ticks, labels=list_sequence_length)
plt.scatter(list_ticks, list_cpu_times)
plt.scatter(list_ticks, list_gpu_times)
plt.xlabel("Sequence length")
plt.ylabel("Latency")
plt.legend()

plt.show()