import torch
import time
import matplotlib.pyplot as plt

# N values to test (from tiny to as large as your system can handle)
Ns = [1000, 2000, 5000, 8000, 10000]

cpu_times = []
gpu_times = []

for N in Ns:
    print(f"Testing N = {N} ...")

    # --- CPU O(n) FP16 ---
    try:
        cpu_vec = torch.ones(N * N, dtype=torch.float16, device="cpu")
        start = time.time()
        cpu_vec = cpu_vec * 2 + cpu_vec  # linear O(n)
        end = time.time()
        cpu_times.append(end - start)
    except RuntimeError:
        cpu_times.append(None)  # skip if too large

    # --- GPU O(n^3) FP16 ---
    try:
        a = torch.randn(N, N, dtype=torch.float16, device="cuda")
        b = torch.randn(N, N, dtype=torch.float16, device="cuda")
        torch.cuda.synchronize()
        start = time.time()
        c = torch.matmul(a, b)  # cubic O(n^3)
        torch.cuda.synchronize()
        end = time.time()
        gpu_times.append(end - start)
    except RuntimeError:
        gpu_times.append(None)  # skip if memory fails

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(Ns, cpu_times, "o-", label="CPU O(n) FP16", color="blue")
plt.plot(Ns, gpu_times, "s-", label="GPU O(n³) FP16", color="orange")
plt.xlabel("Matrix dimension N / Vector sqrt(size)")
plt.ylabel("Time (s)")
plt.title("CPU O(n) vs GPU O(n³) FP16 Benchmark")
plt.legend()
plt.grid(True)
plt.show()
