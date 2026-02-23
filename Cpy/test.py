import torch
import time

N = 1000

# --- CPU O(n) FP16 ---
cpu_vec = torch.ones(N * N, dtype=torch.float16, device="cpu")
start = time.time()
cpu_vec = cpu_vec * 2 + cpu_vec  # O(n)
end = time.time()
print("CPU O(n) time:", end - start)

# --- GPU O(n^3) FP16 ---
a = torch.randn(N, N, dtype=torch.float16, device="cuda")
b = torch.randn(N, N, dtype=torch.float16, device="cuda")

torch.cuda.synchronize()
start = time.time()
c = torch.matmul(a, b)  # O(N^3)
torch.cuda.synchronize()
end = time.time()
print("GPU O(n^3) time:", end - start)
