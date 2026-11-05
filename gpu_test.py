import torch
import time

print("PyTorch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\nTesting Apple GPU...")
else:
    device = torch.device("cpu")
    print("\nMPS unavailable. Testing CPU instead.")

# Create two matrices
a = torch.randn(2000, 2000, device=device)
b = torch.randn(2000, 2000, device=device)

# Multiply them and measure execution time
start = time.time()
c = torch.matmul(a, b)

if device.type == "mps":
    torch.mps.synchronize()

elapsed = time.time() - start

print("Device used:", device)
print("Result shape:", c.shape)
print(f"Execution time: {elapsed:.4f} seconds")
print("\nGPU test completed!")