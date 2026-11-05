import torch

print("PyTorch:", torch.__version__)
print("Testing MPS matrix multiplication...")

device = "mps"

a = torch.randn(512, 512, device=device)
b = torch.randn(512, 512, device=device)
c = a @ b

torch.mps.synchronize()
print("Matrix multiplication: SUCCESS")

print("Testing MPS neural network layer...")

layer = torch.nn.Linear(512, 256).to(device)
x = torch.randn(10, 512, device=device)
y = layer(x)

torch.mps.synchronize()
print("Linear layer: SUCCESS")
print("Output shape:", y.shape)
