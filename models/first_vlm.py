import torch
import time
import psutil

from transformers import pipeline

MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

# Determine hardware
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {'Apple GPU' if device == 'mps' else 'CPU'}")

# Measure model loading time
start = time.perf_counter()

vlm = pipeline(
    task="image-text-to-text",
    model=MODEL_ID,
    device=device
)

load_time = time.perf_counter() - start

print(f"Model loading time: {load_time:.2f} seconds")

# Use a publicly available test image
image_url = (
    "https://huggingface.co/datasets/"
    "huggingface/documentation-images/"
    "resolve/main/bee.jpg"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_url},
            {
                "type": "text",
                "text": "Describe the objects in this image."
            }
        ]
    }
]

# Run inference
start = time.perf_counter()

result = vlm(
    text=messages,
    max_new_tokens=100,
    return_full_text=False
)

if device == "mps":
    torch.mps.synchronize()

inference_time = time.perf_counter() - start

print("\nMODEL OUTPUT:")
print(result)

print("\nBENCHMARK RESULTS")
print("-------------------------")
print(f"Model: {MODEL_ID}")
print(f"Loading time: {load_time:.2f} seconds")
print(f"Inference time: {inference_time:.2f} seconds")
print(f"System memory used: {psutil.virtual_memory().used / 1e9:.2f} GB")