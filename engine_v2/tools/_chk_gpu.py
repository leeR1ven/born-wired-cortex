import torch, os
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("name", torch.cuda.get_device_name(0))
print("BORN_WIRED_DEVICE", os.environ.get("BORN_WIRED_DEVICE"))