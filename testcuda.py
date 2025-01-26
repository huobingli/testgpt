# import torch

# print(torch.__version__)
# my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cpu")
# print(my_tensor)
# torch.cuda.is_available()


# import torch
# if torch.cuda.is_available():
#     print("CUDA is available")
#     print(f"Current device: {torch.cuda.get_device_name(0)}")
# else:
#     print("CUDA is not available")


# import torch
# print(torch.cuda.is_available()) 


import torch

# 检查 CUDA 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")