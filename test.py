import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(1, 3, 224, 224).to(device)
print(x.device)  # 應該顯示 cuda:0