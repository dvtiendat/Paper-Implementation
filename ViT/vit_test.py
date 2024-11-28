import torch

from VisionTransformer import ViT

def test_vit():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = torch.rand(1, 3, 224, 224).to(device)
    model = ViT().to(device)
    out = model(img)
    print('Out shape: ', out.shape)

test_vit()
