from PIL import Image
import torch
import torchvision

img = Image.open(r"D:\data\eye\training\images\22_training.tif").convert("RGB")
img = torchvision.transforms.ToTensor()(img)
# img = torchvision.transforms.ToPILImage()(img)
# img.show()
print(img.shape)
# a = torch.randn(1, 1, 224, 224)
# mean = torch.mean(a)
# print((a > mean).float())

