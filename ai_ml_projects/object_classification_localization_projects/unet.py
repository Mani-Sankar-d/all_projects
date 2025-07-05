import torch 
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#Pascal_VOC dataset reqd
class DoubleConv(nn.Module):
  def __init__(self,inn_channels=3, out_channels=1):
    super(DoubleConv,self).__init__()
    self.double_conv = nn.Sequential([
        nn.Conv2d(in_channels, out_channels, 3,1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels,3,1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ])
    def forward(self,x):
      return self.double_conv(x)
    
class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
    super(UNet,self).__init__()
    self.down1 = DoubleConv(in_channels, features[0])
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.down2 = DoubleConv(in_channels, features[1])
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.down3 = DoubleConv(in_channels, features[256])
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.down4 = DoubleConv(in_channels, features[512])
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.bottleneck = DoubleConv(in_channels ,1024 ,3 ,1 )

    self.up4  = nn.ConvTranspose2d(1024,512,2,2)
    self.conv4  = nn.DoubleConv(1024,512)

    self.up3  = nn.ConvTranspose2d(512,256,2,2)
    self.conv3  = nn.DoubleConv(512,256)

    self.up2 = nn.ConvTranspose2d(256, 128, 2,2)
    self.conv2 = nn.DoubleConv(256,128)

    self.up1 = nn.ConvTranspose2d(128, 64)
    self.conv1 = nn.DoubleConv(128,64)

    self.final = nn.Conv2d(64, out_channels, kernel_size=1)
  
  def forward(self, x):
    d1 = self.down1(x)
    p1 = self.pool1(d1)

    d2 = self.down2(p1)
    p2 = self.pool2(d2)

    d3 = self.down3(p2)
    p3 = self.pool3(d3)

    d4 = self.down4(p3)
    p4 = self.pool4(d4)

    bn = self.bottleneck(p4)

    u4 = self.up4(bn)
    u4 = torch.cat([u4, d4], dim=1)
    u4 = self.conv4(u4)

    u3 = self.up3(u4)
    u3 = torch.cat([u3, d3], dim=1)
    u3 = self.conv3(u3)

    u2 = self.up2(u3)
    u2 = torch.cat([u2, d2], dim=1)
    u2 = self.conv2(u2)

    u1 = self.up1(u2)
    u1 = torch.cat([u1, d4], dim=1)
    u1 = self.conv1(u1)

    return self.final(u1)
    
class VOCDataset(Dataset):
  def __init__(self, root, ):
    self.root = root
    list_path = os.path.join(root, f'ImageSets/Main/{image_set}.txt')
    with open(list_path) as f:
      self.ids = [x.strip() for x in f.readlines()]
    self.transform = transform
    self.mask_transform = transforms.resize((256,256), interpolation = Image.NEAREST)
    self.img_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    def __len__(self):
      return len(self.ids)

    def __getitem__(self, idx):
      image_id = self.ids[idx]
      img_path = os.path.join(self.root, 'JPEGImages', f'{image_id}.jpg')
      mask_path = os.path.join(self.root, 'SegmentationClass', image_id+'.png')
      image = Image.open(img_path).convert('RGB')
      mask = Image.open(mask_path)

      image = self.img_transform(image)
      mask = self.mask_tranforms(mask)
      mask = torch.as_tensor(np.array(mask), dtype=torch.long)

      return image, mask


train_dataset = VOCDataset(root='E:\ml_projects\catvsdog\VOCdevkit\VOCdevkit\VOC2012')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
model = UNet(3,21).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs=10
model.train()
for epoch in range(epochs):
  epoch_loss = 0
  for images,masks in train_loader:
    images = images.to(device)
    masks = masks.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs,masks)
    loss.backward()
    optimizer.step()
    epoch_loss+=loss.item()
  print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}')


model.eval()
image, mask = train_dataset[0]
image = image.unsqueeze(0).to(device)
with torch.no_grad():
  output = model(image)
  pred = output.argmax(dim=1).cpu()[0]
torch.save(model.state_dict(), "unet_segmentation_voc.pth")
print("✅ Model weights saved!")

plt.subplot(1,2,1); plt.imshow(mask); plt.title('Ground Truth')
plt.subplot(1,2,2); plt.imshow(pred); plt.title('Prediction')
plt.show()
