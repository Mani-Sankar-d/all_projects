import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)

])
data_dir = 'E:\ml_projects\VOCdevkit\VOCdevkit\VOC2012\JPEGImages'
image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
image_tensors = []
count=0
for fname in image_files:
    count+=1
    img_path = os.path.join(data_dir, fname)
    image = Image.open(img_path).convert("RGB")
    tensor_img = transform(image)
    image_tensors.append(tensor_img)
    if(count==5000):
        break

image_tensors = torch.stack(image_tensors)
# flat = nn.Flatten(start_dim=1)
# image_tensors = flat(image_tensors)
class CustomDataset(Dataset):
    def __init__(self,images):
        super().__init__()
        self.images = images
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = self.images[index]
        return image
dataset = CustomDataset(image_tensors)
dataloader = DataLoader(dataset,batch_size=32,shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, means, stds):
        noise = torch.randn_like(stds)
        return means + stds*noise

class Encoder(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,1,1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3,1,1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.norm3 = nn.BatchNorm2d(128)
        self.means = nn.Linear(28*28*128, 256)
        self.stds = nn.Linear(28*28*128, 256)
        self.relu = nn.ReLU()
        self.max = nn.MaxPool2d(2,2)
        self.flat = nn.Flatten(1)
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.norm1(x)
        x=self.max(x)
        
        x=self.conv2(x)
        x=self.relu(x)
        x=self.norm2(x)
        x=self.max(x)
        
        x=self.conv3(x)
        x=self.relu(x)
        x=self.norm3(x)
        x=self.max(x)
        x=self.flat(x)
    
        return self.means(x), torch.nn.functional.softplus(self.stds(x))

class Decoder(nn.Module):
    def __init__(self,img_dim):
        super().__init__()
        self.fcc1 = nn.Linear(256,28*28*128)
        self.relu = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(128,64,4,2,1)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64,32,4,2,1)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32,3,4,2,1)
        self.tanh = nn.Tanh()
    def forward(self,samples):
        x = self.fcc1(samples)
        x = self.relu(x)
        x = x.view(-1, 128, 28, 28)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        return self.tanh(x)

class Model(nn.Module):
    def __init__(self,encoder,decoder,sampler):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
    def forward(self, img):
        means, stds = self.encoder(img)
        samples = self.sampler(means,stds)
        new_img = self.decoder(samples)
        return new_img,means,stds

    
#controller block
encoder = Encoder(224)
decoder = Decoder(224)
sampler = Sampler()
model = Model(encoder=encoder,decoder=decoder,sampler=sampler).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
flat = nn.Flatten(1)
def get_loss(new_img,img,stds,means):
    recon_loss = torch.nn.functional.mse_loss(new_img, img, reduction='mean')
    stds = torch.clamp(stds, min=1e-3)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + torch.log(stds**2 + 1e-8) - means**2 - stds**2))
    return recon_loss+0.7*kl_loss

def train(epochs):
    for epoch in range(epochs):
        epoch_loss=0
        for img in dataloader:
            img = img.to(device)
            new_img,means,stds = model(img)
            loss = get_loss(new_img,img,stds,means)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        print(f"epoch:{epoch+1} loss:{epoch_loss}")

train(20)
torch.save(model.state_dict(), "vae_model.pth")