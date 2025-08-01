import torch
from torch import nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
tranformer = transforms.Compose(
    [transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
dataset = CIFAR10('./root',train=True, transform=tranformer,download=True)
dataloader = DataLoader(dataset,64,shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(512, 1, img_dim//16, 1,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.classify(x).view(x.size(0),-1)

class View(nn.Module):
    def __init__(self,n_f,x,y):
        super().__init__()
        self.n_f = n_f
        self.x = x
        self.y = y
    def forward(self, img):
        return img.view(img.size(0),self.n_f,self.x,self.y)
class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim):
        super().__init__()
        self.generate = nn.Sequential(
            nn.Linear(latent_dim, 512*(img_dim//16)*(img_dim//16)),
            nn.ReLU(True),
            View(512,img_dim//16,img_dim//16),

            nn.ConvTranspose2d(512,256,2,2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256,128,2,2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128,64,2,2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64,3,2,2),
            nn.Tanh()

        )
    def forward(self, x):
        return self.generate(x)
z=100
img_dim=64
lr=2e-4
generator = Generator(latent_dim=z, img_dim=img_dim).to(device)
discriminator = Discriminator(img_dim=img_dim).to(device)
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr)
generator_optimizer = torch.optim.Adam(generator.parameters(),lr=lr)

criterion = nn.BCELoss()
def train(epochs):
    for epoch in range(epochs):
        L_G = 0
        L_D = 0
        for real_imgs,_ in dataloader:
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.shape[0]
            vectors = torch.randn(b_size, z).to(device)
            fake_imgs = generator(vectors).detach()
            pred_label_real = discriminator(real_imgs)
            pred_label_fake = discriminator(fake_imgs)
            T_real = torch.ones_like(pred_label_real)
            T_fake = torch.zeros_like(pred_label_fake)
            loss_D = criterion(pred_label_real, T_real) + criterion(pred_label_fake,T_fake)
            discriminator_optimizer.zero_grad()
            loss_D.backward()
            discriminator_optimizer.step()
            L_D+=loss_D.item()
         ######################################################

            vectors = torch.randn(b_size, z).to(device)
            fake_imgs = generator(vectors).to(device)
            
            pred_label_fake = discriminator(fake_imgs)
            T_fake = torch.ones_like(pred_label_fake)
            loss_G = criterion(pred_label_fake, T_fake)
            generator_optimizer.zero_grad()
            loss_G.backward()
            generator_optimizer.step()
            L_G+=loss_G.item()
        print(f"Epoch {epoch} loss_G: {L_G}  loss_D: {L_D}")
    torch.save(generator.state_dict(),"generator.pth")
        
train(20)