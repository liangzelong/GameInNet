import torch
import numpy as np
import cv2
import  time
import torch.nn as nn
import argparse

args=argparse.ArgumentParser()
args.add_argument('--mr_size', type=int, default=48)
args.add_argument('--slm_size', type=int, default=54)
args.add_argument('--channels', type=int, default=1)
args.add_argument('--latent_dim', type=int, default=100)
args.add_argument('--img_size', type=int, default=64)
args.add_argument('--gan_path', type=str, default='./weights/backup.pkl')
args.add_argument('--lstm_path', type=str, default='./weights/lstm1.pkl')
args.add_argument('--device', type=str, default='cuda:0')
opt=args.parse_args()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 16
        self.emb_labelx1 = nn.Embedding(opt.mr_size, opt.mr_size)
        self.emb_labelx2 = nn.Embedding(opt.mr_size, opt.mr_size)
        self.emb_labely1 = nn.Embedding(opt.slm_size, opt.slm_size)
        self.emb_labely2 = nn.Embedding(opt.slm_size, opt.slm_size)
        self.conv_dim_init=1024

        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim + opt.mr_size*2+opt.slm_size*2, self.conv_dim_init * self.init_size *self.init_size ))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.conv_dim_init),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.conv_dim_init, self.conv_dim_init//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.conv_dim_init//2, 0.8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.conv_dim_init//2, self.conv_dim_init//4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.conv_dim_init//4, 0.8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.conv_dim_init//4, self.conv_dim_init//8, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.conv_dim_init//8, 0.8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.conv_dim_init//8, self.conv_dim_init//16, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.conv_dim_init//16, 0.8),
            nn.ReLU(),
            nn.Conv2d(self.conv_dim_init//16, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z,label):

        emb_labelx1 = self.emb_labelx1(label[:,0])
        emb_labelx2 = self.emb_labelx2(label[:,1])
        emb_labely1 = self.emb_labely1(label[:,2])
        emb_labely2 = self.emb_labely2(label[:,3])


        cat_latent = torch.cat((z, emb_labelx1,emb_labelx2,emb_labely1,emb_labely2), dim=-1)
        out = self.l1(cat_latent)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class state_transf(nn.Module):
    def __init__(self,hid_len=60):
        super(state_transf,self).__init__()

        self.l0=nn.Linear(2,30*3)
        self.lstm=nn.LSTM(9,64,2,bidirectional=True,batch_first=True)
        self.l1=nn.Linear(30*64*2,1024)
        self.l2=nn.Linear(1024,512)
        self.l3=nn.Linear(512,5)
        self.sigmoid=nn.Sigmoid()

    def forward(self, old,mouse):

        mouse=self.l0(mouse/54)
        mouse=mouse.view(-1,30,3)
        x=torch.cat([old/54,mouse],dim=2)
        x=self.lstm(x)[0]
        x=x.reshape(-1,30*64*2)
        x=self.l1(x)
        x=self.l2(x)
        x=self.l3(x)
        o1=self.sigmoid(x[:,:4])*54
        o2=self.sigmoid(x[:,4:])

        return o1,o2

class game(nn.Module):
    def __init__(self):
        super(game,self).__init__()
        self.state_transfor=state_transf()
        self.state_transfor.load_state_dict(torch.load(opt.lstm_path,map_location=opt.device))
        self.generate_image=Generator()
        self.generate_image.load_state_dict(torch.load(opt.gan_path,map_location=opt.device)['gen_model'])
        self.noise=torch.rand(1, 100).to(opt.device)
        self.state_temp=(torch.rand(1,30,6)*30).to(opt.device)

    def forward(self, state,mouse):
        o1,o2=self.state_transfor(state,mouse)
        att = torch.cat([o1.clone(), mouse.clone()], dim=1).unsqueeze(0)
        o1[:,:2]=o1[:,:2]-8
        o1[:, 2:] = o1[:, 2:] - 5
        img=self.generate_image(self.noise,o1.long())
        self.state_temp[0,:-1,:]=state[:,1:,:].clone()
        self.state_temp[0,-1,:]=att.clone()
        return img,self.state_temp


if __name__ == '__main__':

    state0 = [[[8, 8, 17, 17, 17, 17]]]
    state0 = torch.tensor(state0).repeat((1, 30, 1)).float()
    state0 = state0.to(opt.device)
    mouse = torch.tensor([[52,52]]).float()
    mouse = mouse.to(opt.device)
    imgnet=torch.load('../GameInNet.pkl')
    while 1:
        if torch.rand(1) < 0.1:
            mouse = torch.rand(1, 2).to(opt.device) * 34 + 30
        gen_img, state_new = imgnet(state0, mouse)
        # print(state0,mouse)
        state0 = state_new
        imgout = gen_img.detach().cpu().squeeze().numpy()
        imgout = ((imgout / 2 + 0.5) * 255).astype(np.uint8)
        imgout = cv2.cvtColor(imgout, cv2.COLOR_GRAY2BGR)

        imgout[int(mouse[0, 0].detach().cpu().item()), int(mouse[0, 1].detach().cpu().item()), 2] = 255

        imgout = cv2.resize(imgout, (256, 256))

        cv2.imshow('GAME', imgout)
        cv2.waitKey(1)
