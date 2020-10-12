import cv2
import torch
import numpy as np
from model.model import *
import argparse

args=argparse.ArgumentParser()
args.add_argument('--device', type=str, default='cuda:0')
opt=args.parse_args()


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global mouse_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pos=[y,x]
        cv2.imshow("GAME",)

mouse_pos=[200,200]
model=torch.load("GameInNet.pkl")
cv2.namedWindow("GAME")
cv2.setMouseCallback("GAME", on_EVENT_LBUTTONDOWN)
hid_state=torch.tensor([[[8,8,17,17,17,17]]]).repeat((1, 30, 1)).float()
hid_state=hid_state.cuda()

model.cuda()

while 1:
    print(hid_state,torch.tensor([mouse_pos]).float().to(opt.device)/256*64)
    image,hid_out=model(hid_state,torch.tensor([mouse_pos]).float().to(opt.device)/256*64)
    hid_state=hid_out

    imgout = image.detach().cpu().squeeze().numpy()
    imgout = ((imgout / 2 + 0.5) * 255).astype(np.uint8)
    imgout = cv2.cvtColor(imgout, cv2.COLOR_GRAY2BGR)
    imgout[mouse_pos[0]//4, mouse_pos[1]//4, 2] = 255

    imgout = cv2.resize(imgout, (256, 256))
    cv2.imshow('GAME', imgout)
    cv2.waitKey(100)


