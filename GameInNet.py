import cv2
import torch



def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global mouse_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pos=[y,x]
        cv2.imshow("GAME",)

mouse_pos=[0,225]
model=torch.load("GAME.pkl")
cv2.namedWindow("GAME")
cv2.setMouseCallback("GAME", on_EVENT_LBUTTONDOWN)
hid_state=torch.tensor([0,0,17,17])
hid_state=hid_state.cuda()
model.cuda()
while 1:

    image,hid_out=model(hid_state,torch.tensor(mouse_pos))
    hid_state=hid_out

    cv2.imshow('GAME',image)
    cv2.waitKey(1)

    print('state-mouse', hid_state.detch().numpy(), mouse_pos.detch().numpy())
