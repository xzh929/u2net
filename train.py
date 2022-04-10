from data import EyeDataTrain, EyeDataTest
from net import U2NET
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
from torchvision.utils import save_image
import os

train_dataset = EyeDataTrain(r"D:\data\eye\training")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataset = EyeDataTest(r"D:\data\eye\test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

net = U2NET()
opt = optim.Adam(net.parameters())
loss_func = nn.BCELoss()

if os.path.exists(r"module/u2net.pth"):
    net.load_state_dict(torch.load(r"module/u2net.pth"))
    print("load module success!")
else:
    print("no module")


def all_loss_func(o0, o1, o2, o3, o4, o5, o6, tag):
    l0 = loss_func(o0, tag)
    l1 = loss_func(o1, tag)
    l2 = loss_func(o2, tag)
    l3 = loss_func(o3, tag)
    l4 = loss_func(o4, tag)
    l5 = loss_func(o5, tag)
    l6 = loss_func(o6, tag)
    loss = l0 + l1 + l2 + l3 + l4 + l5 + l6
    return loss, l0


img_save_path = r"test_img3"

for epoch in range(1000):
    train_loss = 0.
    for i, (data, tag) in enumerate(train_loader):
        net.train()
        # data, tag = data.cuda(), tag.cuda()
        o0, o1, o2, o3, o4, o5, o6 = net(data)
        loss, loss0 = all_loss_func(o0, o1, o2, o3, o4, o5, o6, tag)
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    epoch += 1
    print("epoch:{} train_loss:{}".format(epoch, avg_train_loss))

    for i, (data) in enumerate(test_loader):
        net.eval()
        # data = data.cuda()
        o0, o1, o2, o3, o4, o5, o6 = net(data)
        out_mean = torch.mean(o0)
        out = (o0 > out_mean).float()

    save_image(data, os.path.join(img_save_path, "{}.jpg".format(epoch)))
    save_image(out, os.path.join(img_save_path, "{}.png".format(epoch)))
    torch.save(net.state_dict(), r"module/u2net.pth")
    print("save success!")
