import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from colorama import init, AnsiToWin32
import sys
from matplotlib import pyplot as plt
import de_Animator as Animator
import os
import model_cvit
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    #torchvision.transforms.Resize(40),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64～1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    # torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
    # ratio=(1.0, 1.0)),
    # torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                    [0.2023, 0.1994, 0.2010])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                    [0.2023, 0.1994, 0.2010])])



data_dir = 'C:/Users/Administrator/Desktop/DragFree/dataset_image/'

train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    # 读取按标签分好的数据集，其中每个子文件夹代表一个类别，并包含属于该类别的图像文件。
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', 'valid'),
    transform=transform_test)

batch_size = 64

train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
drop_last=True)






class c_vision_transformer(nn.Module):
    '''c_vit architecture'''

    def __init__(self, patch_size, num_hiddens, dropout, num_layers, key_size, query_size, value_size,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 mlp_hiddens, mlp_outs, use_bias=False, **kwargs):
        super(c_vision_transformer, self).__init__(**kwargs)
        self.conv = model_cvit.Conv_block(patch_size, num_hiddens)
        self.class_token = model_cvit.Class_token(num_hiddens)
        self.pos = model_cvit.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 model_cvit.EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                                    norm_shape, ffn_num_input, ffn_num_hiddens,
                                                    num_heads, dropout, use_bias))
        self.class_mlp = model_cvit.Class_token_mlp(num_hiddens, mlp_hiddens, mlp_outs)

    def forward(self, X, *args):
        X = self.conv(X)
        # 64*1*32*32
        X = self.class_token(X)
        # print(X[:1][:1])
        X = self.pos(X)
        self.attention_weights = [None] * len(self.blks)
        # 记录各头注意力权重的值
        for i, blk in enumerate(self.blks):
            X = blk(X)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        X = self.class_mlp(X[:,0,:])
        return X

path = 'C:/Users/Administrator/Desktop/vision-trans/C_ViT.pth'

patch_size = 8
num_hiddens = 64
dropout = 0.1
batch_size = 64
num_layers = 2
key_size, query_size, value_size = 64, 64, 64
norm_shape = [64]
ffn_num_input, ffn_num_hiddens, num_heads = 64, 128, 4

mlp_hiddens, mlp_outs = 16, 4

loss = nn.CrossEntropyLoss(reduction="none")

num_epochs, lr, wd = 5, 5e-4, 5e-4

lr_period, lr_decay= 4, 0.9

net = c_vision_transformer(patch_size, num_hiddens, dropout, num_layers, key_size, query_size, value_size,
                           norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, mlp_hiddens, mlp_outs)

device = torch.device('cuda:0')

def train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period,lr_decay):
    #trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=wd)
    trainer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=wd)
    #设置L2正则化系数wd
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    #在 StepLR 调度器中，学习率将在每个 step_size 个 epoch 之后进行衰减，衰减系数为 gamma。衰减后的学习率将用于下一个 epoch。
    num_batches, timer = len(train_iter), d2l.Timer()
    #num_batches代表批次数量
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')

    animator = Animator.Animator(xlabel='epoch', ylabel='accuracy/loss',xlim=[0, num_epochs],legend=legend)

    # xlim控制x轴的显示范围
    # net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    net.to(device)
    for param in net.parameters():
        param.to(device)
    # 将一个单 GPU 的模型并行地复制到多个 GPU 上，以实现数据的并行处理和模型的加速。
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3) # 三个数据分别是训练损失总和、训练准确度总和、样本数
        for i, (features, labels) in enumerate(train_iter):  #i是索引，feature是图像，labels是标签
            timer.start()
            l, acc = model_cvit.train_batch_ch13(net, features, labels, loss, trainer, device)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                # d2l.plot( epoch + (i + 1)/num_batches, (metric[0] / metric[2], metric[1] / metric[2],None),legend=legend )
                # d2l.plt.show()
                animator.add(epoch + (i + 1) / num_batches,(metric[0] / metric[2], metric[1] / metric[2],None))
            #每个周期间隔5绘制一个点，表示的分别为每个样本的平均损失，平均准确度

        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
            scheduler.step()
        measures = (f'train loss {metric[0] / metric[2]:.3f}, 'f'train acc {metric[1] / metric[2]:.3f}')

    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
        print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'f' examples/sec on {str(device)}')
    # plt.savefig("C:/Users/Administrator/Desktop/vision-trans/train_valid_acc.png", dpi=600)
    plt.pause(20)


# plt.ion()
#
# train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period, lr_decay)
# # torch.save(net.state_dict(), path)
# plt.ioff()


net.load_state_dict(torch.load(path))



for i, (features, labels) in enumerate(train_iter):
    X = features[0:1]
    y = labels
    break

X = net(X)



# for name, param in net.named_parameters():
#     print(name, param.shape)

enc_attention_weights = torch.cat(net.attention_weights, 0).reshape((num_layers, num_heads,
                                                -1, 17))

# print(enc_attention_weights.shape)
print(X)
X = nn.functional.softmax(X,dim=1)
print(X)

       # fig:[2,4,17,17]
model_cvit.show_heatmaps(
                enc_attention_weights.cpu(), xlabel='Key positions',
                ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
                figsize=(7, 3.5))



