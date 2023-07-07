import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

from model import Model,Encoder,Decoder
from lib import Parse,roc,Mnisttox#,CustomImageDataset

parse = Parse()
opt = parse.args


cuda = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

n = 10 #number of test sample
x_dim  = 784
hidden_dim = 400
latent_dim = 200
lr = 1e-3

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
prev_decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim).to(device)
model = Model(Encoder=encoder, Decoder=decoder, device=device).to(device)

test_label = [0,1,2,3,4,5,6,7,8,9]
auroc_scores = []
diffs = []
roc_labels = []

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

optimizer = torch.optim.Adam(model.parameters(),
                             lr=opt.learning_rate)

if cuda:
    model.cuda()

img_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Resize(opt.image_size)
])

train_labels = [[1],[2],[3],[4],[5],[6],[7],[8],[9]]
anomaly_label = [0]
test_labels = [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6],
                    [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8,9]]#, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
for s in range(8):
    # train_dataset = CustomImageDataset(root_dir="../DATASET/MNIST/train", transform=img_transform)
    train_dataset = MNIST('./data', download=True, train=True, transform=img_transform)
    train_dataset = Mnisttox(train_dataset, train_labels[s])
    if s == 0:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=int(opt.batch_size/2), shuffle=True)

    losses = np.zeros(opt.num_epochs)
    model.train()
    if s != 0:
        p_decoder = torch.load(f'./save/weight/decoder_scenario{int(s - 1)}.pth')
        prev_decoder.load_state_dict(p_decoder)
        prev_decoder.eval()

    for epoch in range(opt.num_epochs):
        i = 0
        for img,_ in train_loader:
            if s != 0:
                img = torch.cat([img.to(device).view(img.size(0), -1),prev_decoder(torch.randn(int(opt.batch_size/2),latent_dim).to(device))],dim=0)

            if opt.flatten:
                x = img.view(img.size(0), -1)
            else:
                x = img

            if cuda:
                x = Variable(x).cuda()
            else:
                x = Variable(x)

            xhat, mean, log_var = model(x)
            loss = loss_function(x, xhat, mean, log_var)

            losses[epoch] = losses[epoch] * (i / (i + 1.)) + loss * (1. / (i + 1.))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1

        print('epoch [{}/{}], loss: {:.4f}'.format(
            epoch + 1,
            opt.num_epochs,
            loss))

    torch.save(decoder.state_dict(), f'./save/weight/decoder_scenario{s}.pth')

    # test_dataset = CustomImageDataset(root_dir="../DATASET/MNIST/test", transform=img_transform)
    test_dataset = MNIST('./data', download=True, train=False, transform=img_transform)
    test_dataset = Mnisttox(test_dataset,test_labels[s])
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)
    model.eval()

    j = 0
    for img,labels in test_loader:
        roc_label = [1 if labels[i] in [0] else 0 for i in range(img.size(0))]
        if opt.flatten:
            x = img.view(img.size(0), -1)
        else:
            x = img

        if cuda:
            x = Variable(x).cuda()
        else:
            x = Variable(x)

        xhat,_,_ = model(x)
        x = x.cpu().detach()
        xhat = xhat.cpu().detach()

        if opt.flatten:
            diff = np.abs(x - xhat).mean(axis=1)
        else :
            diff = np.abs(x.view(img.size(0), -1) - xhat.view(img.size(0), -1)).mean(axis=1)

        diffs = np.append(diffs,diff)
        roc_labels = np.append(roc_labels,roc_label)

        if j == 0:
            sample_x = x
            sample_xhat = xhat

        j += 1

    plt.figure(figsize=(12, 6))

    for i in range(n):
        # テスト画像を表示
        ax = plt.subplot(3, n, i + 1)
        x_a = sample_x[i].reshape(opt.n_channels,opt.image_size, opt.image_size).permute(1,2,0).numpy()
        plt.imshow(x_a)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 出力画像を表示
        ax = plt.subplot(3, n, i + 1 + n)

        x_b = sample_xhat[i].reshape(opt.n_channels,opt.image_size, opt.image_size).permute(1,2,0).numpy()
        plt.imshow(x_b)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if opt.flatten:
            diff_img = np.abs(x_a- x_b).mean(axis=2)
        else:
            diff_img = np.abs(sample_x[i] - sample_xhat[i]).mean(axis=0)
        diff = np.abs(sample_x[i].view(1, -1) - sample_xhat[i].view(1, -1)).mean(axis=1)

        ax = plt.subplot(3, n, i + 1 + n * 2)
        plt.imshow(diff_img.reshape(opt.image_size, opt.image_size),cmap="jet")
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        ax.set_xlabel('score = ' + str(diff))

    auroc_score = roc(roc_labels, diffs)
    print(roc(roc_labels,diffs))
    auroc_scores.append(auroc_score)
    plt.savefig(f"./save/result_sinario_{s}.png")
    plt.close()

plt.plot(range(1,9), auroc_scores, label='Scenario')
plt.xlabel('Scenario')
plt.ylabel('AUROC Score')
plt.legend()
plt.savefig("./save/result.jpg")
plt.show()


