import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import sys



parser = argparse.ArgumentParser(description='VAE MNIST Example') # collect arguments passed to file
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test-batch-size', type=int, default=6000)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu") # Use NVIDIA CUDA GPU if available

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def latent_numpy(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        latent = z.detach().numpy()
        return latent




def VAE_loss_function(recon_x, x, mu, logvar):
    # TO DO: Implement reconstruction + KL divergence losses summed over all elements and batch

    # see lecture 12 slides for more information on the VAE loss function
    # for additional information on computing KL divergence
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114

    # Take the reconstruction loss as MSE loss of x and reconstructed x

    x = torch.reshape(x, list(recon_x.shape))
    recon_loss = nn.functional.mse_loss(recon_x, x)
    print(recon_loss)
    KLD = torch.sum(-0.5*(1 + logvar - mu ** 2 - torch.exp(logvar)))

    recon_weight = 1
    KLD_weight = 1e-5

    result = recon_weight * recon_loss + KLD_weight * KLD
    print(result)
    return result


def train(epoch, model):
    model.train()
    fig, axes = plt.subplots(figsize=(12*2, 2*2), dpi=120, nrows=1, ncols=6,
                             gridspec_kw={'width_ratios':[0.19,0.19,0.19,0.19,0.19,0.05]})
    train_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # print(list(data.shape))
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        latent_20D = model.latent_numpy(data)
        cmap = cm.get_cmap('Spectral', 11)
        latentscat = axes[0].scatter(latent_20D[:,0], latent_20D[:,1], c=targets, cmap=cmap, s=1.5, alpha=0.7)
        axes[0].set(xlabel='0', ylabel='1')
        latentscat = axes[1].scatter(latent_20D[:, 0], latent_20D[:, 2], c=targets, cmap=cmap, s=1.5, alpha=0.7)
        latentscat = axes[2].scatter(latent_20D[:, 1], latent_20D[:, 4], c=targets, cmap=cmap, s=1.5, alpha=0.7)
        latentscat = axes[3].scatter(latent_20D[:, 2], latent_20D[:, 3], c=targets, cmap=cmap, s=1.5, alpha=0.7)
        latentscat = axes[4].scatter(latent_20D[:, 2], latent_20D[:, 4], c=targets, cmap=cmap, s=1.5, alpha=0.7)
        for ax in axes:
            ax.axis('off')
        print(recon_batch)
        print(data)
        print(mu)
        print(logvar)

        loss = VAE_loss_function(recon_batch, data, mu, logvar)
        print(loss)
        sys.exit()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    cbar = plt.colorbar(latentscat, ax=axes[5])
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.91, wspace=0.01, hspace=0.01)
    fig.savefig("latent_stamps_train_epoch_{0}.png".format(epoch),bbox_inches='tight', pad_inches=0.01)


def test(epoch, model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += VAE_loss_function(recon_batch, data, mu, logvar).item()
            reparametrization = model.latent_numpy(data)

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.test_batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                reparametrizations = reparametrization
                all_targets = targets.detach().numpy()

                reparam1 = reparametrization[0]
                reparam2 = reparametrization[1]
                med_reparam = np.median([reparam1, reparam2], axis=0)
                decode_1 = model.decode(torch.from_numpy(reparam1))
                decode_2 = model.decode(torch.from_numpy(reparam2))
                decode_interpolation = model.decode(torch.from_numpy(med_reparam))
                fig, axes = plt.subplots(nrows=1, ncols=3, dpi=150)
                decode_1= decode_1.detach().numpy().reshape(28, 28)
                decode_1 = decode_1.reshape(28, 28)
                axes[0].imshow(decode_1)
                decode_2 = decode_2.detach().numpy().reshape(28, 28)
                decode_2 = decode_2.reshape(28, 28)
                axes[1].imshow(decode_2)
                decode_interpolation = decode_interpolation.detach().numpy().reshape(28, 28)
                decode_interpolation = decode_interpolation.reshape(28, 28)
                axes[2].imshow(decode_interpolation)
                for ax in axes:
                    ax.axis('off')
                # decode_interpolation(decode_interpolation.detach().numpy().reshape(28,28))
                fig.savefig("4_1_interpolation_triptych_epoch_{0}.png".format(epoch))

                fig, axispack = plt.subplots(figsize=(4.9, 4.1), nrows=4, ncols=5, dpi=150)
                for ii in range(5):
                    indices = np.where(targets == ii)[0]
                    idx=indices[0]
                    if len(indices)>1:
                        reparam1 = reparametrization[indices[0]]
                        reparam2 = reparametrization[indices[1]]
                        med_reparam = np.median([reparam1, reparam2], axis=0)
                        decode_1 = model.decode(torch.from_numpy(reparam1))
                        decode_2 = model.decode(torch.from_numpy(reparam2))
                        decode_interpolation = model.decode(torch.from_numpy(med_reparam))

                        medfig, medaxes = plt.subplots(nrows=1, ncols=5, dpi=200)
                        for ax in medaxes:
                            ax.axis('off')

                        medaxes[0].set_title('Input 1')
                        input_data = data[indices[0]].numpy()
                        input_data = input_data.reshape(28, 28)
                        medaxes[0].imshow(input_data)


                        decode_1 = decode_1.detach().numpy().reshape(28, 28)
                        decode_1 = decode_1.reshape(28, 28)
                        medaxes[1].imshow(decode_1)
                        medaxes[1].set_title('Output 1')

                        input_data = data[indices[1]].numpy()
                        input_data = input_data.reshape(28, 28)
                        medaxes[2].imshow(input_data)

                        medaxes[2].set_title('Input 2')
                        decode_2 = decode_2.detach().numpy().reshape(28, 28)
                        decode_2 = decode_2.reshape(28, 28)
                        medaxes[3].imshow(decode_2)
                        medaxes[3].set_title('Output 2')

                        medaxes[4].set_title('Interpolated')
                        decode_interpolation = decode_interpolation.detach().numpy().reshape(28, 28)
                        decode_interpolation = decode_interpolation.reshape(28, 28)
                        medaxes[4].imshow(decode_interpolation)
                        # decode_interpolation(decode_interpolation.detach().numpy().reshape(28,28))
                        medfig.savefig("4_1_interpolation_triptych_epoch_{0}_digit_{1}.png".format(epoch, ii),
                                       bbox_inches='tight', pad_inches=0.01)

                    input_data = data[idx].numpy()
                    input_data = input_data.reshape(28, 28)

                    axispack[0][ii].imshow(input_data)
                    axispack[0][ii].axis('off')

                    output = torch.reshape(recon_batch, [args.test_batch_size, 28, 28])[idx]
                    axispack[1][ii].imshow(output)
                    axispack[1][ii].axis('off')

                    idx = np.where(targets == ii + 5)[0][0]
                    input_data = data[idx].numpy()
                    input_data = input_data.reshape(28, 28)

                    axispack[2][ii].imshow(input_data)
                    axispack[2][ii].axis('off')
                    output = torch.reshape(recon_batch, [args.test_batch_size, 28, 28])[idx]
                    axispack[3][ii].imshow(output)
                    axispack[3][ii].axis('off')

                fig.subplots_adjust(left=0.1, bottom=0.01, right=0.99, top=0.91, wspace=0.01, hspace=0.01)
                fig.savefig("4_1_digits_epoch_{0}.png".format(int(epoch)), bbox_inches='tight', pad_inches=0.01)
            else:
                reparametrizations = np.vstack([reparametrizations, reparametrization])
                all_targets = np.hstack([all_targets, targets.detach().numpy()])
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    np.save("all_targets.npy", all_targets)
    np.save("reparametrizations.npy", reparametrizations)


if __name__ == "__main__":
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        train(epoch, model)
        test(epoch, model)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

