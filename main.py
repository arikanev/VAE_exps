import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision



# MNIST dataset
dataset = datasets.MNIST(root='../data/datasets/mnist',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=100,
                                          shuffle=True)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# We want to optimize N
def N_to_var(n):
	if torch.cuda.is_available():
		n = n.cuda()
	return Variable(n, requires_grad=True)

# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2))  # 2 for mean and variance.

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid())

    def reparameterize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = to_var(torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return z

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)  # mean and log variance.
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var

    def sample(self, z):
        return self.decoder(z)

vae = VAE()

if torch.cuda.is_available():
    vae.cuda()

optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
iter_per_epoch = len(data_loader)
data_iter = iter(data_loader)

targets = []
for epoch in range(50):
    for i, (images, _) in enumerate(data_loader):

        images = to_var(images.view(images.size(0), -1))
        out, mu, log_var = vae(images)

        targets.append(images)

        # Compute reconstruction loss and kl divergence
        # For kl_divergence, see Appendix B in the paper or http://yunjey47.tistory.com/43
        reconst_loss = F.binary_cross_entropy(out, images, size_average=False)
        kl_divergence = torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))

        # Backprop + Optimize
        total_loss = reconst_loss + kl_divergence
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "Reconst Loss: %.4f, KL Div: %.7f" 
                   %(epoch+1, 50, i+1, iter_per_epoch, total_loss.data[0], 
                     reconst_loss.data[0], kl_divergence.data[0]))



# After initial training, freeze the weights

for param in vae.parameters():
    param.requires_grad = False

# Generate uniform random tensors with range(0,1) == N
N = torch.FloatTensor(100, 784).uniform_(0, 1)

if torch.cuda.is_available():
    vae.cuda()

# Swap out vae weights with N as parameter
N = N_to_var(N.view(N.size(0), -1))
optimizer = torch.optim.Adam([N], lr=0.001)

N_to_save = N.view(N.size(0), 1, 28, 28)
torchvision.utils.save_image(N_to_save.data.cpu(), '../data/main/input_noise.png')

target_to_save = targets[0].view(targets[0].size(0), 1, 28, 28)
torchvision.utils.save_image(target_to_save.data.cpu(), '../data/main/target.png')

# Run NN with frozen weights, and N as input
for iter in range(10000):

        out, mu, log_var = vae(N)

        # Compute reconstruction loss and kl divergence
        # For kl_divergence, see Appendix B in the paper or http://yunjey47.tistory.com/43
        reconst_loss = F.binary_cross_entropy(out, targets[0], size_average=False)
        kl_divergence = torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))

        # Backprop + Optimize
        total_loss = reconst_loss + kl_divergence
        optimizer.zero_grad()
        total_loss.backward()

        # We update N
        optimizer.step()

        if iter % 100 == 0:
            print ("Iters/Steps: %d, Total Loss: %.4f, "
                   "Reconst Loss: %.4f, KL Div: %.7f" 
                   %(iter+1, total_loss.data[0], 
                     reconst_loss.data[0], kl_divergence.data[0]))
        if iter % 100 == 0:
            out_to_save = out.view(out.size(0), 1, 28, 28)
            torchvision.utils.save_image(out_to_save.data.cpu(), '../data/main/out_iter_%d.png' % (iter + 1))


# Save the reconstructed images
#reconst_images, _, _ = vae(N)
#reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
#torchvision.utils.save_image(reconst_images.data.cpu(), 
#    './data/reconst_images_N%d.png' %(epoch+1))
