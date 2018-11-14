import  torch
from    torch import nn
from    torch.nn import functional as F




class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out, keep_prob=0):
        super(Encoder, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 32)

        self._enc_mu = torch.nn.Linear(32, D_out)
        self._enc_log_sigma = torch.nn.Linear(32, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        # print(x.shape)

        return self._enc_mu(x), self._enc_log_sigma(x)




class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out, keep_prob=0):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


def get_ae(encoder, decoder, x):
    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    # y = torch.clamp(y, 1e-8, 1 - 1e-8)

    return y



def get_z(encoder, x):

    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    return z



def get_loss(encoder, decoder, x, x_target):
    """

    :param encoder:
    :param decoder:
    :param x: input
    :param x_hat: target
    :param dim_img:
    :param dim_z:
    :param n_hidden:
    :param keep_prob:
    :return:
    """
    batchsz = x.size(0)
    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    # y = torch.clamp(y, 1e-8, 1 - 1e-8)


    # loss
    # marginal_likelihood2 = torch.sum(x_target * torch.log(y) + (1 - x_target) * torch.log(1 - y)) / batchsz
    # marginal_likelihood = -F.binary_cross_entropy(y, x_target, reduction='sum') / batchsz
    marginal_likelihood = -torch.pow(x_target - y, 2).sum() / batchsz
    # print(marginal_likelihood2.item(), marginal_likelihood.item())

    KL_divergence = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batchsz

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, z, loss, marginal_likelihood, KL_divergence