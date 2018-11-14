import  torch
from    torch import nn
from    torch.nn import functional as F


class Encoder(nn.Module):


    def __init__(self, imgsz, n_hidden, n_output, keep_prob):
        super(Encoder, self).__init__()

        self.imgsz = imgsz
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.keep_prob = keep_prob

        self.net = nn.Sequential(
            nn.Linear(imgsz, n_hidden),
            nn.ELU(inplace=True),
            nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, n_output*2)

        )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        mu_sigma = self.net(x)


        # The mean parameter is unconstrained
        mean = mu_sigma[:, :self.n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + F.softplus(mu_sigma[:, self.n_output:])


        return mean, stddev



class Decoder(nn.Module):


    def __init__(self, dim_z, n_hidden, n_output, keep_prob):
        super(Decoder, self).__init__()

        self.dim_z = dim_z
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.keep_prob = keep_prob

        self.net = nn.Sequential(
            nn.Linear(dim_z, n_hidden),
            nn.Tanh(),
            nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, n_output),
            nn.Sigmoid()
        )

    def forward(self, h):
        """

        :param h:
        :return:
        """
        return self.net(h)



def init_weights(encoder, decoder):

    def init_(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    for m in encoder.modules():
        m.apply(init_)
    for m in decoder.modules():
        m.apply(init_)

    print('weights inited!')



def get_ae(encoder, decoder, x):
    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    y = torch.clamp(y, 1e-8, 1 - 1e-8)

    return y



def get_z(encoder, x):

    # encoding
    mu, sigma = encoder(x)
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
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    y = torch.clamp(y, 1e-8, 1 - 1e-8)


    # loss
    # marginal_likelihood2 = torch.sum(x_target * torch.log(y) + (1 - x_target) * torch.log(1 - y)) / batchsz
    marginal_likelihood = -F.binary_cross_entropy(y, x_target, reduction='sum') / batchsz
    # print(marginal_likelihood2.item(), marginal_likelihood.item())

    KL_divergence = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batchsz

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, z, loss, marginal_likelihood, KL_divergence


# # Gaussian MLP as encoder
# def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):
#     with tf.variable_scope("gaussian_MLP_encoder"):
#         # initializers
#         w_init = tf.contrib.layers.variance_scaling_initializer()
#         b_init = tf.constant_initializer(0.)
#
#         # 1st hidden layer
#         w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
#         b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
#         h0 = tf.matmul(x, w0) + b0
#         h0 = tf.nn.elu(h0)
#         h0 = tf.nn.dropout(h0, keep_prob)
#
#         # 2nd hidden layer
#         w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
#         b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
#         h1 = tf.matmul(h0, w1) + b1
#         h1 = tf.nn.tanh(h1)
#         h1 = tf.nn.dropout(h1, keep_prob)
#
#         # output layer
#         # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
#         wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
#         bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
#         gaussian_params = tf.matmul(h1, wo) + bo
#
#         # The mean parameter is unconstrained
#         mean = gaussian_params[:, :n_output]
#         # The standard deviation must be positive. Parametrize with a softplus and
#         # add a small epsilon for numerical stability
#         stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
#
#     return mean, stddev
#
#
# # Bernoulli MLP as decoder
# def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):
#     with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
#         # initializers
#         w_init = tf.contrib.layers.variance_scaling_initializer()
#         b_init = tf.constant_initializer(0.)
#
#         # 1st hidden layer
#         w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
#         b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
#         h0 = tf.matmul(z, w0) + b0
#         h0 = tf.nn.tanh(h0)
#         h0 = tf.nn.dropout(h0, keep_prob)
#
#         # 2nd hidden layer
#         w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
#         b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
#         h1 = tf.matmul(h0, w1) + b1
#         h1 = tf.nn.elu(h1)
#         h1 = tf.nn.dropout(h1, keep_prob)
#
#         # output layer-mean
#         wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
#         bo = tf.get_variable('bo', [n_output], initializer=b_init)
#         y = tf.sigmoid(tf.matmul(h1, wo) + bo)
#
#     return y
#
#
# # Gateway
# def autoencoder2(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):
#     # encoding
#     mu, sigma = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob)
#
#     # sampling by re-parameterization technique
#     z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
#
#     # decoding
#     y = bernoulli_MLP_decoder(z, n_hidden, dim_img, keep_prob)
#     y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)
#
#     # loss
#     marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
#     KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
#
#     marginal_likelihood = tf.reduce_mean(marginal_likelihood)
#     KL_divergence = tf.reduce_mean(KL_divergence)
#
#     ELBO = marginal_likelihood - KL_divergence
#
#     loss = -ELBO
#
#     return y, z, loss, -marginal_likelihood, KL_divergence
#
#
# def decoder(z, dim_img, n_hidden):
#     y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
#
#     return y
