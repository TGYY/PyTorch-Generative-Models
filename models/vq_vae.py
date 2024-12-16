import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

# class VectorQuantizer(nn.Module):
#     """
#     Reference:
#     [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
#     """
#     def __init__(self,
#                  num_embeddings: int,
#                  embedding_dim: int,
#                  beta: float = 0.25):
#         super(VectorQuantizer, self).__init__()
#         self.K = num_embeddings
#         self.D = embedding_dim
#         self.beta = beta

#         self.embedding = nn.Embedding(self.K, self.D)
#         limit = 3 ** 5
#         self.embedding.weight.data.uniform_(-1 / limit, 1 / limit)    # initialize embedding

#     def forward(self, latents: Tensor) -> Tensor:
#         flat_latents = latents.permute(0, 2, 3, 1).reshape(-1, self.D)  # [BHW x D]

#         # Compute L2 distance between latents and embedding weights
#         # dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
#         #        torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True) - \
#         #        2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

#         dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
#                torch.sum(self.embedding.weight.t() ** 2, dim=0, keepdim=True) - \
#                2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

#         # Get the encoding that has the min distance
#         encoding_inds = torch.argmin(dist, dim=1)  # [BHW, 1]
#         quantized_latents = F.embedding(
#             encoding_inds.view(latents.shape[0], *latents.shape[2:]), self.embedding.weight
#         ).permute(0, 3, 1, 2)

#         # # Convert to one-hot encodings
#         # device = latents.device
#         # encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
#         # encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

#         # # Quantize the latents
#         # quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
#         # quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

#         # Compute the VQ Losses
#         commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
#         embedding_loss = F.mse_loss(quantized_latents, latents.detach())

#         vq_loss = commitment_loss * self.beta + embedding_loss

#         # Add the residue back to the latents,  implements the “straight-through estimator”. 
#         # It passes gradients back to the original latent input while using the quantized values in the forward pass. 
#         quantized_latents = latents + (quantized_latents - latents).detach()

#         return quantized_latents, vq_loss  # [B x D x H x W]

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.ReLU(True),
                                      nn.Conv2d(in_channels, hidden_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(hidden_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, beta, use_ema, decay, epsilon):
        super().__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon
        self.beta = beta

        # Dictionary embeddings.
        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        # Exponential moving average of the cluster counts.
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 3, 1, 2)

        # See second term of Equation (3).
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
        else:
            dictionary_loss = None

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        # Straight-through gradient. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()


        if self.use_ema and self.training:
            with torch.no_grad():
                # See Appendix A.1 of "Neural Discrete Representation Learning".

                # Cluster counts.
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                # Updated exponential moving average of the cluster counts.
                # See Equation (6).
                self.N_i_ts(n_i_ts)

                # Exponential moving average of the embeddings. See Equation (7).
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)

                # This is kind of weird.
                # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                # and Equation (8).
                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon)
                    / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        return (quantized_x, dictionary_loss, commitment_loss, encoding_indices.view(x.shape[0], -1) )
    

class VQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 dataset_var: float = 1,
                 residual_hidden_channels: int = 32,
                 epsilon: float = 1e-5,
                 use_ema: bool = False,
                 decay: float = 0.99,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        self.dataset_var = dataset_var
        self.use_ema = use_ema

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 128]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.ReLU())
            )
            in_channels = h_dim

        # Encoder final layer
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                )
        )

        for _ in range(2):
            modules.append(ResidualLayer(in_channels, residual_hidden_channels, in_channels))

        # Pre vq
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                )
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(embedding_dim,
                                        num_embeddings,
                                        beta,
                                        use_ema,
                                        decay,
                                        epsilon)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                )
        )

        for _ in range(2):
            modules.append(ResidualLayer(hidden_dims[-1], residual_hidden_channels, hidden_dims[-1]))

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.ReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result] # for consistency with other model's code

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]
        (quantized_inputs, dictionary_loss, commitment_loss, encoding_indices) = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, dictionary_loss, commitment_loss]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        dictionary_loss = args[2]
        commitment_loss = args[3]

        recons_loss = F.mse_loss(recons, input) / self.dataset_var
        vq_loss = self.beta * commitment_loss

        if not self.use_ema:
            vq_loss += dictionary_loss
        
        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss.detach(),
                'VQ_Loss':vq_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]