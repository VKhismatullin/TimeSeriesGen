####
# The first part is my implementation of RealNVP
# It is based on other models from the section below
####

import numpy as np
import torch
import torch.nn as nn


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        
        # Create a flow
        # nets:  a function that returns a PyTorch neural network, e.g., nn.Sequential, s = nets(), s: dim(X) -> dim(X)
        # nett:  a function that returns a PyTorch neural network, e.g., nn.Sequential, t = nett(), t: dim(X) -> dim(X)
        # mask:  a torch.Tensor of size #number_of_coupling_layers x #dim(X)
        # prior: an object from torch.distributions e.g., torch.distributions.MultivariateNormal
        
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])
        self.A = torch.nn.ModuleList([ActNorm(64) for _ in range(len(mask))])
        
    def g(self, z):
        # Compute and return g(z) = x, 
        #    where self.mask[i], self.t[i], self.s[i] define a i-th masked coupling layer   
        # z: a torch.Tensor of shape batchSize x 1 x dim(X)
        # return x: a torch.Tensor of shape batchSize x 1 x dim(X)
        for i in range(len(self.t))[::-1]:
            h_1 = self.mask[i] * z
            x = h_1 + (1 - self.mask[i]) *(z - self.t[i](h_1)) / torch.exp(self.s[i](h_1))
            x = self.A[i].inverse(x)[0] # leave Jacob
            z = x
        
        return x
    
    def decode(self, z):
        return self.g(z)

    def f(self, x):
        # Compute f(x) = z and log_det_Jakobian of f, 
        #    where self.mask[i], self.t[i], self.s[i] define a i-th masked coupling layer   
        # x: a torch.Tensor, of shape batchSize x dim(X), is a datapoint
        # return z: a torch.Tensor of shape batchSize x dim(X), a hidden representations
        # return log_det_J: a torch.Tensor of len batchSize
        log_det_J = 0 
        for i in range(len(self.t)):
            x, Act_J = self.A[i](x)  # save Jacob
            h_1 = self.mask[i] * x 
            fwd = self.s[i](h_1)
            z = h_1 +  (1-self.mask[i]) * (x * torch.exp(fwd) + self.t[i](h_1))
            
            log_det_J += ((1-self.mask[i]) * fwd).sum(dim=1) + Act_J
            x = z

        return z, log_det_J
    
    def encode(self, z):
        return self.f(z)[0]
    
    def log_prob(self, x, cent=None):
        # Compute and return log p(x)
        # using the change of variable formula and log_det_J computed by f
        # return logp: torch.Tensor of len batchSize
        z, log_det_J = self.f(x)
        if cent is None:
            logp = self.prior.log_prob(z) + log_det_J
        else:
            logp = self.prior.log_prob(z-cent) + log_det_J
        return logp
        
    def sample(self, batchSize): 
        # Draw and return batchSize samples from flow using implementation of g
        # return x: torch.Tensor of shape batchSize x 1 x dim(X)
        z = self.prior.sample((batchSize,))
        x = self.g(z)
        return x.reshape(x.shape[0], 1, x.shape[1])

def get_mask (dim, number_of_coupling_layers):
    # 1. Create a random permutation such that all numbers in the same row are unique and random*
    permutation = np.random.permutation(np.arange(number_of_coupling_layers * dim)).reshape(number_of_coupling_layers, dim)
    # 2. calculate the median of the array row-wise and leave only the numbers higher than the median
    # This way if dim_X = n we will always have [n / 2] '1' elements
    median = np.median(permutation, axis=1)
    mask = permutation > median.reshape(-1, 1)
    return mask.astype(int)



####
# Some of this code comes from https://github.com/ikostrikov/pytorch-flows/tree/master
# Big thanks to the team!
####



class Flow(nn.Module):
    """
    Generic class for flow functions
    """

    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        Args:
          z: input variable, first dimension is batch dim
        Returns:
          transformed z and log of absolute determinant
        """
        raise NotImplementedError("Forward pass has not been implemented.")

    def inverse(self, z):
        raise NotImplementedError("This flow has no algebraic inverse.")


class Reverse(Flow):
    """
    Switches the forward transform of a flow layer with its inverse and vice versa
    """

    def __init__(self, flow):
        """Constructor
        Args:
          flow: Flow layer to be reversed
        """
        super().__init__()
        self.flow = flow

    def forward(self, z):
        return self.flow.inverse(z)

    def inverse(self, z):
        return self.flow.forward(z)


class Composite(Flow):
    """
    Composes several flows into one, in the order they are given.
    """

    def __init__(self, flows):
        """Constructor
        Args:
          flows: Iterable of flows to composite
        """
        super().__init__()
        self._flows = nn.ModuleList(flows)

    @staticmethod
    def _cascade(inputs, funcs):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = torch.zeros(batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs):
        funcs = self._flows
        return self._cascade(inputs, funcs)

    def inverse(self, inputs):
        funcs = (flow.inverse for flow in self._flows[::-1])
        return self._cascade(inputs, funcs)


def zero_log_det_like_z(z):
    return torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

class Split(Flow):
    """
    Split features into two sets
    """

    def __init__(self, mode="channel"):
        """Constructor
        The splitting mode can be:
        - channel: Splits first feature dimension, usually channels, into two halfs
        - channel_inv: Same as channel, but with z1 and z2 flipped
        - checkerboard: Splits features using a checkerboard pattern (last feature dimension must be even)
        - checkerboard_inv: Same as checkerboard, but with inverted coloring
        Args:
         mode: splitting mode
        """
        super().__init__()
        self.mode = mode

    def forward(self, z):
        if self.mode == "channel":
            z1, z2 = z.chunk(2, dim=1)
        elif self.mode == "channel_inv":
            z2, z1 = z.chunk(2, dim=1)
        elif "checkerboard" in self.mode:
            n_dims = z.dim()
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z.size(n_dims - i))]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z.size(n_dims - i))]
            cb = cb1 if "inv" in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
            cb = cb.to(z.device)
            z_size = z.size()
            z1 = z.reshape(-1)[torch.nonzero(cb.view(-1), as_tuple=False)].view(
                *z_size[:-1], -1
            )
            z2 = z.reshape(-1)[torch.nonzero((1 - cb).view(-1), as_tuple=False)].view(
                *z_size[:-1], -1
            )
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        if self.mode == "channel":
            z = torch.cat([z1, z2], 1)
        elif self.mode == "channel_inv":
            z = torch.cat([z2, z1], 1)
        elif "checkerboard" in self.mode:
            n_dims = z1.dim()
            z_size = list(z1.size())
            z_size[-1] *= 2
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z_size[n_dims - i])]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z_size[n_dims - i])]
            cb = cb1 if "inv" in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
            cb = cb.to(z1.device)
            z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z = cb * z1 + (1 - cb) * z2
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return z, log_det


class Merge(Split):
    """
    Same as Split but with forward and backward pass interchanged
    """

    def __init__(self, mode="channel"):
        super().__init__(mode)

    def forward(self, z):
        return super().inverse(z)

    def inverse(self, z):
        return super().forward(z)


class Squeeze(Flow):
    """
    Squeeze operation of multi-scale architecture, RealNVP or Glow paper
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()

    def forward(self, z):
        log_det = 0
        s = z.size()
        z = z.view(s[0], s[1] // 4, 2, 2, s[2], s[3])
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        z = z.view(s[0], s[1] // 4, 2 * s[2], 2 * s[3])
        return z, log_det

    def inverse(self, z):
        log_det = 0
        s = z.size()
        z = z.view(*s[:2], s[2] // 2, 2, s[3] // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
        z = z.view(s[0], 4 * s[1], s[2] // 2, s[3] // 2)
        return z, 

class AffineConstFlow(Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    """

    def __init__(self, shape, scale=True, shift=True):
        """Constructor
        Args:
          shape: Shape of the coupling layer
          scale: Flag whether to apply scaling
          shift: Flag whether to apply shift
          logscale_factor: Optional factor which can be used to control the scale of the log scale factor
        """
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("s", torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("t", torch.zeros(shape)[None])
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(
            torch.tensor(self.s.shape) == 1, as_tuple=False
        )[:, 0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(self.s)
        return z_, log_det

    def inverse(self, z):
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)
        return z_, log_det


class CCAffineConst(Flow):
    """
    Affine constant flow layer with class-conditional parameters
    """

    def __init__(self, shape, num_classes):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.s = nn.Parameter(torch.zeros(shape)[None])
        self.t = nn.Parameter(torch.zeros(shape)[None])
        self.s_cc = nn.Parameter(torch.zeros(num_classes, np.prod(shape)))
        self.t_cc = nn.Parameter(torch.zeros(num_classes, np.prod(shape)))
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(
            torch.tensor(self.s.shape) == 1, as_tuple=False
        )[:, 0].tolist()

    def forward(self, z, y):
        s = self.s + (y @ self.s_cc).view(-1, *self.shape)
        t = self.t + (y @ self.t_cc).view(-1, *self.shape)
        z_ = z * torch.exp(s) + t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det

    def inverse(self, z, y):
        s = self.s + (y @ self.s_cc).view(-1, *self.shape)
        t = self.t + (y @ self.t_cc).view(-1, *self.shape)
        z_ = (z - t) * torch.exp(-s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det


class AffineCoupling(Flow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """

    def __init__(self, param_map, scale=True, scale_map="exp"):
        """Constructor
        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' uses multiplicative sigmoid scale when sampling from the model
        """
        super().__init__()
        self.add_module("param_map", param_map)
        self.scale = scale
        self.scale_map = scale_map

    def forward(self, z):
        """
        z is a list of z1 and z2; ```z = [z1, z2]```
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1
        Args:
          z
        """
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == "exp":
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 * scale + shift
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 = z2 + param
            log_det = zero_log_det_like_z(z2)
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == "exp":
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) / scale
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 = z2 - param
            log_det = zero_log_det_like_z(z2)
        return [z1, z2], log_det


class MaskedAffineFlow(Flow):
    """RealNVP as introduced in [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)
    Masked affine flow:
    ```
    f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    ```
    - class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    - NICE is AffineFlow with only shifts (volume preserving)
    """

    def __init__(self, b, t=None, s=None):
        """Constructor
        Args:
          b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
          t: translation mapping, i.e. neural network, where first input dimension is batch dim, if None no translation is applied
          s: scale mapping, i.e. neural network, where first input dimension is batch dim, if None no scale is applied
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        if s is None:
            self.s = lambda x: torch.zeros_like(x)
        else:
            self.add_module("s", s)

        if t is None:
            self.t = lambda x: torch.zeros_like(x)
        else:
            self.add_module("t", t)

    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det


class AffineCouplingBlock(Flow):
    """
    Affine Coupling layer including split and merge operation
    """

    def __init__(self, param_map, scale=True, scale_map="exp", split_mode="channel"):
        """Constructor
        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow
          split_mode: Splitting mode, for possible values see Split class
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [Split(split_mode)]
        # Affine coupling layer
        self.flows += [AffineCoupling(param_map, scale, scale_map)]
        # Merge layer
        self.flows += [Merge(split_mode)]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, 

    
class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done_cpu = torch.tensor(0.0)
        self.register_buffer("data_dep_init_done", self.data_dep_init_done_cpu)

    def forward(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.0:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (
                -z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)
            ).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().forward(z)

    def inverse(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().inverse(z)

    
    


