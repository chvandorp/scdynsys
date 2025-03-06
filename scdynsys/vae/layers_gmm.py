"""
Neural Network Layers for use in GMM models
"""

import torch
import torch.nn as nn
from pyro.nn.module import PyroModule
from typing import Literal

from .bnn import RegularizedLinear

class DecoderX(PyroModule):
    """
    Use a Gaussian mixture model on the latent space to cluster the data.
    This is the decoder network.
    """
    def __init__(
        self, 
        z_dim: int, 
        hidden_dim: int, 
        data_dim: int, 
        num_batch: int
    ) -> None:
        super().__init__()
        # parameters
        self.data_dim = data_dim
        self.num_batch = num_batch
        # setup the linear transformations used
        self.fc_in = PyroModule[nn.Linear](z_dim, hidden_dim)
        if self.num_batch > 0:
            self.fc_batch = PyroModule[nn.Linear](self.num_batch, hidden_dim, bias=False)
        else:
            self.fc_batch = None
        self.fc_loc = PyroModule[nn.Linear](hidden_dim, data_dim)
        self.fc_scale = PyroModule[nn.Linear](hidden_dim, data_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def sample(self) -> None:
        pass

    def forward(
        self,
        z: torch.Tensor,
        s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.fc_in(z)
        if self.num_batch > 0:
            hidden = hidden + self.fc_batch(s)
        # apply nonlinearity
        hidden = self.softplus(hidden)
        # location of reconstructed data
        x_loc = self.fc_loc(hidden)
        #x_scale = torch.exp(self.fc_scale(hidden)) ## TESTING: replace exp with softplus to reduce outliers
        x_scale = self.softplus(self.fc_scale(hidden))
        return x_loc, x_scale


class EncoderZ(PyroModule):
    """
    Use a Gaussian mixture model on the latent space to cluster the data.
    This is the encoder network for Z (loc and scale), given X.
    """
    def __init__(
        self, 
        z_dim: int, 
        hidden_dim: int, 
        data_dim: int, 
        num_batch: int
    ) -> None:
        super().__init__()
        # parameters
        self.data_dim = data_dim
        self.num_batch = num_batch
        # setup the n linear transformations used
        self.fc_in = PyroModule[nn.Linear](data_dim, hidden_dim)
        if self.num_batch > 0:
            self.fc_batch = PyroModule[nn.Linear](self.num_batch, hidden_dim, bias=False)
        self.fc_loc = PyroModule[nn.Linear](hidden_dim, z_dim)
        self.fc_scale = PyroModule[nn.Linear](hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def sample(self) -> None:
        pass

    def forward(
        self, 
        x: torch.Tensor, 
        s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # define the forward computation on the expression vector x
        hidden = self.fc_in(x)
        if self.num_batch > 0:
            hidden = hidden + self.fc_batch(s)
        hidden = self.softplus(hidden)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc_loc(hidden)
        #z_scale = torch.exp(self.fc_scale(hidden))
        z_scale = self.softplus(self.fc_scale(hidden))
        return z_loc, z_scale

    

    
## decoder and encoder networks with regularized batch effect correction
    
class RegularizedDecoderX(PyroModule):
    """
    Use a Gaussian mixture model on the latent space to cluster the data.
    This is the decoder network.
    The batch effects correction has a Bayesian prior for regularization
    """
    def __init__(
        self, 
        z_dim: int, 
        hidden_dim: int, 
        data_dim: int, 
        num_batch: int,
        reg_scale: float,
        reg_norm: Literal["l1", "l2"] = "l2",
        use_cuda: bool = False
    ) -> None:
        super().__init__()
        # device
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        # setup the linear transformations used
        self.fc_in = PyroModule[nn.Linear](z_dim, hidden_dim)
        self.num_batch = num_batch
        self.reg_scale = reg_scale

        self.fc_batch = RegularizedLinear(
            "decoder_x.fc_batch",
            self.num_batch,
            hidden_dim,
            self.reg_scale,
            reg_norm=reg_norm,
            bias=False,
            use_cuda=use_cuda
        )
        
        self.fc_loc = PyroModule[nn.Linear](hidden_dim, data_dim)
        self.fc_scale = PyroModule[nn.Linear](hidden_dim, data_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        # put tensors on GPU?
        if use_cuda:
            self.cuda()

    def sample(self) -> None:
        self.fc_batch.sample()

    def forward(
        self, 
        z: torch.Tensor, 
        s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # define the forward computation on the latent vector z
        hidden = self.fc_in(z)
        if self.num_batch > 0:
            hidden = hidden + self.fc_batch(s)
        hidden = self.softplus(hidden)
        # location of reconstructed data
        x_loc = self.fc_loc(hidden)
        #x_scale = torch.exp(self.fc_scale(hidden))
        x_scale = self.softplus(self.fc_scale(hidden))
        return x_loc, x_scale



class RegularizedEncoderZ(PyroModule):
    """
    Use a Gaussian mixture model on the latent space to cluster the data.
    This is the encoder network.
    The batch effects correction has a Bayesian prior for regularization
    """
    def __init__(
        self, 
        z_dim: int, 
        hidden_dim: int, 
        data_dim: int, 
        num_batch: int,
        reg_scale: float,
        reg_norm: Literal["l1", "l2"] = "l2",
        use_cuda: bool = False
    ) -> None:
        super().__init__()
        # device
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        # setup the linear transformations used
        self.fc_in = PyroModule[nn.Linear](data_dim, hidden_dim)
        self.num_batch = num_batch
        self.reg_scale = reg_scale

        self.fc_batch = RegularizedLinear(
            "decoder_x.fc_batch",
            self.num_batch,
            hidden_dim,
            self.reg_scale,
            reg_norm=reg_norm,
            bias=False,
            use_cuda=use_cuda
        )
        
        self.fc_loc = PyroModule[nn.Linear](hidden_dim, z_dim)
        self.fc_scale = PyroModule[nn.Linear](hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        # put tensors on GPU?
        if use_cuda:
            self.cuda()

    def sample(self) -> None:
        self.fc_batch.sample()

    def forward(
        self, 
        x: torch.Tensor, 
        s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # define the forward computation on the latent vector z
        hidden = self.fc_in(x)
        if self.num_batch > 0:
            hidden = hidden + self.fc_batch(s)
        hidden = self.softplus(hidden)
        # location of reconstructed data
        z_loc = self.fc_loc(hidden)
        #z_scale = torch.exp(self.fc_scale(hidden))
        z_scale = self.softplus(self.fc_scale(hidden))
        return z_loc, z_scale





class FullRegularizedDecoderX(PyroModule):
    """
    Put regularization on all the NN weights and biases
    """
    def __init__(
        self, 
        z_dim: int, 
        hidden_dim: int, 
        data_dim: int, 
        num_batch: int,
        reg_scale: float,
        reg_scale_batch: float,
        reg_norm: Literal["l1", "l2"] = "l2",
        use_cuda: bool = False
    ) -> None:
        super().__init__()
        # device
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")

        self.num_batch = num_batch
        self.reg_scale = reg_scale
        self.reg_scale_batch = reg_scale_batch

        # setup the linear transformations used
        self.fc_in = RegularizedLinear(
            "decoder_x.fc_in",
            z_dim, 
            hidden_dim,
            reg_scale,
            reg_norm=reg_norm,
            bias=True,
            use_cuda=use_cuda
        )

        self.fc_batch = RegularizedLinear(
            "decoder_x.fc_batch",
            num_batch,
            hidden_dim,
            reg_scale_batch,
            reg_norm=reg_norm,
            bias=False, # batch layer does not have a bias term
            use_cuda=use_cuda
        )
        
        self.fc_loc = RegularizedLinear(
            "decoder_x.fc_loc",
            hidden_dim,
            data_dim,
            reg_scale,
            reg_norm=reg_norm,
            bias=True,
            use_cuda=use_cuda
        )

        self.fc_scale = RegularizedLinear(
            "decoder_x.fc_scale",
            hidden_dim,
            data_dim,
            reg_scale,
            reg_norm=reg_norm,
            bias=True,
            use_cuda=use_cuda
        )
        # setup the non-linearities
        self.softplus = nn.Softplus()
        # put tensors on GPU?
        if use_cuda:
            self.cuda()

    def sample(self) -> None:
        self.fc_batch.sample()
        self.fc_in.sample()
        self.fc_loc.sample()
        self.fc_scale.sample()

    def forward(
        self, 
        z: torch.Tensor, 
        s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # define the forward computation on the latent vector z
        hidden = self.fc_in(z)
        if self.num_batch > 0:
            hidden = hidden + self.fc_batch(s)
        hidden = self.softplus(hidden)
        # location of reconstructed data
        x_loc = self.fc_loc(hidden)
        #x_scale = torch.exp(self.fc_scale(hidden))
        x_scale = self.softplus(self.fc_scale(hidden))
        return x_loc, x_scale
        



class FullRegularizedEncoderZ(PyroModule):
    """
    Put regularization on all the NN weights and biases
    """
    def __init__(
        self, 
        z_dim: int, 
        hidden_dim: int, 
        data_dim: int, 
        num_batch: int,
        reg_scale: float,
        reg_scale_batch: float,
        reg_norm: Literal["l1", "l2"] = "l2",
        use_cuda: bool = False
    ) -> None:
        super().__init__()
        # device
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")

        self.num_batch = num_batch
        self.reg_scale = reg_scale
        self.reg_scale_batch = reg_scale_batch

        # setup the linear transformations used
        self.fc_in = RegularizedLinear(
            "encoder_z.fc_in",
            data_dim, 
            hidden_dim,
            reg_scale,
            reg_norm=reg_norm,
            bias=True,
            use_cuda=use_cuda
        )

        self.fc_batch = RegularizedLinear(
            "encoder_z.fc_batch",
            num_batch,
            hidden_dim,
            reg_scale_batch,
            reg_norm=reg_norm,
            bias=False, # batch layer does not have a bias term
            use_cuda=use_cuda
        )
        
        self.fc_loc = RegularizedLinear(
            "encoder_z.fc_loc",
            hidden_dim,
            z_dim,
            reg_scale,
            reg_norm=reg_norm,
            bias=True,
            use_cuda=use_cuda
        )

        self.fc_scale = RegularizedLinear(
            "encoder_z.fc_scale",
            hidden_dim,
            z_dim,
            reg_scale,
            reg_norm=reg_norm,
            bias=True,
            use_cuda=use_cuda
        )
        # setup the non-linearities
        self.softplus = nn.Softplus()
        # put tensors on GPU?
        if use_cuda:
            self.cuda()

    def sample(self) -> None:
        self.fc_batch.sample()
        self.fc_in.sample()
        self.fc_loc.sample()
        self.fc_scale.sample()

    def forward(
        self, 
        x: torch.Tensor, 
        s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # define the forward computation on the latent vector z
        hidden = self.fc_in(x)
        if self.num_batch > 0:
            hidden = hidden + self.fc_batch(s)
        hidden = self.softplus(hidden)
        # location of reconstructed data
        z_loc = self.fc_loc(hidden)
        #z_scale = torch.exp(self.fc_scale(hidden))
        z_scale = self.softplus(self.fc_scale(hidden))
        return z_loc, z_scale
        

class FullRegularizedTimeEncoderZ(FullRegularizedEncoderZ):
    """
    Let the encoder use time information.
    """
    def __init__(
        self, 
        z_dim: int, 
        hidden_dim: int, 
        data_dim: int,
        num_batch: int,
        time_scale: float,
        reg_scale: float,
        reg_scale_batch: float,
        reg_norm: Literal["l1", "l2"] = "l2",
        use_cuda: bool = False
    ) -> None:
    
        # use init method from FullRegularizedEncoderZ
        super().__init__(
            z_dim,
            hidden_dim,
            data_dim,
            num_batch,
            reg_scale,
            reg_scale_batch,
            reg_norm,
            use_cuda
        )

        self.time_scale = time_scale

        # add layer for time
        self.fc_time = RegularizedLinear(
            "encoder_z.fc_time",
            1, # i.e. there is only a single time dimension
            hidden_dim,
            reg_scale,
            reg_norm=reg_norm,
            bias=False, # time layer does not have a bias term
            use_cuda=use_cuda
        )

    def sample(self) -> None:
        super().sample()
        self.fc_time.sample()

    def forward(
        self, 
        x: torch.Tensor,
        time: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # define the forward computation on the latent vector z
        hidden = self.fc_in(x) + self.fc_time(self.time_scale * time.view(time.shape + (1,)))
        if self.num_batch > 0:
            hidden = hidden + self.fc_batch(s)
        hidden = self.softplus(hidden)
        # location of reconstructed data
        z_loc = self.fc_loc(hidden)
        #z_scale = torch.exp(self.fc_scale(hidden))
        z_scale = self.softplus(self.fc_scale(hidden))
        return z_loc, z_scale


    
