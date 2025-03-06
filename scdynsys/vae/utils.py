from typing import Tuple, Any, List, Optional, Union, Callable
import torch
from torch.utils.data import random_split, DataLoader
import pyro
from pyro.infer.svi import SVI
from pyro.infer.elbo import ELBO
from pyro.optim.optim import PyroOptim
import numpy as np
import tqdm
import tqdm.notebook

def setup_data_loaders(
    raw_data: Any,
    num_train: int,
    num_test: int,
    batch_size: int = 128,
    seed: Optional[int] = None,
    use_cuda: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    random split of the dataset into test and train, then create data loaders.
    Provide a seed for consistent split across different runs.
    """
    if seed is None:
        train_set, test_set = random_split(raw_data, [num_train, num_test])
    else:
        train_set, test_set = random_split(
            raw_data, 
            [num_train, num_test], 
            generator=torch.Generator().manual_seed(seed)
        )

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs
    )
    return train_loader, test_loader 
    

def initial_rejection_sampler(
    num_inits: int, 
    vae_builder: Callable, ## should return a VAE
    train_loader: DataLoader,
    optimizer: PyroOptim,
    loss_method: ELBO,
    persuasiveness: float,
    addl_data: Tuple[torch.Tensor,...] = (),
    use_cuda: bool = False,
    show_progress: Union[bool, str] = True
) -> Tuple[Any, SVI]: ## FIXME: restrict return type
    """
    Function to repeatedly initiate a VAE, compute the initial loss
    and then returns the instance with the smallest loss, as well
    as setting the pyro parameter store to the corresponding scope
    """
    if type(show_progress) is bool:
        trange = tqdm.trange if show_progress else range
    elif type(show_progress) is str and show_progress == "notebook":
        trange = tqdm.notebook.trange
    else:
        raise Exception("invalid value for argument 'show_progress'")            

    store = pyro.get_param_store()
    min_loss = np.inf
    for i in trange(num_inits):
        # setup the inference algorithm
        with store.scope() as scope_try:
            vae_try = vae_builder()
            svi_try = SVI(vae_try.model, vae_try.guide, optimizer, loss=loss_method)
            # get initial loss
            loss = train(
                svi_try, 
                train_loader, 
                persuasiveness, 
                addl_data=addl_data, 
                use_cuda=use_cuda, 
                eval_only=True
            )
            if loss < min_loss:
                min_loss = loss
                svi = svi_try
                vae = vae_try
                scope = scope_try
    
    store.set_state(scope)
    return vae, svi


def train(
    svi: SVI,
    data_loader: DataLoader,
    persuasiveness: float,
    addl_data: Tuple[torch.Tensor, ...] = (),
    use_cuda: bool = False,
    eval_only: bool = False
) -> float:
    """
    generic training function for SVI object
    """
    # initialize loss accumulator
    epoch_loss = 0.0
    normalizer = len(data_loader.dataset)
    N = torch.tensor(normalizer)
    phi = torch.tensor(persuasiveness)
    if use_cuda:
        N = N.cuda()
        phi = phi.cuda()
        addl_data = tuple(y.cuda() for y in addl_data)
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for xs in data_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            xs = tuple(x.cuda() for x in xs)
        # do ELBO gradient and accumulate loss
        if not eval_only:
            epoch_loss += svi.step(*xs, N, phi, *addl_data)
        else:
            epoch_loss += svi.evaluate_loss(*xs, N, phi, *addl_data)
    # return epoch loss
    total_epoch_loss = epoch_loss / len(data_loader)
    return total_epoch_loss


def train_test_loop(
    svi: SVI,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    test_interval: int,
    persuasiveness_shrink_rate: float,
    initial_persuasiveness: float = 1.0,
    addl_data: Tuple[torch.Tensor, ...] = (),
    use_cuda: bool = False,
    show_progress: Union[bool, str] = True
) -> tuple[list[tuple[int, float]], ...]:
    """
    Test-train loop for SVI. repeatedly call the test_train function.

    Parameters
    ----------
    svi : SVI
        Pyro SVI object containing model and guide.
    train_loader : DataLoader
        Torch dataloader with mini-batched train data.
    test_loader : DataLoader
        Torch dataloader with mini-batched test data. Not used for SVI steps
    num_epochs : int
        The number of epochs used for training.
    test_interval : int
        Evaluate the loss with the test data every `test_interval` epochs.
    persuasiveness_shrink_rate : float
        Parameter determining how fast the cluster hints are weighted down.
        The likelihood of cluster hints is weighted with a factor
        exp(-persuasiveness_shrink_rate * epoch) using pyro.poutine.scale
    initial_persuasiveness : float
        Initial persuasiveness vale. The default is 1.0.
    addl_data : Tuple[torch.Tensor, ...], optional
        additional data required by the model that is not mini-batched. 
        The default is ().
    use_cuda : bool, optional
        Use the GPU. The default is False.
    show_progress : Union[bool, str], optional
        Use tqdm to show a progress bar. The default is True.

    Raises
    ------
    Exception
        Throw an exception if incorrect `show_progress` are given.

    Returns
    -------
    train_elbo : list[tuple[int, float]]
        list of epochs and train loss pairs
    test_elbo : list[tuple[int, float]]
        list of epochs and test loss pairs
    """
    
    if type(show_progress) is bool:
        trange = tqdm.trange if show_progress else range
    elif type(show_progress) is str and show_progress == "notebook":
        trange = tqdm.notebook.trange
    else:
        raise Exception("invalid value for argument 'show_progress'")            
    
    train_elbo = []
    test_elbo = []
        
    for epoch in (pbar := trange(num_epochs)):
        persuasiveness = np.exp(-epoch*persuasiveness_shrink_rate)
        total_epoch_loss_train = train(
            svi, 
            train_loader,
            persuasiveness,
            addl_data=addl_data,
            use_cuda=use_cuda
        )
        train_elbo.append((epoch, total_epoch_loss_train))
        if show_progress:
            pbar.set_description(f"average train loss: {total_epoch_loss_train:0.2f}")

        if epoch % test_interval == 0:
            # report test diagnostics
            total_epoch_loss_test = train(
                svi, 
                test_loader,
                persuasiveness,
                addl_data=addl_data, 
                use_cuda=use_cuda,
                eval_only=True
            )
            test_elbo.append((epoch, total_epoch_loss_test))
            if show_progress:
                pbar.set_description(f"average test loss: {total_epoch_loss_test:0.2f}")
                        
    return train_elbo, test_elbo
    

def train_test_loop_full_dataset(
    svi: SVI,
    raw_train_data: tuple[np.array, ...],
    raw_test_data: tuple[np.array, ...],
    num_epochs: int,
    test_interval: int,
    persuasiveness_shrink_rate: float,
    initial_persuasiveness: float = 1.0,
    raw_addl_data: tuple[np.array, ...] = (),
    use_cuda: bool = False,
    show_progress: Union[bool, str] = True
) -> tuple[list[tuple[int, float]], ...]:
    """
    Do a train-test loop without data loaders. Just put the 
    entire dataset in GPU RAM (or regular RAM) and feed it 
    to SVI in every step. This avoids having to move data 
    around.

    Parameters
    ----------
    svi : SVI
        The training object containing the model and guide.
    raw_train_data : tuple[np.array, ...]
        A tuple with all the raw (non-tensor) training data.
    raw_test_data : tuple[np.array, ...]
        A tuple with all the raw (non-tensor) testing data.
    num_epochs : int
        Total number of iterations.
    test_interval : int
        Evaluate the loss against the testing data every so many training 
        steps.
    persuasiveness_shrink_rate : float
        This determines how fast the persuasiveness parameter shrinks.
        This is used to gradually turn off semi-sipervised clustering.
    initial_persuasiveness : float, optional
        The initial persuasiveness parameter. Weight of the paired data 
        likelihood contribution. The default is 1.0.
    raw_addl_data : tuple[np.array, ...], optional
        Additional data to feed to the model. The default is ().
    use_cuda : bool, optional
        Should we use the GPU?. The default is False.
    show_progress : Union[bool, str], optional
        Settings for tqdm progress bar. The default is True.

    Raises
    ------
    Exception
        Raise an exception of the tqdm settings are invalid.

    Returns
    -------
    train_elbo : TYPE
        A timeseries of training epochs and losses.
    test_elbo : TYPE
        A timeseries of testing epochs and losses.

    """
    if type(show_progress) is bool:
        trange = tqdm.trange if show_progress else range
    elif type(show_progress) is str and show_progress == "notebook":
        trange = tqdm.notebook.trange
    else:
        raise Exception("invalid value for argument 'show_progress'")            
    
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    train_data = tuple(torch.tensor(x, device=device) for x in raw_train_data)
    test_data = tuple(torch.tensor(x, device=device) for x in raw_test_data)
    addl_data = tuple(torch.tensor(x, device=device) for x in raw_addl_data)
    
    N = torch.tensor(train_data[0].shape[0], device=device)
    
    train_elbo = []
    test_elbo = []
    
    psr = torch.tensor(persuasiveness_shrink_rate, device=device)
        
    for epoch in (pbar := trange(num_epochs)):
        # shrink persuasiveness
        persuasiveness = initial_persuasiveness * (-epoch * psr).exp()
        
        total_epoch_loss_train = svi.step(*train_data, N, persuasiveness, *addl_data)
        
        train_elbo.append((epoch, total_epoch_loss_train))
        if show_progress:
            pbar.set_description(f"train loss: {total_epoch_loss_train:0.2f}")

        if epoch % test_interval == 0:
            # report test diagnostics
            total_epoch_loss_test = svi.evaluate_loss(*test_data, N, persuasiveness, *addl_data)

            test_elbo.append((epoch, total_epoch_loss_test))
            if show_progress:
                pbar.set_description(f"test loss: {total_epoch_loss_test:0.2f}")
                        
    return train_elbo, test_elbo



def onehot_encoding(
    unique_batch: List[str], 
    raw_batch: List[str],
    unique_expt: List[str], 
    raw_expt: List[str], 
    scale_batch: float,
    batch_ref: Optional[str] = None,
    expt_ref: Optional[str] = None,
) -> Tuple[np.ndarray, ...]:
    """
    One-hot encode batch and experiment codes, and concatenate the results.
    Scale the batch encoding with a given factor for regularized batch corrections

    Parameters
    ----------
    unique_batch : List[str]
        list of unique batch IDs. The order determines the index of the ones.
    raw_batch : List[str]
        list of batch ID per sample.
    unique_expt : List[str]
        list of unique experiment IDs.
    raw_expt : List[str]
        list of experiment ID per sample.
    scale_batch : float
        Scale the ones at batch indices with this factor.
    batch_ref: str | None
        if not None, use this as a reference, meaning that the zero vector 
        is the encoding for this batch. If None, use redundant encoding
    expt_ref: str | None
        same is for batch_ref, but then for the expt ID.

    Returns
    -------
    batch_expt_onehot : np.array
        one-hot encoded batch and experiment IDs per sample.
        Concatenated into a single array.
    batch_onehot : np.ndarray
        one-hot encoded batch ID.
    expt_onehot : np.ndarray
        one-hot encoded experiment ID.
    """
    
    
    num_samples = len(raw_batch)
    num_batch = len(unique_batch)
    num_expt = len(unique_expt)
    
    assert len(raw_expt) == num_samples, "expt list and batch list must have the same length"
    
    batch_idx_dict = {b: i for i, b in enumerate(unique_batch)}
    batch_idx = [batch_idx_dict[b] for b in raw_batch]

    batch_onehot = np.zeros((num_samples, num_batch), dtype=np.float32)
    batch_onehot[np.arange(num_samples), batch_idx] = 1.0
    
    # optionally, remove the reference column
    if batch_ref is not None:
        ref_idx = batch_idx_dict[batch_ref]
        batch_onehot = np.delete(batch_onehot, ref_idx, axis=1)

    expt_idx_dict = {b: i for i, b in enumerate(unique_expt)}
    expt_idx = [expt_idx_dict[b] for b in raw_expt]

    expt_onehot = np.zeros((num_samples, num_expt), dtype=np.float32)
    expt_onehot[np.arange(num_samples), expt_idx] = 1.0
    
    # optionally, remove the reference column
    if expt_ref is not None:
        ref_idx = expt_idx_dict[expt_ref]
        expt_onehot = np.delete(expt_onehot, ref_idx, axis=1)

    batch_expt_onehot = np.concatenate([scale_batch*batch_onehot, expt_onehot], axis=1)
    
    return batch_expt_onehot, batch_onehot, expt_onehot


def permute_matrix(A: torch.Tensor, perm: list[int]) -> torch.Tensor:
    """
    Permute the rows and columns of a matrix to a given ordering

    Parameters
    ----------
    A : torch.Tensor
        Square matrix with possibly batch dimensions on the left.
    perm : list[int]
        permutation.

    Returns
    -------
    ppA : torch.Tensor
        permuted matrix.
    """
    pA = A[..., perm, :]
    ppA = pA[..., :, perm]
    return ppA
    

