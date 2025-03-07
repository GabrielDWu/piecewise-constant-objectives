import torch as th
import blobfile as bf

# BEGIN CODE FROM JACOB'S HEX-ALG LIBRARY #####################################
class OneLayerRNN(th.nn.Module):
    def __init__(self, hidden_size, output_size, bias=False):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias
        super().__init__()
        self.rnn = th.nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            nonlinearity="relu",
            bias=bias,
            batch_first=True,
        )
        self.linear = th.nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, x, init_state=None):
        output, final_state = self.rnn(x[..., None], init_state)
        return self.linear(final_state.squeeze(0))

    @property
    def device(self):
        return self.linear.weight.device

    @property
    def dtype(self):
        return self.linear.weight.dtype


class DistRNN(OneLayerRNN):
    def __init__(self, hidden_size, seq_len, bias=False):
        self.seq_len = seq_len
        super().__init__(hidden_size, seq_len, bias=bias)

def load_dist_rnn(model_path, bias=False):
    with bf.BlobFile(model_path, "rb") as model_file:
        state_dict = th.load(model_file, weights_only=True, map_location="cpu")
    seq_len, hidden_size = state_dict["linear.weight"].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=bias)
    model.load_state_dict(state_dict)
    return model

def zoo_2nd_argmax(hidden_size, seq_len):
    if hidden_size not in [2, 3, 4, 5, 6, 8, 16, 32]:
        raise ValueError(f"No zoo model with hidden_size {hidden_size}.")
    if seq_len not in [2, 3, 4, 5, 6, 8, 10]:
        raise ValueError(f"No zoo model with seq_len {seq_len}.")
    model = load_dist_rnn(
        f"gs://arc-ml-public/alg/zoo/2nd_argmax_{hidden_size}_{seq_len}_{2**30}_bo5.pth",
        bias=False,
    )
    assert model.hidden_size == hidden_size, (
        f"Model hidden_size {model.hidden_size} "
        f"did not match requested hidden_size {hidden_size}."
    )
    assert model.seq_len == seq_len, (
        f"Model seq_len {model.seq_len} " f"did not match requested seq_len {seq_len}."
    )
    return model

# END OF CODE FROM JACOB'S HEX-ALG LIBRARY #####################################

dtype = th.float64
class RNN(th.nn.Module):
    """
    A simple recursive neural network as defined in the paper.
    """
    def __init__(self, hidden_size, seq_len, load_from_zoo=True):
        self.device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")
        super().__init__()
        self.dtype = th.float64
        if load_from_zoo:
            self.load_from_zoo(hidden_size, seq_len)
        else:
            self.n = seq_len
            self.d = hidden_size
            self.Whh = th.nn.Parameter(th.randn(hidden_size, hidden_size, dtype=dtype, device=self.device))
            self.Whi = th.nn.Parameter(th.randn(hidden_size, 1, dtype=dtype, device=self.device))
            self.Woh = th.nn.Parameter(th.randn(seq_len, hidden_size, dtype=dtype, device=self.device))
    
    def to(self, device):
        self.device = device
        return super().to(device)
    
    def load_from_zoo(self, hidden_size, seq_len):
        # The way we load this model is a bit convoluted because it historically interfaced with an internal ARC codebase.
        # Don't worry about it.
        self.n = seq_len
        self.d = hidden_size
        zoo_model = zoo_2nd_argmax(hidden_size=hidden_size, seq_len=seq_len)
        
        self.Whh = th.nn.Parameter(zoo_model.rnn.weight_hh_l0.detach().to(dtype).to(self.device))
        self.Whi = th.nn.Parameter(zoo_model.rnn.weight_ih_l0.detach().to(dtype).to(self.device))
        self.Woh = th.nn.Parameter(zoo_model.linear.weight.detach().to(dtype).to(self.device))

    def forward(self, x):
        state = th.zeros(self.d, dtype=self.dtype, device=self.device)
        for t in range(self.n):
            state = th.nn.functional.relu(state @ self.Whh.T + x[:, t:t+1] @ self.Whi.T)
        return state @ self.Woh.T