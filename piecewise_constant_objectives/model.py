import torch as th
from hex_alg import zoo_2nd_argmax

dtype = th.float64
class RNN(th.nn.Module):
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