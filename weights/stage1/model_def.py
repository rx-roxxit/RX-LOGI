"""Stage-1 model class = the vendored, byte-identical ChainNet.

To load a prod_seed*.pt: ck = torch.load(path, weights_only=False);
net = ChainNet(ck["n_ctx"], n_out=ck["n_out"], **ck["arch"]);
net.load_state_dict(ck["state_dict"]); net.eval();
normalize context with (ctx - ck["ctx_mu"]) / ck["ctx_sd"]. See ../../inference.py.
"""
from src.core.s19.model import ChainNet  # noqa: F401
