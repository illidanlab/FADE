"""Split networks"""
from torch import nn
from typing import List


class SplitNet(nn.Module):
    """Only partial network is sharable. The shared subnet is stored in the attribute `shared`.

    Note: by default, the loss will be cross-entropy.
    """
    def __init__(self):
        super().__init__()
        # put shared sub-modules in the list.
        self.shared = []  # type: List[nn.Parameter]
        self.private = []  # type: List[nn.Parameter]

    def get_shared_parameters(self, detach=True):
        """Return a list of shared parameters."""
        return self.shared

    def get_private_parameters(self):
        """Return a list of shared parameters."""
        return self.private

    def get_shared_submodule(self):
        """This is used in DP engine. A whole module is required by dp engine.
        If no module can be returned, (for example, shared part is not in a module), then None is
        returned.
        """
        return None


class SplitEncoder(SplitNet):
    def encode(self, x, a=0.5):
        return self.encoder(x)

    def decode(self, z, a=0.5):
        return self.decoder(z)

    def forward(self, x, a=0.5):
        z = self.encode(x, a=a)
        return self.decode(z, a=a)
