"""PCE Model.

Written by Xiaoshu Zeng while at the University of Southern California
Year: 2024
"""

import math
import numpy as np
from scipy import special
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init, Module


class Hermite1d:
    """
    Construct 1-dimensional normalized Hermite polynomials.
    """

    _nord = None

    def __init__(self, nord=1):
        """
        Ininializes the object
        """
        self._nord = nord

    def eval(self, x):
        H = np.zeros(self._nord + 1)
        H[0], H[1] = 1.0, x
        for i in range(2, H.shape[0]):
            H[i] = x * H[i - 1] - (i - 1) * H[i - 2]
        # normalization
        H = H / [math.sqrt(math.factorial(i)) for i in range(H.shape[0])]
        return H

    def __call__(self, x):
        N = x.shape[0]
        H = np.zeros((N, self._nord + 1))
        for i in range(N):
            H[i, :] = self.eval(x[i])
        return H


class Legendre1d:
    """
    Construct 1-dimensional Legendre polynomials.
    """

    _nord = None

    def __init__(self, nord=1):
        """
        Initializes the object
        """
        self._nord = nord

    def eval(self, x):
        H = np.zeros(self._nord + 1)
        H[0], H[1] = 1.0, x
        for i in range(2, H.shape[0]):
            H[i] = ((2 * i - 1) * x * H[i - 1] - (i - 1) * H[i - 2]) / i
        # H = H / [math.sqrt(2 / (2*i+1))
        #          for i in range(H.shape[0])]  # normalized
        return H

    def __call__(self, x):
        N = x.shape[0]
        H = np.zeros((N, self._nord + 1))
        for i in range(N):
            H[i, :] = self.eval(x[i])
        return H


class PolyBasis:
    """
    Construct PCE basis terms
    """

    _nord = None
    _ndim = None
    _MI_terms = None
    _type = None

    def __init__(self, ndim=1, nord=1, pol_type="HG"):
        # Ininializes the object
        assert pol_type in ["hermite", "legendre"], (
            "Only Hermite and Legendre polynomials are currently supported! Please choose among "
            + str(["hermite", "legendre"])
            + "!"
        )
        self._nord = nord
        self._ndim = ndim
        self._MI_terms = self.mi_terms(self._ndim, self._nord)
        self._type = pol_type

    def __call__(self, xi):
        assert xi.shape[1] == self._ndim

        if self._type == "hermite":
            H = [
                Hermite1d(nord=self._nord)(xi[:, i]) for i in range(self._ndim)
            ]
            psi_xi = np.ones((xi.shape[0], self._MI_terms.shape[0]))
            for i in range(self._MI_terms.shape[0]):
                for j in range(self._ndim):
                    psi_xi[:, i] *= H[j][:, self._MI_terms[i, j]]

        elif self._type == "legendre":
            H = [
                Legendre1d(nord=self._nord)(xi[:, i])
                for i in range(self._ndim)
            ]
            psi_xi = np.ones((xi.shape[0], self._MI_terms.shape[0]))
            for i in range(self._MI_terms.shape[0]):
                for j in range(self._ndim):
                    psi_xi[:, i] *= H[j][:, self._MI_terms[i, j]]
        else:
            raise NotImplementedError(f"Poly type {self._type}")

        return psi_xi

    def mi_terms(self, ndim, nord):
        """
        Multiindex matrix

        ndim: integer

        nord: PCE order
        """
        q_num = [int(special.comb(ndim + i - 1, i)) for i in range(nord + 1)]
        mul_ind = np.array(np.zeros(ndim, dtype=int), dtype=int)
        mul_ind = np.vstack([mul_ind, np.eye(ndim, dtype=int)])
        I = np.eye(ndim, dtype=int)
        ind = [1] * ndim
        for j in range(1, nord):
            ind_new = []
            for i in range(ndim):
                a0 = np.copy(I[int(np.sum(ind[:i])) :, :])
                a0[:, i] += 1
                mul_ind = np.vstack([mul_ind, a0])
                ind_new += [a0.shape[0]]
            ind = ind_new
            I = np.copy(mul_ind[np.sum(q_num[: j + 1]) :, :])
        return mul_ind

    def mi_terms_loc(self, d1, d2, nord):
        """
        Locate basis terms in Multi-index matrix
        """
        assert d1 < d2
        MI2 = self.mi_terms(d2, nord)
        if d2 == d1 + 1:
            return np.where(MI2[:, -1] == 0)[0]
        else:
            TFs = MI2[:, d1:] == [0] * (d2 - d1)
            locs = []
            for i in range(TFs.shape[0]):
                if TFs[i, :].all():
                    locs.append(i)
            return locs


class PCE(Module):
    r"""Applies a PCE on the incoming data: :math:`y = \Psi(x) C^T`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        pce_order: PCE order
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        weight: the learnable PCE coefficients of the module of shape
            :math:`(\text{out\_features}, \text{num\_PCE})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{num\_PCE}}`. :math:`\text{num\_PCE}` is the number
            of PCE terms, determined by in_features and pce_order.

    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    pce_order: int
    poly_name: str
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pce_order: int,
        poly_name: str,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pce_order = pce_order
        self.poly_name = poly_name
        self.npce = int(
            math.factorial(in_features + pce_order)
            / (math.factorial(pce_order) * math.factorial(in_features))
        )
        self.weight = Parameter(
            torch.empty((out_features, self.npce), **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        psi_xi = Tensor(
            PolyBasis(self.in_features, self.pce_order, self.poly_name)(input)
        )
        output = torch.mm(psi_xi, torch.t(self.weight))
        return output
