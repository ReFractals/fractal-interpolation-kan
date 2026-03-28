# FI-KAN: Fractal Interpolation Kolmogorov-Arnold Networks
# --------------------------------------------------------
# Author: Gnankan Landry Regis N'guessan
# Affiliations:
#   Axiom Research Group
#   Dept. of Applied Mathematics and Computational Science, NM-AIST, Arusha, Tanzania
#   AIMS Research and Innovation Centre, Kigali, Rwanda
# Contact: rnguessan@aimsric.org
# --------------------------------------------------------

from .bases import fractal_bases, fractal_dim_from_d
from .layers import PureFIKANLinear, HybridFIKANLinear
from .models import PureFIKAN, HybridFIKAN

__all__ = [
    "fractal_bases",
    "fractal_dim_from_d",
    "PureFIKANLinear",
    "HybridFIKANLinear",
    "PureFIKAN",
    "HybridFIKAN",
]
