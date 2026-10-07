from torch import exp, randn_like
from math import exp as m_exp

def rep_sample(mu, log_var):
    std = exp(0.5 * log_var)
    sample = randn_like(std)

    return mu + std * sample

def normalize_minst(batch):
    return (batch - 0.1307) / 0.3081

def update_beta(
    epoch: int,
    full_weight_epoch: int,
    steep: float = 3.0
) -> float:
    prog = min(
        (epoch - 1) / (full_weight_epoch - 1),
        1.0
    )

    # Uncomment for fast increment then convergence
    # beta = (1 - m_exp(-steep * prog)) / (1 - m_exp(-steep))
    beta = (m_exp(steep * prog) - 1) / (m_exp(steep) - 1)

    return beta