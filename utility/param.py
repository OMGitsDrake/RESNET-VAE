from torch import exp, randn_like

def rep_sample(mu, log_var):
    std = exp(0.5 * log_var)
    sample = randn_like(std)

    return mu + std * sample

def normalize_minst(batch):
    return (batch - 0.1307) / 0.3081