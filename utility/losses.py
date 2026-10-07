def total_variation_loss(imgs):
    vertical_diff = (
        imgs[:, :, 1:, :] - imgs[:, :, :-1, :]
    )

    horizontal_diff = (
        imgs[:, :, :, 1:] - imgs[:, :, :, :-1]
    )

    return vertical_diff.abs().mean() + horizontal_diff.abs().mean()

def KL_loss(mu, log_var):
    return 0.5 * (
        log_var.exp()
        + mu.square()
        - 1
        - log_var
    ).sum(dim=1).mean()