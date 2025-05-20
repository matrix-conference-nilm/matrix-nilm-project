import torch


def quantile_regression_loss(y_hat, y, taus):
    """
        Function that computes the quantile regression loss

        Arguments:
            y_hat (torch.Tensor) : Shape (B x T x N x M) model regression predictions
            y (torch.Tensor) : Shape (B x T x M) ground truth targets
            taus (torch.Tensor) : Shape (N, ) Vector of used quantiles
        Returns:
            loss (float): value of quantile regression loss
    """
    # print(y_hat.size(), y.size())
    iy = y.unsqueeze(2).unsqueeze(-1).expand_as(y_hat)
    error = (iy - y_hat).permute(0, 1, 3, 2)
    # error = (iy - y_hat)
    loss = torch.max(taus * error, (taus - 1.) * error)
    return torch.mean(torch.sum(loss, dim=-1))