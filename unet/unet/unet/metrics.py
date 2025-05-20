import torch


def _compute_f1(s, s_hat):
    """
       Function that computes true positives (tp), false positives (fp) and
       false negatives (fn).

       Arguments:
           s_hat (torch.Tensor) : Shape (B x T x M) model state predictions
           s (torch.Tensor) : Shape (B x T x M) ground truth states
       Returns:
           tp (int), fp (int), fn (int) : Tuple[int] containing tp, fp and fn
    """
    tp = torch.sum(s * s_hat).float()
    fp = torch.sum(torch.logical_not(s) * s_hat).float()
    fn = torch.sum(s * torch.logical_not(s_hat)).float()
    tn = torch.sum(torch.logical_not(s) * torch.logical_not(s_hat)).float()

    # Calculate Precision and Recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # Calculate F1 Score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return f1, tp, fp, fn, tn, accuracy
