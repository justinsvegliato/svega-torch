import torch
import torch.nn.functional as F


def SvegaSoftmax(logits):
    exponential_scores = torch.exp(logits)
    exponential_sums = exponential_scores.sum(dim=-1, keepdim=True)
    probabilities = exponential_scores / exponential_sums

    return probabilities


def SvegaCrossEntropyLoss(logits, labels):
    probabilities = SvegaSoftmax(logits)
    log_probabilities = torch.log(probabilities + 1e-9)

    one_hot_labels = F.one_hot(labels, num_classes=logits.size(-1)).float()

    total_log_likelihood = torch.sum(log_probabilities * one_hot_labels, dim=-1)
    average_log_likelihood = total_log_likelihood.sum() / total_log_likelihood.numel()

    return -average_log_likelihood
