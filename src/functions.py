import torch
import torch.nn.functional as F


def SvegaSoftmax(logits):
    exponential_scores = torch.exp(logits)
    exponential_sums = exponential_scores.sum(dim=1, keepdim=True)
    probabilities = exponential_scores / exponential_sums

    return probabilities


def SvegaCrossEntropyLoss(logits, labels):
    probabilities = SvegaSoftmax(logits)
    one_hot_labels = F.one_hot(labels, logits.size(1)).float()
    total_log_likelihood = torch.sum(torch.log(probabilities + 1e-9) * one_hot_labels) 
    average_log_likelihood = total_log_likelihood / logits.size(0)

    return -average_log_likelihood
