"""CDAN Loss:

References:
    https://github.com/thuml/CDAN/blob/f7889063b76fca0b9a7147c88103d356531924bd/pytorch/loss.py#L21
"""
import torch


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def CDAN_predict_task(feature, softmax_output, model, random_layer=None, alpha=None):
    softmax_output = softmax_output.detach()
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = model.predict_task(op_out.view(-1, softmax_output.size(1) * feature.size(1)),
                                    rev_lambda=alpha)
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = model.predict_task(random_out.view(-1, random_out.size(1)), rev_lambda=alpha)
    return ad_out


def CDAN(group_loss, softmax_output, group_labels, compute_ent_weights=False, alpha=None):
    # group_loss = F.binary_cross_entropy_with_logits(pred_group, group_labels)
    if compute_ent_weights:
        entropy = Entropy(softmax_output)
        entropy.register_hook(grl_hook(alpha))
        entropy = 1.0+torch.exp(-entropy)
        for g in (0, 1):
            mask = group_labels == g
            entropy[mask] = entropy[mask] / torch.sum(entropy[mask]).detach().item()
        return torch.sum(entropy.view(-1, 1) * group_loss) / torch.sum(entropy).detach().item()
    else:
        return group_loss
