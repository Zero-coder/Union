import math

import torch


def adjust_learning_rate(optimizer, epoch, args):
    if getattr(args, "lradj", "constant") == "constant":
        return
    lr = args.learning_rate * (0.5 ** max(epoch - 1, 0))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    total_iters = epochs * niter_per_ep
    warmup_iters = warmup_epochs * niter_per_ep
    schedule = []
    for it in range(total_iters):
        if it < warmup_iters and warmup_iters > 0:
            value = base_value * it / warmup_iters
        else:
            progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
            value = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))
        schedule.append(value)
    return schedule


def cal_accuracy(pred, true):
    return (pred.argmax(dim=-1) == true).float().mean().item()


def adjustment(gt, pred):
    return gt, pred


class NativeScalerWithGradNormCount:
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        loss.backward(create_graph=create_graph)
        norm = None
        if update_grad:
            if clip_grad is not None and parameters is not None:
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return norm
