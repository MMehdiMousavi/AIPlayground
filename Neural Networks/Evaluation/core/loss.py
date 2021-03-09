def dice_loss_binary(outputs=None, target=None, beta=1, weights=None):
    """
    :param weights: element-wise weights
    :param outputs:
    :param target:
    :param beta: More beta, better precision. 1 is neutral
    :return:
    """
    from torch import min as tmin
    smooth = 1.0
    if weights:
        w = weights.contiguous().float().view(-1)
        if tmin(w).item() == 0:
            w += smooth
    else:
        w = 1.0

    iflat = outputs.contiguous().float().view(-1)
    tflat = target.contiguous().float().view(-1)
    intersection = (iflat * tflat * w).sum()

    return (((1 + beta ** 2) * intersection) + smooth) / (
            ((beta ** 2 * (w * iflat).sum()) + (w * tflat).sum()) + smooth)
