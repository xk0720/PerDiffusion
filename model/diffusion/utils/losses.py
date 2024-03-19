from __future__ import print_function

import torch


def TemporalLoss(Y):
    diff = Y[:, 1:, :] - Y[:, :-1, :]
    t_loss = torch.mean(torch.norm(diff, dim=2, p=2) ** 2)
    return t_loss


def L1Loss(prediction, target, k=1, reduction="min", **kwargs):
    # prediction has shape of [batch_size, num_preds, features]
    # target has shape of [batch_size, num_preds, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of L1 loss
    loss = (torch.abs(prediction - target)).mean(axis=-1)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELoss(prediction, target, k=1, reduction="mean", **kwargs):
    # prediction has shape of [batch_size, num_preds==k, features]
    # target has shape of [batch_size, num_preds==k, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of MSE loss
    loss = ((prediction - target) ** 2).mean(axis=-1)  # (batch_size, k)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELossWithAct(prediction, target, k=1, reduction="mean", **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    bs, k, _ = prediction.shape
    prediction = prediction.reshape(bs, k, 50, 25) # window_size==50, emotion_dim==25

    AU = prediction[:, :, :, :15] # (bs, k, 50, 15)
    AU = torch.sigmoid(AU)

    middle_feat = prediction[:, :, :, 15:17] # (bs, k, 50, 2)
    middle_feat = torch.tanh(middle_feat)

    emotion = prediction[:, :, :, 17:] # (bs, k, 50, 8)
    emotion = torch.softmax(emotion, dim=-1)

    prediction = torch.cat((AU, middle_feat, emotion), dim=-1)
    prediction = prediction.reshape(bs, k, -1)
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"

    # manual implementation of MSE loss
    loss = ((prediction - target) ** 2).mean(axis=-1)  # (batch_size, k)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def KApproMSELoss(prediction, target, k, **kwargs):
    # prediction has shape of [batch_size, num_preds==k, features]
    # target has shape of [batch_size, num_preds==k, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    bs, _, feat_dim = prediction.shape
    metrics = torch.zeros(size=(bs, 0, k)).to(prediction.device)
    preds = prediction.detach().clone()
    for idk in range(k):
        pred = preds[:, idk:idk + 1, :]  # (bs, 1, features)
        pred = pred.repeat(1, k, 1)  # (bs, k, features)
        mse = ((pred - target) ** 2).mean(axis=-1).unsqueeze(1)  # (bs, 1, k) each idk ==> all k
        metrics = torch.cat((metrics, mse), dim=1)
    # metrics.shape: (bs, k, k)
    minimum_mse = torch.argmin(metrics, dim=-1, keepdim=True)  # (bs, k, 1)
    minimum_mse = minimum_mse.repeat(1, 1, feat_dim).long()
    new_target = torch.gather(target, 1, minimum_mse)  # (bs, k, features)
    loss = MSELoss(prediction, new_target, k=1, reduction="mean")

    return loss


def DiffusionLoss(
        output_prior,
        output_decoder,
        # ['KApproMSELoss', 'KApproMSELoss'] | [MSELoss, MSELoss] | [L1Loss, L1Loss] | ['...', 'MSELossWithAct']
        losses_type=['MSELoss', 'MSELoss'], # MSELossWithAct for decoder training.
        losses_multipliers=[1, 1],
        losses_decoded=[False, True],
        k=10,  # k appropriate reactions
        temporal_loss_w=0.1,  # loss weight
        **kwargs):
    encoded_prediction = output_prior["encoded_prediction"].squeeze()  # shape: (batch_size, k, encoded_dim)
    encoded_target = output_prior["encoded_target"].squeeze()  # shape: (batch_size, k, encoded_dim)
    prediction_emotion = output_decoder["prediction_emotion"]  # shape: (batch_size, k, window_size, dim==25)
    target_emotion = output_decoder["target_emotion"]  # shape: (batch_size, k, window_size, dim==25)

    _, _, window_size, emotion_dim = prediction_emotion.shape
    # compute losses
    losses_dict = {"loss": 0.0}

    losses_dict["temporal_loss"] = TemporalLoss(prediction_emotion.reshape(-1, window_size, emotion_dim))
    losses_dict["loss"] += losses_dict["temporal_loss"] * temporal_loss_w
    assert temporal_loss_w <= 0.0, "we first disregard temporal loss."

    # join last two dimensions of prediction and target
    prediction_emotion = prediction_emotion.reshape(-1, k, window_size * emotion_dim)
    target_emotion = target_emotion.reshape(-1, k, window_size * emotion_dim)

    # reconstruction loss
    for loss_name, w, decoded in zip(losses_type, losses_multipliers, losses_decoded):
        # loss_final_name = loss_name + f"_{'decoded' if decoded else 'encoded'}"
        loss_final_name = f"{'decoded' if decoded else 'encoded'}"

        if decoded:
            losses_dict[loss_final_name] = eval(loss_name)(prediction_emotion, target_emotion, k=k)
        else:
            losses_dict[loss_final_name] = eval(loss_name)(encoded_prediction, encoded_target, k=k)

        losses_dict["loss"] += losses_dict[loss_final_name] * w

    return losses_dict


def WeightedDiffusionLoss(
        output_prior,
        output_decoder,
        # ['KApproMSELoss', 'KApproMSELoss'] | [MSELoss, MSELoss] | [L1Loss, L1Loss]
        losses_type=['MSELoss', 'MSELoss'],
        losses_multipliers=[1, 1],
        losses_decoded=[False, True],
        k=10,  # k appropriate reactions
        temporal_loss_w=0.1,  # loss weight
        **kwargs):
    pass
