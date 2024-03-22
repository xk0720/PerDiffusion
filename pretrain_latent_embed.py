import logging
from functools import partial
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import argparse
from dataset.dataset_embedder import get_dataloader
from utils import losses
from torch.utils.tensorboard import SummaryWriter
from utils.util import load_config, init_seed, get_logging_path, get_tensorboard_path, AverageMeter, \
    save_checkpoint_pretrain, compute_statistics
import model as module_arch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Make only the first GPU visible
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging


def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--exp_num", type=int, help="the number of the experiment.", required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


def train(cfg, model, train_loader, optimizer, criterion, device):
    whole_losses = AverageMeter()
    rec_losses = AverageMeter()
    kld_losses = AverageMeter()
    div_losses = AverageMeter()

    model.train()
    for batch_idx, (_, _, reaction_emotion, reaction_3dmm) in enumerate(tqdm(train_loader)):
        reaction_emotion, reaction_3dmm = reaction_emotion.to(device), reaction_3dmm.to(device)

        output = model(emotion=reaction_emotion, _3dmm=reaction_3dmm)
        loss_output = criterion(**output)

        loss, rec_loss, kld_loss, div_loss = (
            loss_output["loss"], loss_output["mse"], loss_output["kld"], loss_output["coeff"]
        )

        whole_losses.update(loss.data.item(), reaction_emotion.size(0))
        rec_losses.update(rec_loss.data.item(), reaction_emotion.size(0))
        kld_losses.update(kld_loss.data.item(), reaction_emotion.size(0))
        div_losses.update(div_loss.data.item(), reaction_emotion.size(0))

        optimizer.zero_grad()
        loss.backward()
        if cfg.trainer.clip_grad:
            clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

    return whole_losses.avg, rec_losses.avg, kld_losses.avg, div_losses.avg


def validate(cfg, model, val_loader, criterion, device):
    whole_losses = AverageMeter()
    rec_losses = AverageMeter()
    kld_losses = AverageMeter()
    coeff_losses = AverageMeter()
    model.eval()

    for batch_idx, (_, _, reaction_emotion, reaction_3dmm) in enumerate(tqdm(val_loader)):
        reaction_emotion, reaction_3dmm = reaction_emotion.to(device), reaction_3dmm.to(device)

        with torch.no_grad():
            output = model(emotion=reaction_emotion, _3dmm=reaction_3dmm)
            loss_output = criterion(**output)

            loss, rec_loss, kld_loss, coeff_loss = (
                loss_output["loss"], loss_output["mse"], loss_output["kld"], loss_output["coeff"]
            )

            whole_losses.update(loss.data.item(), reaction_emotion.size(0))
            rec_losses.update(rec_loss.data.item(), reaction_emotion.size(0))
            kld_losses.update(kld_loss.data.item(), reaction_emotion.size(0))
            coeff_losses.update(coeff_loss.data.item(), reaction_emotion.size(0))

    return whole_losses.avg, rec_losses.avg, kld_losses.avg, coeff_losses.avg


def main(args):
    # load yaml config
    cfg = load_config(args=args, config_path=args.config)

    init_seed(seed=cfg.trainer.seed)  # seed initialization
    lowest_val_loss = 10000

    # logging
    logging_path = get_logging_path(cfg.trainer.log_dir)
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(filename=logging_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    writer_path = get_tensorboard_path(cfg.trainer.tb_dir)
    writer = SummaryWriter(writer_path)

    train_loader = get_dataloader(cfg, dataset_path=cfg.dataset.dataset_path,
                                  split=cfg.dataset.split)
    val_loader = get_dataloader(cfg, dataset_path=cfg.validation_dataset.dataset_path,
                                split=cfg.validation_dataset.split)

    # Set device ordinal if GPUs are available
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda:0") # Adjust the device ordinal as needed
    else:
        device = torch.device("cpu")
    model = getattr(module_arch, cfg.model.type)(cfg.model.args)

    if cfg.trainer.resume is not None:
        checkpoint_path = cfg.trainer.resume
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)
    model.to(device)

    if cfg.optimizer.type == "adamW":
        optimizer = optim.AdamW(model.parameters(), betas=cfg.optimizer.args.beta, lr=cfg.optimizer.args.lr,
                                weight_decay=cfg.optimizer.args.weight_decay)
    elif cfg.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), cfg.optimizer.args.lr, weight_decay=cfg.optimizer.args.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), cfg.optimizer.args.lr, momentum=cfg.optimizer.args.momentum,
                              weight_decay=cfg.optimizer.args.weight_decay)
    else:
        NotImplemented("The optimizer {} not implemented.".format(cfg.optimizer.type))

    criterion = partial(getattr(losses, cfg.loss.type), **cfg.loss.args)

    for epoch in range(cfg.trainer.start_epoch, cfg.trainer.epochs):
        train_loss, rec_loss, kld_loss, coeff_loss = (
            train(cfg, model, train_loader, optimizer, criterion, device)
        )

        logging.info(
            "Epoch: {} train_whole_loss: {:.5f} train_rec_loss: {:.5f} train_kld_loss: {:.5f} train_coeff_loss: {:.5f}"
            .format(epoch + 1, train_loss, rec_loss, kld_loss, coeff_loss))

        writer.add_scalar("Train/whole_loss", train_loss, epoch)
        writer.add_scalar("Train/rec_loss", rec_loss, epoch)
        writer.add_scalar("Train/kld_loss", kld_loss, epoch)
        writer.add_scalar("Train/coeff_loss", coeff_loss, epoch)

        if (epoch + 1) % cfg.trainer.val_period == 0:
            val_loss, rec_loss, kld_loss, coeff_loss = validate(cfg, model, val_loader, criterion, device)

            logging.info(
                "Epoch: {} val_whole_loss: {:.5f} val_rec_loss: {:.5f} val_kld_loss: {:.5f} val_coeff_loss: {:.5f}"
                .format(epoch + 1, val_loss, rec_loss, kld_loss, coeff_loss))

            writer.add_scalar("Val/whole_loss", val_loss, epoch)
            writer.add_scalar("Val/rec_loss", rec_loss, epoch)
            writer.add_scalar("Val/kld_loss", kld_loss, epoch)
            writer.add_scalar("Val/coeff_loss", coeff_loss, epoch)

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                save_checkpoint_pretrain(cfg, model, optimizer, epoch, is_best=True)
            else:
                save_checkpoint_pretrain(cfg, model, optimizer, epoch, is_best=False)

    # Finally compute the statistics
    # cfg.trainer.resume = os.path.join(cfg.trainer.checkpoint_dir, 'best_checkpoint.pth')
    # compute_statistics(cfg, model, train_loader, device)

    writer.close()


if __name__ == '__main__':
    main(args=parse_arg())
