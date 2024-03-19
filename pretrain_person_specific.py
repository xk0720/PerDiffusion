import logging
import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import losses
import model as module_arch
from dataset.dataset_personal import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from utils.util import (AverageMeter, tsne_visualisation, tensorboard_log_personal, get_tensorboard_path,
                        get_logging_path, plot_supcon_similarity, collect_grad_value_, get_lr, load_config, init_seed,
                        save_checkpoint_pretrain)


def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--exp_num", type=int, help="the number of the experiment.", required=True)
    parser.add_argument("--similarity_outdir", type=str, default="./results/personal")
    args = parser.parse_args()
    return args


# Train
def train(cfg, device, model, train_loader, optimizer, scheduler, criterion, epoch, writer=None):
    losses = AverageMeter()

    model.train()

    if epoch == 0 or (epoch + 1) % cfg.trainer.val_period == 0:
        debug = True  # for debug
    else:
        debug = False

    save_dir = os.path.join(cfg.similarity_outdir, 'contrast', 'exp_' + str(cfg.exp_num), 'Train')
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, (listener_emotion_clip_personal,
                    listener_3dmm_clip_personal,
                    listeners_label_personal) in enumerate(tqdm(train_loader)):

        (listener_emotion_clip_personal,
         listener_3dmm_clip_personal,
         listeners_label_personal) = (
            listener_emotion_clip_personal.to(device),
            listener_3dmm_clip_personal.to(device),
            listeners_label_personal.to(device))

        if len(listener_3dmm_clip_personal) < 1 or len(torch.unique(listeners_label_personal)) < 2:
            # no samples or no negative samples
            continue

        # length_recording = [len(e) for e in listener_emotion_clip_personal]
        # listener_emotion_clip_personal = torch.cat(listener_emotion_clip_personal, dim=0)
        if cfg.trainer.data_aug: # adding the noise
            # TODO: following the Supervised Contrastive Learning, make n_views == 2.
            noise = torch.clamp(torch.randn_like(listener_3dmm_clip_personal) * 0.1, -0.2, 0.2)
            listener_3dmm_clip_personal = listener_3dmm_clip_personal + noise

        personal_feat, personal_proj = model(listener_3dmm_clip_personal)
        # personal_proj = torch.split(personal_proj, length_recording)

        # SupConLoss
        loss = criterion(feature=personal_proj, label=listeners_label_personal, device=device)
        losses.update(loss.item(), personal_proj.shape[0])

        optimizer.zero_grad()
        loss.backward()
        if cfg.trainer.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
        grad_values = collect_grad_value_(model.parameters())
        optimizer.step()

        if debug:
            # plot positive and negative avg similarity
            plot_supcon_similarity(personal_proj, listeners_label_personal.detach().cpu(),
                                   writer, epoch, 'Train', save_dir)
            # t-SNE distribution visualization
            tsne_visualisation(cfg, personal_proj, listeners_label_personal.detach().cpu(),
                               epoch, 'Train', exp=cfg.exp_num)
            debug = False

        if writer is not None:
            iters = epoch * len(train_loader) + batch_idx
            tensorboard_log_personal(writer, iters, loss, 0, 0, grad_values, is_train=True)

    # warmup for the first 5 epochs
    if scheduler is not None and (epoch + 1) >= 5:
        scheduler.step()

    # obtain the learning rate
    lr = get_lr(optimizer=optimizer)
    if writer is not None:
        writer.add_scalar("Train/lr", lr, epoch)

    return losses.avg


# Validation
def val(cfg, device, model, val_loader, criterion, epoch, writer):
    losses = AverageMeter()

    model.eval()

    debug = True # for debug
    save_dir = os.path.join(cfg.similarity_outdir, 'contrast', 'exp_' + str(cfg.exp_num), 'Val')
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, (listener_emotion_clip_personal,
                    listener_3dmm_clip_personal,
                    listeners_label_personal) in enumerate(tqdm(val_loader)):

        (listener_emotion_clip_personal,
         listener_3dmm_clip_personal,
         listeners_label_personal) = (
            listener_emotion_clip_personal.to(device),
            listener_3dmm_clip_personal.to(device),
            listeners_label_personal.to(device))

        if len(listener_3dmm_clip_personal) < 1 or len(torch.unique(listeners_label_personal)) < 2:
            # no samples or no negative samples
            continue

        with torch.no_grad():
            personal_feat, personal_proj = model(listener_3dmm_clip_personal)

            # SupConLoss
            loss = criterion(feature=personal_proj, label=listeners_label_personal, device=device)
            losses.update(loss.item(), personal_proj.shape[0])

            if debug:
                # plot positive and negative avg similarity in validation stage
                plot_supcon_similarity(personal_proj, listeners_label_personal.detach().cpu(),
                                       writer, epoch, 'Val', save_dir)
                # t-SNE distribution visualization
                tsne_visualisation(cfg, personal_proj, listeners_label_personal.detach().cpu(),
                                   epoch, 'Val', exp=cfg.exp_num)
                debug = False

    return losses.avg


def main(args, config_path):
    cfg = load_config(args=args, config_path=config_path)
    init_seed(seed=cfg.trainer.seed)  # seed initialization
    lowest_val_loss = 10000

    logging_path = get_logging_path(cfg.trainer.log_dir)
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(filename=logging_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    writer_path = get_tensorboard_path(cfg.trainer.tb_dir)
    writer = SummaryWriter(writer_path)

    # load listener video for contrastive learning
    train_loader = get_dataloader(cfg.dataset)
    val_loader = get_dataloader(cfg.validation_dataset)

    # Set device ordinal if GPUs are available
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda:0') # Adjust the device ordinal as needed
    else:
        device = torch.device('cpu')
    model = getattr(module_arch, cfg.model.type)(**cfg.model.args, device=device)

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

    criterion = getattr(losses, cfg.loss.type)(**cfg.loss.args)

    if cfg.optimizer.scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
    else:
        scheduler = None

    for epoch in range(cfg.trainer.start_epoch, cfg.trainer.epochs):
        train_loss = train(cfg, device, model, train_loader, optimizer, scheduler, criterion, epoch, writer)
        logging.info("Epoch: {} train_loss: {:.5f}".format(epoch + 1, train_loss))

        if (epoch + 1) % cfg.trainer.val_period == 0 or epoch < 9:
            val_loss = val(cfg, device, model, val_loader, criterion, epoch, writer)

            if writer is not None:
                tensorboard_log_personal(writer, epoch, val_loss, 0, 0, grad_values=None, is_train=False)

            logging.info("Epoch: {} val_loss: {:.5f}".format(epoch + 1, val_loss))

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                save_checkpoint_pretrain(cfg, model, optimizer, epoch, is_best=True)
            else:
                save_checkpoint_pretrain(cfg, model, optimizer, epoch, is_best=False)

if __name__ == "__main__":
    main(args=parse_arg(), config_path="config/person_specific.yaml")
