from tqdm import tqdm
import os
import numpy as np
import torch
from datetime import datetime
import yaml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.backends import cudnn


def init_seed(seed, rank=0):
    """Set up seed for each GPU."""
    process_seed = seed + rank
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    np.random.seed(process_seed)
    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load_config_from_file(path):
    return OmegaConf.load(path)

def load_config(args=None, config_path=None):
    if args is not None:
        config_from_args = OmegaConf.create(vars(args))
    else:
        config_from_args = OmegaConf.from_cli()
    # config_from_file = OmegaConf.load(cli_conf.pop('config') if config_path is None else config_path)
    config_from_file = load_config_from_file(config_path)
    return OmegaConf.merge(config_from_file, config_from_args)


def store_config(config):
    # store config to directory
    dir = config.trainer.out_dir
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(config), f)


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(0, 2, 3, 1)


def torch_img_to_np2(img):
    img = img.detach().cpu().numpy()
    # img = img * np.array([0.229, 0.224, 0.225]).reshape(1,-1,1,1)
    # img = img + np.array([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
    img = img * np.array([0.5, 0.5, 0.5]).reshape(1,-1,1,1)
    img = img + np.array([0.5, 0.5, 0.5]).reshape(1,-1,1,1)
    img = img.transpose(0, 2, 3, 1)
    img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]

    return img


def _fix_image(image):
    if image.max() < 30.:
        image = image * 255.
    image = np.clip(image, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]
    return image


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def collect_grad_value_(parameters):
    grad_values = []
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        grad_values.append(p.grad.data.abs().mean().item())
    grad_values = np.array(grad_values)
    return grad_values


def get_tensorboard_path(tb_path):
    current_time = datetime.now()
    time_str = str(current_time)
    time_str = '-'.join(time_str.split(' '))
    time_str = time_str.split('.')[0]
    tb_path = os.path.join(tb_path, time_str)
    os.makedirs(tb_path, exist_ok=True)
    return tb_path


def get_logging_path(log_dir):
    current_time = datetime.now()
    time_str = str(current_time)
    time_str = '-'.join(time_str.split(' '))
    time_str = time_str.split('.')[0]
    path = os.path.join(log_dir, time_str)
    return path


def plot_supcon_similarity(feat, label, writer, iter, split, save_dir):
    """
    :param feat: [N, dim]
    :param mask: [N, N]
    """
    feat = feat.detach().cpu()
    sim = torch.matmul(feat, feat.T)

    label = label.detach().cpu().contiguous().view(-1, 1)
    mask = torch.eq(label, label.T).bool() # shape: (N, N)
    _mask = ~mask
    positives = torch.sum(sim * mask, dim=-1) / torch.sum(mask, dim=-1)
    negatives = torch.sum(sim * _mask, dim=-1) / torch.sum(_mask, dim=-1)

    writer.add_scalar("{}/Similarity/positives".format(split), torch.nanmean(positives), iter)
    writer.add_scalar("{}/Similarity/negatives".format(split), torch.nanmean(negatives), iter)

    similarity = sim.numpy()
    mask = mask.numpy()
    np.save(os.path.join(save_dir, 'iter_' + str(iter+1) + '_similarity.npy'), similarity)
    np.save(os.path.join(save_dir, 'iter_' + str(iter+1) + '_mask.npy'), mask)


def tsne_visualisation(cfg, listener_features, listeners_label, epoch, split, exp):
    unique_values = torch.unique(listeners_label)
    listener_features = listener_features.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=10) # default 30
    tsne_results = tsne.fit_transform(listener_features)

    colors = plt.cm.get_cmap('tab10', len(unique_values))
    color_list = [colors(i) for i in range(len(unique_values))]

    plt.figure(figsize=(10, 6))
    for i, value in enumerate(unique_values):
        x = tsne_results[listeners_label == value][:, 0]
        y = tsne_results[listeners_label == value][:, 1]
        plt.scatter(x, y, color=color_list[i], label='Person x_{} reactions'.format(i))

    plt.title('t-SNE visualization of Listener Reactions')
    plt.legend()

    save_dir = os.path.join(cfg.similarity_outdir, 't-sne', 'exp_' + str(exp), split)
    os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(args.outdir, 't-sne', f'e{epoch + 1}_b{batch_idx + 1}_t-sne.png')
    save_path = os.path.join(save_dir, f'e{epoch + 1}_t-sne.png')
    plt.savefig(save_path)
    plt.close()


def tensorboard_log_personal(writer, iter, loss, p_rate, n_rate, grad_values=None, is_train=True):
    if is_train:
        writer.add_scalar('Train/train_loss', loss, iter)
        writer.add_scalar('Train/train_p_rate', p_rate, iter)
        writer.add_scalar('Train/train_n_rate', n_rate, iter)
    else:
        writer.add_scalar('Val/val_loss', loss, iter)
        writer.add_scalar('Val/val_p_rate', p_rate, iter)
        writer.add_scalar('Val/val_n_rate', n_rate, iter)

    if grad_values is not None:
        writer.add_scalar('grad_mean', grad_values.mean(), iter)
        writer.add_scalar('grad_max', grad_values.max(), iter)


def save_checkpoint_pretrain(cfg, model, optimizer, epoch, is_best):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    # save_path = cfg.trainer.checkpoint_dir
    save_dir = os.path.join(cfg.trainer.checkpoint_dir, 'exp_' + str(cfg.exp_num))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'epoch_' + "{:03d}".format(epoch) + '_checkpoint.pth')
    torch.save(checkpoint, save_path)

    if is_best:
        save_path = os.path.join(save_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, save_path)


def save_checkpoint(cfg, model, optimizer, epoch, is_best):
    diffusion_prior_model = model.diffusion_prior.model
    diffusion_prior_name = diffusion_prior_model.get_model_name()
    diffusion_prior_dir = os.path.join(cfg.trainer.checkpoint_dir, diffusion_prior_name)

    prior_checkpoint = {
        'epoch': epoch + 1,
        'state_dict': diffusion_prior_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_dir = os.path.join(diffusion_prior_dir, 'exp_' + str(cfg.exp_num))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'epoch_' + "{:03d}".format(epoch) + '_checkpoint.pth')
    torch.save(prior_checkpoint, save_path)
    if is_best:
        save_path = os.path.join(save_dir, 'best_checkpoint.pth')
        torch.save(prior_checkpoint, save_path)

    # diffusion decoder net
    diffusion_decoder_model = model.diffusion_decoder.model
    diffusion_decoder_name = diffusion_decoder_model.get_model_name()
    diffusion_decoder_dir = os.path.join(cfg.trainer.checkpoint_dir, diffusion_decoder_name)

    decoder_checkpoint = {
        'epoch': epoch + 1,
        'state_dict': diffusion_decoder_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_dir = os.path.join(diffusion_decoder_dir, 'exp_' + str(cfg.exp_num))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'epoch_' + "{:03d}".format(epoch) + '_checkpoint.pth')
    torch.save(decoder_checkpoint, save_path)
    if is_best:
        save_path = os.path.join(save_dir, 'best_checkpoint.pth')
        torch.save(decoder_checkpoint, save_path)


def checkpoint_load(cfg, model, device):
    checkpoint_dir = cfg.trainer.checkpoint_dir
    model_name = model.get_model_name()
    model_dir = os.path.join(checkpoint_dir, model_name, 'exp_' + str(cfg.exp_num))
    model_path = os.path.join(model_dir, 'epoch_' + "{:03d}".format(cfg.epoch_num) + '_checkpoint.pth')
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)


def checkpoint_resume(cfg, model, device):
    model_name = model.get_model_name()
    model_path = os.path.join(cfg.trainer.checkpoint_dir, model_name, cfg.trainer.resume)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    print("Successfully resume model: ", model_name)


def optimizer_resume(cfg, model, optimizer, device):
    model_name = model.diffusion_prior.model.get_model_name() # or diffusion decoder
    model_path = os.path.join(cfg.trainer.checkpoint_dir, model_name, cfg.trainer.resume)
    checkpoint = torch.load(model_path, map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer.to(device)
    print("Successfully resume optimizer: ", model_name)


def compute_statistics(config, model, data_loader, device):
    checkpoint_path = config.trainer.resume

    # reload checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch_idx, (_, _, reaction_emotion, _) in enumerate(tqdm(data_loader)):
            reaction_emotion = reaction_emotion.to(device)
            prediction = model.encode_all(reaction_emotion)
            preds.append(prediction.detach().clone())

    preds = torch.cat(preds, axis=0)
    checkpoint["statistics"] = {
        "min": preds.min(axis=0).values,
        "max": preds.max(axis=0).values,
        "mean": preds.mean(axis=0),
        "std": preds.std(axis=0),
        "var": preds.var(axis=0),
    }

    torch.save(checkpoint, config.resume) # add 'statistics'


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def collect_grad_stats(parameters):
    grad_values = []
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        # Store the absolute values of gradients
        grad_values.extend(p.grad.data.abs().view(-1).cpu().numpy())

    # Convert to a numpy array for statistical computation
    grad_values = np.array(grad_values)
    if grad_values.size == 0:
        return {"min": None, "max": None, "mean": None}

    # Compute min, max, and mean
    grad_stats = {
        "min": grad_values.min(),
        "max": grad_values.max(),
        "mean": grad_values.mean()
    }
    return grad_stats
