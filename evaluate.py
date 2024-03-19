import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import logging
from utils.render import Render
from metric import *
from dataset.dataset import get_dataloader
from utils.util import load_config, init_seed, get_logging_path
import model as module_arch


def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    # for loading the trained-model weights.
    parser.add_argument("--epoch_num", type=int, help="epoch number of saving model weight", required=True)
    parser.add_argument("--exp_num", type=int, help="the number of training experiment.", required=True)
    parser.add_argument("--mode", type=str, help="train (val) or test", required=True)
    parser.add_argument("--config", type=str, help="config path", required=True)
    parser.add_argument("--evaluate_log_dir", type=str, default="./log/evaluate")  # evaluate
    # parser.add_argument("--split", type=str, default="test", help="test | val")
    args = parser.parse_args()
    return args


def evaluate(cfg, device, model, test_loader, split, binarize=False):
    model.eval()

    out_dir = os.path.join(cfg.trainer.out_dir, split, 'exp_' + str(cfg.exp_num))
    os.makedirs(out_dir, exist_ok=True)

    # save emotions
    speaker_emotion_list = []
    listener_emotion_gt_list = []
    listener_emotion_pred_list = []

    for batch_idx, (
            speaker_audio_clip,
            _, # speaker_video_clip # (bs, token_len, 3, 224, 224)
            speaker_emotion_clip,
            speaker_3dmm_clip,
            _ , # listener_video_clip  # (bs, token_len, 3, 224, 224)
            listener_emotion_clip,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            _, # listener_reference
    ) in enumerate(tqdm(test_loader)):

        (speaker_audio_clip,  # (bs, token_len, 78)
         speaker_emotion_clip,  # (bs, token_len, 25)
         speaker_3dmm_clip,  # (bs, token_len, 58)
         listener_emotion_clip,  # (bs, token_len, 25)
         listener_3dmm_clip,  # (bs, token_len, 58)
         # (bs * k, token_len, 58)
         listener_3dmm_clip_personal) = \
            (speaker_audio_clip.to(device),
             speaker_emotion_clip.to(device),
             speaker_3dmm_clip.to(device),
             listener_emotion_clip.to(device),
             listener_3dmm_clip.to(device),
             listener_3dmm_clip_personal.to(device))  # (bs, 3, 224, 224)

        listener_emotion_gt = listener_emotion_clip.detach().clone().cpu()
        # just for dimension compatibility during inference
        listener_emotion_clip = listener_emotion_clip.repeat_interleave(
            cfg.test_dataset.k_appro, dim=0)  # (bs * k, token_len, 25)
        listener_3dmm_clip = listener_3dmm_clip.repeat_interleave(
            cfg.test_dataset.k_appro, dim=0)  # (bs * k, token_len, 58)

        with torch.no_grad():
            _, listener_emotion_pred = model(
                speaker_audio=speaker_audio_clip,
                speaker_emotion_input=speaker_emotion_clip,
                speaker_3dmm_input=speaker_3dmm_clip,
                listener_emotion_input=listener_emotion_clip,
                listener_3dmm_input=listener_3dmm_clip,
                listener_personal_input=listener_3dmm_clip_personal,
            )
            listener_emotion_pred = listener_emotion_pred["prediction_emotion"]
            # shape: (bs, k_appro==10, seq_len==750, emo_dim==25)

            # TODO: Note: if we use activation during function, we also need to use at the infer stage.
            # AU = listener_emotion_pred[:, :, :15]
            # AU = torch.sigmoid(AU)
            # middle_feat = listener_emotion_pred[:, :, 15:17]
            # middle_feat = torch.tanh(middle_feat)
            # emotion = listener_emotion_pred[:, :, 17:]
            # emotion = torch.softmax(emotion, dim=-1)
            # listener_emotion_pred = torch.cat((AU, middle_feat, emotion), dim=-1)

            if binarize:
                listener_emotion_pred[:, :, :, :15] = torch.round(listener_emotion_pred[:, :, :, :15])

            # TODO: debug: save min, max, mean value
            # with open("/home/x/xk18/PhD_code_exp/project_react/results/train_main/test/exp_5/save_3dmm.txt", "a") as f:
            #     _3dmm_pred = listener_3dmm_pred.detach().cpu().numpy()
            #     min_value = np.min(_3dmm_pred)
            #     max_value = np.max(_3dmm_pred)
            #     mean_value = np.mean(_3dmm_pred)
            #     f.write("Max_value: {:.5f}  Min_value: {:.5f} Mean_value: {:.5f} \n".
            #             format(max_value, min_value, mean_value))
            # with open("/home/x/xk18/PhD_code_exp/project_react/results/train_main/test/exp_5/save_3dmm_gt.txt", "a") as f:
            #     _3dmm_pred = listener_3dmm_clip.detach().cpu().numpy()
            #     min_value = np.min(_3dmm_pred)
            #     max_value = np.max(_3dmm_pred)
            #     mean_value = np.mean(_3dmm_pred)
            #     f.write("Max_value: {:.5f}  Min_value: {:.5f} Mean_value: {:.5f} \n".
            #             format(max_value, min_value, mean_value))

            # Rendering
            # render.rendering_for_fid(
            #     out_dir,
            #     "{}_iter_{}".format(split, str(iteration + 1)),
            #     listener_3dmm_pred[0, 0],  # (750, 58)
            #     speaker_video_clip[0],  # (750, 3, 224, 224)
            #     listener_reference[0],  # (3, 224, 224)
            #     listener_video_clip[0],  # (750, 3, 224, 224)
            # )

            speaker_emotion_list.append(speaker_emotion_clip.detach().cpu())
            listener_emotion_pred_list.append(listener_emotion_pred.detach().cpu())
            listener_emotion_gt_list.append(listener_emotion_gt.detach().cpu())

            # if batch_idx > 5:
            #     break

    all_speaker_emotion = torch.cat(speaker_emotion_list, dim=0)
    # shape: (N, 750, 3dmm_dim)
    all_listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim=0)
    # shape: (N, 10, 750, emo_dim==25)
    all_listener_emotion_gt = torch.cat(listener_emotion_gt_list, dim=0)
    # shape: (N, 750, emo_dim==25)

    # TODO: Save the prediction and ground truth emotions.
    np.save(os.path.join(out_dir, "all_listener_emotion_pred.npy"),
            all_listener_emotion_pred.cpu().numpy())
    np.save(os.path.join(out_dir, "all_listener_emotion_gt.npy"),
            all_listener_emotion_gt.cpu().numpy())

    logging.info("-----------------Evaluating Metric-----------------")

    p = cfg.test_dataset.threads

    # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    TLCC = compute_TLCC_mp(all_listener_emotion_pred, all_speaker_emotion, p=p)

    # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
    FRC = compute_FRC_mp(cfg.test_dataset, all_listener_emotion_pred, all_listener_emotion_gt, val_test=split, p=p)

    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    FRD = compute_FRD_mp(cfg.test_dataset, all_listener_emotion_pred, all_listener_emotion_gt, val_test=split, p=p)

    FRDvs = compute_FRDvs(all_listener_emotion_pred)
    FRVar = compute_FRVar(all_listener_emotion_pred)
    smse = compute_s_mse(all_listener_emotion_pred)

    logging.info("FRC: {:.5f}  FRD: {:.5f}  FRDvs: {:.5f}  FRVar: {:.5f}  smse: {:.5f}  TLCC: {:.5f}"
                 .format(FRC, FRD, FRDvs, FRVar, smse, TLCC))

    return FRC, FRD, FRDvs, FRVar, smse, TLCC


def main(args):
    # load yaml config
    cfg = load_config(args=args, config_path=args.config)
    init_seed(seed=cfg.trainer.seed)  # seed initialization

    # logging
    logging_path = get_logging_path(cfg.evaluate_log_dir)
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(filename=logging_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    test_loader = get_dataloader(cfg.test_dataset)
    split = cfg.test_dataset.split

    # Set device ordinal if GPUs are available
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda:0')  # Adjust the device ordinal as needed
        # render = Render('cuda')
    else:
        device = torch.device('cpu')
        # render = Render()

    model = getattr(module_arch, cfg.trainer.model)(cfg, device)
    model.to(device)

    FRC, FRD, FRDvs, FRVar, smse, TLCC = evaluate(cfg, device, model, test_loader, split)

    print("FRC: {:.5f}  FRD: {:.5f}  FRDvs: {:.5f}  FRVar: {:.5f}  smse: {:.5f}  TLCC: {:.5f}"
          .format(FRC, FRD, FRDvs, FRVar, smse, TLCC))


if __name__ == '__main__':
    main(args=parse_arg())
