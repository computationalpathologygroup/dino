import sys
import math
import tqdm
import torch
import torch.nn as nn

from pathlib import Path
from collections import defaultdict

import dino.models.vision_transformer as vits

from dino.log import MetricLogger
from dino.distributed import is_main_process
from dino.eval.knn import extract_multiple_features, knn_classifier
from dino.utils.utils import load_weights, clip_gradients, cancel_gradients_last_layer


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    nepochs,
    fp16_scaler,
    clip_grad,
    freeze_last_layer,
    gpu_id,
):
    metric_logger = MetricLogger(delimiter="  ")
    with tqdm.tqdm(
        data_loader,
        desc=(f"Epoch [{epoch+1}/{nepochs}]"),
        unit=" img",
        ncols=80,
        unit_scale=data_loader.batch_size,
        leave=False,
        file=sys.stdout,
        disable=not (gpu_id in [-1, 0]),
    ) as t:
        for it, (images, _) in enumerate(t):
            # update weight decay and learning rate according to their schedule
            it = len(data_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            # move images to gpu
            if gpu_id == -1:
                images = [im.cuda(non_blocking=True) for im in images]
            else:
                device = torch.device(f"cuda:{gpu_id}")
                images = [im.to(device, non_blocking=True) for im in images]
            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                teacher_output = teacher(
                    images[:2]
                )  # only the 2 global views pass through the teacher
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output, epoch)

            if not math.isfinite(loss.item()):
                tqdm.tqdm.write(
                    "Loss is {}, stopping training".format(loss.item()), force=True
                )
                sys.exit(1)

            # student update
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                if clip_grad:
                    _ = clip_gradients(student, clip_grad)
                cancel_gradients_last_layer(epoch, student, freeze_last_layer)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if clip_grad:
                    fp16_scaler.unscale_(
                        optimizer
                    )  # unscale the gradients of optimizer's assigned params in-place
                    _ = clip_gradients(student, clip_grad)
                cancel_gradients_last_layer(epoch, student, freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                if torch.cuda.device_count() > 1:
                    student_params = student.module.parameters()
                else:
                    student_params = student.parameters()
                for param_q, param_k in zip(
                    student_params, teacher_without_ddp.parameters()
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(gpu_id)
    # print("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return train_stats


def tune_one_epoch(
    epoch,
    student: nn.Module,
    teacher: nn.Module,
    query_dataloader,
    test_dataloader,
    features_dir: Path,
    arch: str,
    patch_size: int,
    drop_path_rate: float,
    k: int,
    temperature: float,
    distributed: bool,
    save_features: bool = False,
    use_cuda: bool = False,
):
    student_model = vits.__dict__[arch](
        patch_size=patch_size, drop_path_rate=drop_path_rate, num_classes=0
    )
    teacher_model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    tqdm.tqdm.write(f"Teacher & student models {arch} {patch_size}x{patch_size} built.")
    student_model.cuda()
    teacher_model.cuda()
    tqdm.tqdm.write(f"Loading epoch {epoch} weights...")
    student_weights = student.state_dict()
    teacher_weights = teacher.state_dict()
    load_weights(student_model, student_weights)
    load_weights(teacher_model, teacher_weights)
    student_model.eval()
    teacher_model.eval()

    # ============ extract student features ============
    tqdm.tqdm.write("Extracting features for query set...")
    query_features, query_labels = extract_multiple_features(
        student_model, teacher_model, query_dataloader, distributed, use_cuda
    )
    tqdm.tqdm.write("Extracting features for test set...")
    test_features, test_labels = extract_multiple_features(
        student_model, teacher_model, test_dataloader, distributed, use_cuda
    )

    teacher_query_features, teacher_test_features = (
        query_features["teacher"],
        test_features["teacher"],
    )
    student_query_features, student_test_features = (
        query_features["student"],
        test_features["student"],
    )

    # save features and labels
    if save_features and is_main_process():
        for name, feats in query_features.items():
            torch.save(feats.cpu(), Path(features_dir, f"{name}_query_feat.pth"))
        for name, feats in query_features.items():
            torch.save(feats.cpu(), Path(features_dir, f"{name}_test_feat.pth"))
        torch.save(query_labels.cpu(), Path(features_dir, "query_labels.pth"))
        torch.save(test_labels.cpu(), Path(features_dir, "test_labels.pth"))

    results = defaultdict(dict)
    if is_main_process():
        assert len(torch.unique(query_labels)) == len(
            torch.unique(test_labels)
        ), "query & test dataset have different number of classes!"
        num_classes = len(torch.unique(query_labels))
        if use_cuda:
            teacher_query_features, teacher_test_features = (
                teacher_query_features.cuda(),
                teacher_test_features.cuda(),
            )
            student_query_features, student_test_features = (
                student_query_features.cuda(),
                student_test_features.cuda(),
            )
            query_labels, test_labels = query_labels.cuda(), test_labels.cuda()

        tqdm.tqdm.write("Features are ready!\nStarting kNN classification.")
        teacher_acc, teacher_auc = knn_classifier(
            teacher_query_features,
            query_labels,
            teacher_test_features,
            test_labels,
            k,
            temperature,
            num_classes,
        )
        student_acc, student_auc = knn_classifier(
            student_query_features,
            query_labels,
            student_test_features,
            test_labels,
            k,
            temperature,
            num_classes,
        )
        results["teacher"].update({"acc": teacher_acc, "auc": teacher_auc})
        results["student"].update({"acc": student_acc, "auc": student_auc})

    return results
