"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import json
import os
import numpy as np
import torch
import torch.distributed as dist
import pointops
import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu
from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class LossEvaluator(HookBase):
    def __init__(self, save_metric="neg_loss"):
        if save_metric != "neg_loss":
            raise ValueError(
                "LossEvaluator currently supports save_metric='neg_loss' only"
            )
        self.save_metric = save_metric

    def before_train(self):
        pass

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Loss Evaluation >>>>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()
        loss_sum = torch.zeros(1, device="cuda")
        loss_count = torch.zeros(1, device="cuda")
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"].detach()
            loss_sum += loss
            loss_count += 1
            self.trainer.logger.info(
                "Val: [{iter}/{max_iter}] Loss {loss:.6f} ".format(
                    iter=i + 1,
                    max_iter=len(self.trainer.val_loader),
                    loss=float(loss.item()),
                )
            )
        if comm.get_world_size() > 1:
            (dist.all_reduce(loss_sum), dist.all_reduce(loss_count))
        loss_avg = (loss_sum / loss_count.clamp_min(1.0)).item()
        current_epoch = self.trainer.epoch + 1
        self.trainer.logger.info("Val result: loss {:.6f}.".format(loss_avg))
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<<< End Loss Evaluation <<<<<<<<<<<<<<<<<"
        )
        self.trainer.comm_info["current_metric_value"] = -loss_avg
        self.trainer.comm_info["current_metric_name"] = "neg_loss"

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.6f}".format("neg_loss", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(
        self,
        write_cls_iou=False,
        write_confusion=False,
        extra_knn_ks=(1, 3),
        save_metric="main",
    ):
        self.write_cls_iou = write_cls_iou
        self.write_confusion = write_confusion
        self.extra_knn_ks = tuple(sorted({int(k) for k in extra_knn_ks if int(k) > 0}))
        self.save_metric = save_metric

    def before_train(self):
        pass

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    @staticmethod
    def _default_offset(coord, offset=None):
        if offset is not None:
            return offset.int()
        return torch.tensor([coord.shape[0]], device=coord.device, dtype=torch.int32)

    @staticmethod
    def _sanitize_metric_name(name):
        return name.replace("/", "_")

    @staticmethod
    def _new_metric_state(num_classes):
        return dict(
            intersection=np.zeros(num_classes, dtype=np.float64),
            union=np.zeros(num_classes, dtype=np.float64),
            target=np.zeros(num_classes, dtype=np.float64),
            confusion=np.zeros((num_classes, num_classes), dtype=np.int64),
        )

    def _accumulate_metric(
        self, metric_states, name, pred, segment, num_classes, ignore_index
    ):
        state = metric_states.setdefault(name, self._new_metric_state(num_classes))
        valid_mask = segment != ignore_index
        if valid_mask.any():
            batch_confusion = torch.bincount(
                segment[valid_mask].long() * num_classes + pred[valid_mask].long(),
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)
        else:
            batch_confusion = torch.zeros(
                (num_classes, num_classes), device=pred.device, dtype=torch.long
            )
        intersection, union, target = intersection_and_union_gpu(
            pred, segment, num_classes, ignore_index
        )
        if comm.get_world_size() > 1:
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target)
            dist.all_reduce(batch_confusion)
        state["intersection"] += intersection.cpu().numpy()
        state["union"] += union.cpu().numpy()
        state["target"] += target.cpu().numpy()
        state["confusion"] += batch_confusion.cpu().numpy().astype(np.int64)

    def _metric_result(self, state):
        iou_class = state["intersection"] / (state["union"] + 1e-10)
        acc_class = state["intersection"] / (state["target"] + 1e-10)
        m_iou = float(np.mean(iou_class))
        m_acc = float(np.mean(acc_class))
        all_acc = float(state["intersection"].sum() / (state["target"].sum() + 1e-10))
        return dict(
            iou_class=iou_class,
            acc_class=acc_class,
            m_iou=m_iou,
            m_acc=m_acc,
            all_acc=all_acc,
            confusion=state["confusion"],
        )

    def _interpolate_logits_knn(
        self, sparse_logits, sparse_coord, sparse_offset, full_coord, full_offset, k
    ):
        if sparse_logits.numel() == 0:
            return sparse_logits.new_zeros(
                (full_coord.shape[0], sparse_logits.shape[-1])
            )
        k = max(1, min(int(k), int(sparse_coord.shape[0])))
        sparse_offset = self._default_offset(sparse_coord, sparse_offset)
        full_offset = self._default_offset(full_coord, full_offset)
        index, distance = pointops.knn_query(
            k,
            sparse_coord.float(),
            sparse_offset.int(),
            full_coord.float(),
            full_offset.int(),
        )
        index = index.long().reshape(-1, k)
        if k == 1:
            return sparse_logits[index.reshape(-1)]
        distance = distance.reshape(-1, k).to(sparse_logits.dtype)
        inv_distance = 1.0 / distance.clamp_min(1e-08)
        exact_mask = distance <= 1e-08
        if exact_mask.any():
            exact_weight = exact_mask.to(sparse_logits.dtype)
            has_exact = exact_mask.any(dim=1, keepdim=True)
            weights = torch.where(has_exact, exact_weight, inv_distance)
        else:
            weights = inv_distance
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-08)
        gathered = sparse_logits[index]
        return (gathered * weights.unsqueeze(-1)).sum(dim=1)

    @staticmethod
    def _align_full_coord_to_sparse_frame(
        sparse_coord, sparse_offset, full_coord, inverse
    ):
        """Align copied full coordinates with val-time sparse-coordinate shifts."""
        if inverse is None or sparse_coord.numel() == 0 or full_coord.numel() == 0:
            return full_coord
        sparse_offset = SemSegEvaluator._default_offset(sparse_coord, sparse_offset)
        inverse = inverse.long().reshape(-1)
        full_coord = full_coord.float()
        aligned = full_coord.clone()
        raw_index_all = torch.arange(
            inverse.numel(), device=inverse.device, dtype=torch.long
        )
        sparse_start = 0
        for sparse_end_tensor in sparse_offset.long():
            sparse_end = int(sparse_end_tensor.item())
            scene_mask = (inverse >= sparse_start) & (inverse < sparse_end)
            raw_index = raw_index_all[scene_mask]
            if raw_index.numel() == 0:
                sparse_start = sparse_end
                continue
            local_inverse = inverse[raw_index] - sparse_start
            scene_sparse_count = sparse_end - sparse_start
            first_raw = torch.full(
                (scene_sparse_count,),
                inverse.numel(),
                device=inverse.device,
                dtype=torch.long,
            )
            first_raw.scatter_reduce_(
                0, local_inverse, raw_index, reduce="amin", include_self=True
            )
            valid = first_raw < inverse.numel()
            if not torch.any(valid):
                sparse_start = sparse_end
                continue
            shift = (
                (
                    full_coord[first_raw[valid]]
                    - sparse_coord[sparse_start:sparse_end][valid].float()
                )
                .median(dim=0)
                .values
            )
            aligned[raw_index] = full_coord[raw_index] - shift
            sparse_start = sparse_end
        return aligned

    def _build_aux_views(self, input_dict, output_dict):
        views = {}
        if "inverse" in input_dict:
            sparse_logits = output_dict["seg_logits"]
            views["subset_sparse"] = dict(
                logits=sparse_logits, segment=input_dict["segment"]
            )
            if "origin_coord" in input_dict and "origin_segment" in input_dict:
                sparse_coord = input_dict["coord"]
                sparse_offset = input_dict.get("offset")
                full_coord = self._align_full_coord_to_sparse_frame(
                    sparse_coord,
                    sparse_offset,
                    input_dict["origin_coord"],
                    input_dict.get("inverse"),
                )
                full_segment = input_dict["origin_segment"]
                for k in self.extra_knn_ks:
                    name = "full_knn1" if k == 1 else f"full_knn{k}_idw"
                    views[name] = dict(
                        logits=self._interpolate_logits_knn(
                            sparse_logits,
                            sparse_coord,
                            sparse_offset,
                            full_coord,
                            None,
                            k,
                        ),
                        segment=full_segment,
                    )
        if "eval_sparse_logits" in output_dict:
            sparse_logits = output_dict["eval_sparse_logits"]
            sparse_segment = output_dict.get("eval_sparse_segment")
            if sparse_segment is not None:
                views["subset_sparse"] = dict(
                    logits=sparse_logits, segment=sparse_segment
                )
            if "eval_sparse_coord" in output_dict:
                sparse_coord = output_dict["eval_sparse_coord"]
                sparse_offset = output_dict.get("eval_sparse_offset")
                full_coord = input_dict["coord"]
                full_offset = input_dict.get("offset")
                full_segment = input_dict["segment"]
                main_is_full = (
                    "inverse" not in input_dict
                    and output_dict["seg_logits"].shape[0] == full_segment.shape[0]
                )
                for k in self.extra_knn_ks:
                    if k == 1 and main_is_full:
                        continue
                    name = "full_knn1" if k == 1 else f"full_knn{k}_idw"
                    views[name] = dict(
                        logits=self._interpolate_logits_knn(
                            sparse_logits,
                            sparse_coord,
                            sparse_offset,
                            full_coord,
                            full_offset,
                            k,
                        ),
                        segment=full_segment,
                    )
        return views

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        num_classes = self.trainer.cfg.data.num_classes
        ignore_index = self.trainer.cfg.data.ignore_index
        metric_states = {"main": self._new_metric_state(num_classes)}
        sample_class_hist = None
        full_class_hist = None
        sampling_region_stats = {
            "boundary_full_count": 0.0,
            "boundary_sample_count": 0.0,
            "non_boundary_full_count": 0.0,
            "non_boundary_sample_count": 0.0,
        }
        tracked_sample_keys = (
            "sample_count",
            "target_sample_count",
            "sample_ratio_actual",
            "target_sample_ratio",
        )
        sample_sums = {key: 0.0 for key in tracked_sample_keys}
        sample_counts = {key: 0 for key in tracked_sample_keys}
        loss_sum = 0.0
        loss_count = 0
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "inverse" in input_dict.keys():
                assert "origin_segment" in input_dict.keys()
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]
            self._accumulate_metric(
                metric_states, "main", pred, segment, num_classes, ignore_index
            )
            aux_views = self._build_aux_views(input_dict, output_dict)
            for name, view in aux_views.items():
                aux_pred = view["logits"].max(1)[1]
                self._accumulate_metric(
                    metric_states,
                    name,
                    aux_pred,
                    view["segment"],
                    num_classes,
                    ignore_index,
                )
            if "sample_class_hist" in output_dict and "full_class_hist" in output_dict:
                batch_sample_hist = output_dict["sample_class_hist"].clone()
                batch_full_hist = output_dict["full_class_hist"].clone()
                if comm.get_world_size() > 1:
                    dist.all_reduce(batch_sample_hist)
                    dist.all_reduce(batch_full_hist)
                batch_sample_hist = batch_sample_hist.cpu().numpy()
                batch_full_hist = batch_full_hist.cpu().numpy()
                sample_class_hist = (
                    batch_sample_hist
                    if sample_class_hist is None
                    else sample_class_hist + batch_sample_hist
                )
                full_class_hist = (
                    batch_full_hist
                    if full_class_hist is None
                    else full_class_hist + batch_full_hist
                )
            for key in sampling_region_stats:
                if key in output_dict:
                    value = output_dict[key].clone()
                    if comm.get_world_size() > 1:
                        dist.all_reduce(value)
                    sampling_region_stats[key] += float(value.item())
            for key in tracked_sample_keys:
                if key in output_dict:
                    value = output_dict[key].clone().detach().float()
                    if comm.get_world_size() > 1:
                        dist.all_reduce(value)
                    sample_sums[key] += float(value.item())
                    sample_counts[key] += 1
            loss_sum += float(loss.item())
            loss_count += 1
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(info + "Loss {loss:.4f} ".format(loss=loss.item()))
        loss_avg = loss_sum / max(loss_count, 1)
        metric_results = {
            name: self._metric_result(state) for name, state in metric_states.items()
        }
        main_result = metric_results["main"]
        iou_class = main_result["iou_class"]
        acc_class = main_result["acc_class"]
        sample_stats = {
            key: sample_sums[key] / sample_counts[key]
            for key in tracked_sample_keys
            if sample_counts[key] > 0
        }
        class_keep_rate = None
        if sample_class_hist is not None and full_class_hist is not None:
            class_keep_rate = sample_class_hist / np.clip(full_class_hist, 1.0, None)
        boundary_keep_rate = None
        non_boundary_keep_rate = None
        boundary_bias = None
        if sampling_region_stats["boundary_full_count"] > 0:
            boundary_keep_rate = (
                sampling_region_stats["boundary_sample_count"]
                / sampling_region_stats["boundary_full_count"]
            )
        if sampling_region_stats["non_boundary_full_count"] > 0:
            non_boundary_keep_rate = (
                sampling_region_stats["non_boundary_sample_count"]
                / sampling_region_stats["non_boundary_full_count"]
            )
        if (
            boundary_keep_rate is not None
            and non_boundary_keep_rate is not None
            and (non_boundary_keep_rate > 0)
        ):
            boundary_bias = boundary_keep_rate / non_boundary_keep_rate
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                main_result["m_iou"], main_result["m_acc"], main_result["all_acc"]
            )
        )
        for name, result in metric_results.items():
            if name == "main":
                continue
            self.trainer.logger.info(
                "Val[{name}] result: mIoU/mAcc/allAcc {m_iou:.4f}/{m_acc:.4f}/{all_acc:.4f}.".format(
                    name=name,
                    m_iou=result["m_iou"],
                    m_acc=result["m_acc"],
                    all_acc=result["all_acc"],
                )
            )
        if sample_stats:
            self.trainer.logger.info(
                "Val sampling: sample_count {:.1f} target_sample_count {:.1f} sample_ratio_actual {:.4f} target_sample_ratio {:.4f}.".format(
                    sample_stats.get("sample_count", 0.0),
                    sample_stats.get("target_sample_count", 0.0),
                    sample_stats.get("sample_ratio_actual", 0.0),
                    sample_stats.get("target_sample_ratio", 0.0),
                )
            )
        if boundary_keep_rate is not None and non_boundary_keep_rate is not None:
            self.trainer.logger.info(
                "Val boundary sampling: boundary_keep_rate {:.4f} non_boundary_keep_rate {:.4f} boundary_bias {:.4f}.".format(
                    boundary_keep_rate,
                    non_boundary_keep_rate,
                    0.0 if boundary_bias is None else boundary_bias,
                )
            )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar(
                "val/mIoU", main_result["m_iou"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/mAcc", main_result["m_acc"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/allAcc", main_result["all_acc"], current_epoch
            )
            for name, result in metric_results.items():
                if name == "main":
                    continue
                safe_name = self._sanitize_metric_name(name)
                self.trainer.writer.add_scalar(
                    f"val/{safe_name}/mIoU", result["m_iou"], current_epoch
                )
                self.trainer.writer.add_scalar(
                    f"val/{safe_name}/mAcc", result["m_acc"], current_epoch
                )
                self.trainer.writer.add_scalar(
                    f"val/{safe_name}/allAcc", result["all_acc"], current_epoch
                )
            for key, value in sample_stats.items():
                self.trainer.writer.add_scalar(f"val/{key}", value, current_epoch)
            if boundary_keep_rate is not None:
                self.trainer.writer.add_scalar(
                    "val/boundary_keep_rate", boundary_keep_rate, current_epoch
                )
            if non_boundary_keep_rate is not None:
                self.trainer.writer.add_scalar(
                    "val/non_boundary_keep_rate", non_boundary_keep_rate, current_epoch
                )
            if boundary_bias is not None:
                self.trainer.writer.add_scalar(
                    "val/boundary_bias", boundary_bias, current_epoch
                )
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
        eval_filename_tag = self.trainer.comm_info.get("eval_filename_tag")
        metrics_stem = (
            f"eval_metrics_{eval_filename_tag}"
            if eval_filename_tag
            else f"eval_metrics_epoch{current_epoch:03d}"
        )
        confusion_stem = (
            f"confusion_{eval_filename_tag}"
            if eval_filename_tag
            else f"confusion_epoch{current_epoch:03d}"
        )
        sampling_stem = (
            f"sampling_{eval_filename_tag}"
            if eval_filename_tag
            else f"sampling_epoch{current_epoch:03d}"
        )
        if comm.is_main_process():
            summary_payload = {}
            for name, result in metric_results.items():
                summary_payload[name] = dict(
                    mIoU=float(result["m_iou"]),
                    mAcc=float(result["m_acc"]),
                    allAcc=float(result["all_acc"]),
                    iou_class=[float(x) for x in result["iou_class"]],
                    acc_class=[float(x) for x in result["acc_class"]],
                )
            metrics_json = os.path.join(
                self.trainer.cfg.save_path, f"{metrics_stem}.json"
            )
            with open(metrics_json, "w") as f:
                json.dump(summary_payload, f, indent=2)
            metrics_csv = os.path.join(
                self.trainer.cfg.save_path, f"{metrics_stem}.csv"
            )
            with open(metrics_csv, "w") as f:
                f.write("metric,mIoU,mAcc,allAcc\n")
                for name, result in metric_results.items():
                    f.write(
                        f"{name},{float(result['m_iou']):.8f},{float(result['m_acc']):.8f},{float(result['all_acc']):.8f}\n"
                    )
        if self.write_confusion and comm.is_main_process():
            main_confusion = metric_results["main"]["confusion"]
            save_stem = os.path.join(self.trainer.cfg.save_path, confusion_stem)
            np.save(save_stem + ".npy", main_confusion)
            np.savetxt(
                save_stem + ".csv",
                main_confusion.astype(np.int64),
                fmt="%d",
                delimiter=",",
            )
            for name, result in metric_results.items():
                if name == "main":
                    continue
                safe_name = self._sanitize_metric_name(name)
                if eval_filename_tag:
                    aux_stem = f"confusion_{safe_name}_{eval_filename_tag}"
                else:
                    aux_stem = f"confusion_{safe_name}_epoch{current_epoch:03d}"
                save_stem = os.path.join(self.trainer.cfg.save_path, aux_stem)
                np.save(save_stem + ".npy", result["confusion"])
                np.savetxt(
                    save_stem + ".csv",
                    result["confusion"].astype(np.int64),
                    fmt="%d",
                    delimiter=",",
                )
        if (
            sample_class_hist is not None
            and full_class_hist is not None
            and comm.is_main_process()
        ):
            sampling_csv = os.path.join(
                self.trainer.cfg.save_path, f"{sampling_stem}.csv"
            )
            with open(sampling_csv, "w") as f:
                f.write("class_idx,class_name,full_count,sample_count,keep_rate\n")
                for i, name in enumerate(self.trainer.cfg.data.names):
                    f.write(
                        f"{i},{name},{int(full_class_hist[i])},{int(sample_class_hist[i])},{float(class_keep_rate[i]):.8f}\n"
                    )
            sampling_summary = dict(
                boundary_full_count=sampling_region_stats["boundary_full_count"],
                boundary_sample_count=sampling_region_stats["boundary_sample_count"],
                boundary_keep_rate=boundary_keep_rate,
                non_boundary_full_count=sampling_region_stats[
                    "non_boundary_full_count"
                ],
                non_boundary_sample_count=sampling_region_stats[
                    "non_boundary_sample_count"
                ],
                non_boundary_keep_rate=non_boundary_keep_rate,
                boundary_bias=boundary_bias,
            )
            sampling_json = os.path.join(
                self.trainer.cfg.save_path, f"{sampling_stem}.json"
            )
            with open(sampling_json, "w") as f:
                json.dump(sampling_summary, f, indent=2)
        save_metric_name = self.save_metric
        if save_metric_name not in metric_results:
            available_metrics = ", ".join(sorted(metric_results.keys()))
            raise KeyError(
                f"Unsupported save_metric '{save_metric_name}'. Available metrics: {available_metrics}"
            )
        save_result = metric_results[save_metric_name]
        if save_metric_name == "main":
            save_metric_label = "mIoU"
        else:
            save_metric_label = f"{save_metric_name}/mIoU"
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = save_result["m_iou"]
        self.trainer.comm_info["current_metric_name"] = save_metric_label

    def after_train(self):
        if self.save_metric == "main":
            best_metric_name = "mIoU"
        else:
            best_metric_name = f"{self.save_metric}/mIoU"
        self.trainer.logger.info(
            "Best {}: {:.4f}".format(best_metric_name, self.trainer.best_metric_value)
        )
