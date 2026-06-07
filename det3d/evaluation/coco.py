from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.data import box_utils


def compute_coco_metrics(
    detector,
    val_outputs_all,
    val_targets_all,
    class_names,
    iou_list=None,
    max_detection=None,
):
    if iou_list is None:
        iou_list = [0.1]
    if max_detection is None:
        max_detection = [100]
    coco_metric = COCOMetric(classes=class_names, iou_list=iou_list, max_detection=max_detection)
    results_metric = matching_batch(
        iou_fn=box_utils.box_iou,
        iou_thresholds=coco_metric.iou_thresholds,
        pred_boxes=[
            val_data_i[detector.target_box_key].cpu().detach().numpy()
            for val_data_i in val_outputs_all
        ],
        pred_classes=[
            val_data_i[detector.target_label_key].cpu().detach().numpy()
            for val_data_i in val_outputs_all
        ],
        pred_scores=[
            val_data_i[detector.pred_score_key].cpu().detach().numpy()
            for val_data_i in val_outputs_all
        ],
        gt_boxes=[
            val_data_i[detector.target_box_key].cpu().detach().numpy()
            for val_data_i in val_targets_all
        ],
        gt_classes=[
            val_data_i[detector.target_label_key].cpu().detach().numpy()
            for val_data_i in val_targets_all
        ],
    )
    return coco_metric(results_metric)[0]
