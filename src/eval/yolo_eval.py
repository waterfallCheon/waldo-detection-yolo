from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


IOU_THRES = 0.5


RUNS = {
    "processed_100_conf01": {
        "image_dir": "dataset/processed/test/images",
        "label_dir": "dataset/processed/test/labels",
        "model_path": "runs/detect/train_yolo26p2_processed_100ep/weights/best.pt",
        "conf": 0.1,
        "imgsz": 640,
    },
    "processed_100_conf005": {
        "image_dir": "dataset/processed/test/images",
        "label_dir": "dataset/processed/test/labels",
        "model_path": "runs/detect/train_yolo26p2_processed_100ep/weights/best.pt",
        "conf": 0.05,
        "imgsz": 640,
    },
    "processed_120_conf01": {
        "image_dir": "dataset/processed/test/images",
        "label_dir": "dataset/processed/test/labels",
        "model_path": "runs/detect/train_yolo26p2_processed_120ep/weights/best.pt",
        "conf": 0.1,
        "imgsz": 960,
    },
    "processed_120_conf005": {
        "image_dir": "dataset/processed/test/images",
        "label_dir": "dataset/processed/test/labels",
        "model_path": "runs/detect/train_yolo26p2_processed_120ep/weights/best.pt",
        "conf": 0.05,
        "imgsz": 960,
    },
    "raw_100_conf01": {
        "image_dir": "dataset/raw/val/images",
        "label_dir": "dataset/raw/val/labels",
        "model_path": "runs/detect/train_yolo26p2_raw_100ep/weights/best.pt",
        "conf": 0.1,
        "imgsz": 640,
    },
    "raw_100_conf005": {
        "image_dir": "dataset/raw/val/images",
        "label_dir": "dataset/raw/val/labels",
        "model_path": "runs/detect/train_yolo26p2_raw_100ep/weights/best.pt",
        "conf": 0.05,
        "imgsz": 640,
    },
    "raw_120_conf01": {
        "image_dir": "dataset/raw/val/images",
        "label_dir": "dataset/raw/val/labels",
        "model_path": "runs/detect/train_yolo26p2_raw_120ep/weights/best.pt",
        "conf": 0.1,
        "imgsz": 960,
    },
    "raw_120_conf005": {
        "image_dir": "dataset/raw/val/images",
        "label_dir": "dataset/raw/val/labels",
        "model_path": "runs/detect/train_yolo26p2_raw_120ep/weights/best.pt",
        "conf": 0.05,
        "imgsz": 960,
    },
}


def yolo_to_xyxy(label_line, img_w, img_h):
    cls, x, y, w, h = map(float, label_line.strip().split())
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def load_gt_boxes(image_path, label_dir):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] 이미지 읽기 실패: {image_path}")
        return []

    img_h, img_w = img.shape[:2]
    label_path = label_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        return []

    text = label_path.read_text().strip()
    if not text:
        return []

    return [
        yolo_to_xyxy(line, img_w, img_h)
        for line in text.splitlines()
        if line.strip()
    ]


def compute_ap(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap


def evaluate_predictions_ap50(all_predictions, gt_by_image):
    all_predictions = sorted(all_predictions, key=lambda x: x["score"], reverse=True)
    total_gt = sum(len(v) for v in gt_by_image.values())

    if total_gt == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    matched = {image_name: set() for image_name in gt_by_image.keys()}
    tp_list = []
    fp_list = []
    matched_ious = []

    for pred in all_predictions:
        image_name = pred["image"]
        pred_box = pred["box"]
        gt_boxes = gt_by_image.get(image_name, [])

        best_iou = 0
        best_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched[image_name]:
                continue

            iou = box_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= IOU_THRES and best_gt_idx is not None:
            tp_list.append(1)
            fp_list.append(0)
            matched[image_name].add(best_gt_idx)
            matched_ious.append(best_iou)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    recall_curve = tp_cum / total_gt
    precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    ap50 = compute_ap(recall_curve, precision_curve)

    tp = int(tp_cum[-1]) if len(tp_cum) else 0
    fp = int(fp_cum[-1]) if len(fp_cum) else 0
    fn = total_gt - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0

    return tp, fp, fn, precision, recall, f1, mean_iou, ap50


def evaluate_yolo(config_name):
    cfg = RUNS[config_name]

    image_dir = Path(cfg["image_dir"])
    label_dir = Path(cfg["label_dir"])
    model_path = Path(cfg["model_path"])
    conf = cfg["conf"]
    imgsz = cfg["imgsz"]

    if not image_dir.exists():
        raise FileNotFoundError(f"IMAGE_DIR 없음: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"LABEL_DIR 없음: {label_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH 없음: {model_path}")

    model = YOLO(str(model_path))

    image_paths = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )

    gt_by_image = {}
    all_predictions = []

    print("YOLO Custom Evaluation Start")
    print("----------------------------")
    print("Config:", config_name)
    print("Image dir:", image_dir)
    print("Label dir:", label_dir)
    print("Model:", model_path)
    print("Conf:", conf)
    print("Image size:", imgsz)
    print("Images:", len(image_paths))
    print()

    for image_path in image_paths:
        gt_boxes = load_gt_boxes(image_path, label_dir)
        gt_by_image[image_path.name] = gt_boxes

        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            device=0,
            verbose=False
        )

        pred_count = 0
        max_score = None

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            for box, score in zip(boxes, scores):
                all_predictions.append(
                    {
                        "image": image_path.name,
                        "box": box.tolist(),
                        "score": float(score),
                    }
                )
                pred_count += 1
                max_score = float(score) if max_score is None else max(max_score, float(score))

        score_text = f" max_score={max_score:.4f}" if max_score is not None else ""
        print(f"{image_path.name}: GT={len(gt_boxes)} PRED={pred_count}{score_text}")

    tp, fp, fn, precision, recall, f1, mean_iou, ap50 = evaluate_predictions_ap50(
        all_predictions,
        gt_by_image
    )

    print()
    print("YOLO Custom Evaluation Result")
    print("-----------------------------")
    print(f"Config: {config_name}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"AP50: {ap50:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=list(RUNS.keys()) + ["all"],
        required=True
    )
    args = parser.parse_args()

    if args.config == "all":
        for config_name in RUNS.keys():
            evaluate_yolo(config_name)
            print("\n" + "=" * 60 + "\n")
    else:
        evaluate_yolo(args.config)


if __name__ == "__main__":
    main()