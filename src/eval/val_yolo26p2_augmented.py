from ultralytics import YOLO


def run_val(model_path, data_yaml, name, imgsz=640, conf=0.1):
    model = YOLO(model_path)

    model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=0.7,
        device=0,
        workers=0,
        plots=True,
        project="runs/eval",
        name=name
    )


def main():
    run_val(
        model_path="runs/train/train_yolo26p2_augmented_100ep/weights/best.pt",
        data_yaml="configs/data_augmented.yaml",
        name="eval_yolo26p2_augmented_100ep_conf01",
        imgsz=640,
        conf=0.1
    )

    run_val(
        model_path="runs/train/train_yolo26p2_augmented_100ep/weights/best.pt",
        data_yaml="configs/data_augmented.yaml",
        name="eval_yolo26p2_augmented_100ep_conf005",
        imgsz=640,
        conf=0.05
    )

    run_val(
        model_path="runs/train/train_yolo26p2_augmented_120ep/weights/best.pt",
        data_yaml="configs/data_augmented.yaml",
        name="eval_yolo26p2_augmented_120ep_conf01",
        imgsz=960,
        conf=0.1
    )

    run_val(
        model_path="runs/train/train_yolo26p2_augmented_120ep/weights/best.pt",
        data_yaml="configs/data_augmented.yaml",
        name="eval_yolo26p2_augmented_120ep_conf005",
        imgsz=960,
        conf=0.05
    )


if __name__ == "__main__":
    main()