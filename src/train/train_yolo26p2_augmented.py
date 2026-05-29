import torch
from ultralytics import YOLO


def main():
    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        raise RuntimeError("GPU가 인식되지 않습니다.")

    model = YOLO("yolo26-p2.yaml")

    model.train(
        data="configs/data_augmented.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        patience=25,

        mosaic=0.5,
        close_mosaic=10,
        fliplr=0.5,
        flipud=0.0,

        project="runs/detect",
        name="train_yolo26p2_augmented_100ep"
    )


if __name__ == "__main__":
    main()