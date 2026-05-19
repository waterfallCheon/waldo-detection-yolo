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
        data="data_raw.yaml",
        epochs=120,
        imgsz=960,
        batch=2,
        device=0,
        workers=0,

        lr0=0.001,
        lrf=0.01,
        optimizer="AdamW",
        patience=30,

        mosaic=0.5,
        close_mosaic=15,
        fliplr=0.5,
        flipud=0.0,

        name="train_yolo26p2_raw_120ep"
    )


if __name__ == "__main__":
    main()