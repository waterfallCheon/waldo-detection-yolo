from ultralytics import YOLO


def main():
    model = YOLO("runs/train/train_yolo26p2_augmented_120ep/weights/best.pt")

    model.predict(
        source="dataset/data_augmented/test/images",
        imgsz=640,
        conf=0.1,
        save=True,
        project="runs/predict",
        name="pred_yolo26p2_augmented_100ep"
    )


if __name__ == "__main__":
    main()