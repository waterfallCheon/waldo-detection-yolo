from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/train_yolo26p2_processed_120ep/weights/best.pt")

    model.predict(
        source="dataset/processed/test/images",
        imgsz=960,
        conf=0.05,
        save=True,
        name="pred_yolo26p2_processed_120ep_conf005"
    )


if __name__ == "__main__":
    main()