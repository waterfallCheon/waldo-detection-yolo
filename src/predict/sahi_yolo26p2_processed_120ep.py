from sahi import AutoDetectionModel
from sahi.predict import predict


def main():
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="runs/detect/train_yolo26p2_processed_120ep/weights/best.pt",
        confidence_threshold=0.05,
        device="cuda:0",
        image_size=960,
    )

    predict(
        detection_model=detection_model,
        source="dataset/processed/test/images",
        slice_height=384,
        slice_width=384,
        overlap_height_ratio=0.35,
        overlap_width_ratio=0.35,
        postprocess_match_threshold=0.5,
        project="runs/sahi",
        name="sahi_yolo26p2_processed_120ep_conf005_slice384_overlap035",
        export_pickle=False,
        export_crop=False,
    )


if __name__ == "__main__":
    main()