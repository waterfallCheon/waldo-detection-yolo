# Waldo Detection using YOLO26-P2

This project detects Waldo in complex images using YOLO26-P2 and SAHI-based sliced inference.

## Methods

- YOLO26-P2 object detection
- Raw dataset training
- Processed tile-based dataset training
- SAHI sliced inference
- Custom evaluation using AP50, Precision, Recall, F1-score, and Mean IoU

## Best Result

| Method | AP50 | Precision | Recall | F1 | Mean IoU |
|---|---:|---:|---:|---:|---:|
| YOLO26-P2 + Processed + SAHI | 0.7119 | 0.6591 | 0.7954 | 0.7209 | 0.7724 |

## Conclusion

Tile-based preprocessing and SAHI-based sliced inference improved small object detection performance for Waldo detection.
