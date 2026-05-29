# Waldo Detection using YOLO26-P2

This project detects Waldo in complex images using YOLO26-P2 and sliced inference.

## Methods

- YOLO26-P2 object detection
- Raw dataset training
- Processed tile-based dataset training
- sliced inference
- Custom evaluation using AP50, Precision, Recall, F1-score, and Mean IoU

## Best Result

| Method | AP50 | Precision | Recall | F1 | Mean IoU |
|---|---:|---:|---:|---:|---:|
| YOLO26-P2 + Processed | 0.7119 | 0.6591 | 0.7954 | 0.7209 | 0.7724 |

## Conclusion

Tile-based preprocessing and sliced inference improved small object detection performance for Waldo detection.
