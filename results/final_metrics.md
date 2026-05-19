# Final Evaluation Metrics

## Best Model

| Method | AP50 | Precision | Recall | F1 | Mean IoU |
|---|---:|---:|---:|---:|---:|
| YOLO26-P2 + Processed + SAHI | 0.7119 | 0.6591 | 0.7954 | 0.7209 | 0.7724 |

## YOLO Baseline Results

| Dataset | Epoch | Conf | TP | FP | FN | Precision | Recall | F1 | Mean IoU | AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Processed | 100 | 0.1 | 735 | 702 | 218 | 0.5115 | 0.7712 | 0.6151 | 0.7738 | 0.6667 |
| Processed | 100 | 0.05 | 781 | 1245 | 172 | 0.3855 | 0.8195 | 0.5243 | 0.7671 | 0.6879 |
| Processed | 120 | 0.1 | 737 | 425 | 216 | 0.6343 | 0.7733 | 0.6969 | 0.7825 | 0.6764 |
| Processed | 120 | 0.05 | 791 | 723 | 162 | 0.5225 | 0.8300 | 0.6413 | 0.7792 | 0.7097 |
| Raw | 100 | 0.1 | 6 | 10 | 12 | 0.3750 | 0.3333 | 0.3529 | 0.6911 | 0.2460 |
| Raw | 100 | 0.05 | 10 | 26 | 8 | 0.2778 | 0.5556 | 0.3704 | 0.6348 | 0.3274 |
| Raw | 120 | 0.1 | 9 | 21 | 9 | 0.3000 | 0.5000 | 0.3750 | 0.7150 | 0.2407 |
| Raw | 120 | 0.05 | 9 | 42 | 9 | 0.1765 | 0.5000 | 0.2609 | 0.7150 | 0.2407 |

## SAHI Evaluation Results

| Dataset | Method | TP | FP | FN | Precision | Recall | F1 | Mean IoU | AP50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | YOLO26-P2 + SAHI | 7 | 15 | 11 | 0.3182 | 0.3889 | 0.3500 | 0.6966 | 0.2532 |
| Processed | YOLO26-P2 + SAHI | 758 | 392 | 195 | 0.6591 | 0.7954 | 0.7209 | 0.7724 | 0.7119 |

## Conclusion

The best performance was achieved by the YOLO26-P2 model trained on the processed dataset and evaluated with SAHI-based sliced inference. This configuration achieved an AP50 of 0.7119, F1-score of 0.7209, recall of 0.7954, and mean IoU of 0.7724.

Compared with raw-image training, the processed dataset substantially improved detection performance. The results indicate that tile-based preprocessing and SAHI-based sliced inference are effective for detecting small Waldo objects in cluttered scenes.