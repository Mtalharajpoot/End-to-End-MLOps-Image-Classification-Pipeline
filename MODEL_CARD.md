# Model Card — VisionOps CIFAR-10 classifier

| | |
|---|---|
| **Task** | 10-class image classification (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| **Architecture** | `timm` ResNet-18 (ImageNet-pretrained), classification head replaced; any `timm` backbone via `--backbone` |
| **Training data** | CIFAR-10 train split (45k train / 5k validation), images resized to 64×64 |
| **Evaluation data** | CIFAR-10 official test split (10k images) |
| **Metric** | Top-1 accuracy (macro-averaged per class). Fill in your own run's number from `outputs/results.json`. |
| **Format** | PyTorch Lightning checkpoint + ONNX (opset 17, dynamic batch axis) |

## Intended use
Demonstration of an end-to-end ML engineering workflow (training, tracking, export, serving). Not intended for any safety-critical use.

## Limitations
- CIFAR-10 images are 32×32 and cover only 10 everyday classes; real-world photos or other classes will be misclassified with high confidence (softmax is not calibrated for out-of-distribution input).
- No fairness or robustness evaluation has been done.
- Results vary by seed, GPU and number of epochs.

## How to reproduce
`python -m visionops.train --config configs/default.yaml --seed 42`
