# CIFAR-100: Intra-Superclass Pairing for Hierarchical Mixup

A course project for **Machine Learning Lab** (2026). A 2.8M-parameter ResNet is trained
from scratch on CIFAR-100 to study one question:

> Can Mixup coexist with hierarchical (superclass) supervision?

## Idea

CIFAR-100 has 100 fine classes grouped into 20 superclasses of 5 fine classes each.
Standard Mixup pairs images uniformly at random, so two images share a superclass with
probability `20 x (5/100)^2 = 5%`. When a superclass auxiliary head is attached, the
superclass label of a mixed sample would have to be interpolated as well — but a
superclass is a discrete category, so `0.5 x flowers + 0.5 x fish` is not a valid label.

This project restricts the second image to the **same superclass**. The superclass label is
then exactly invariant (enforced by an assertion in the training loop), while fine-class
labels are still interpolated as usual.

The same pairing rule is implemented for both Mixup and CutMix, giving two structurally
parallel ladders.

## Experimental design

| Arm | Mixing | Pairing | Superclass head |
|---|---|---|---|
| `base` | none | — | no |
| `hier` | none | — | yes |
| `mix_std` | Mixup | uniform random | yes |
| `mix_intra` | Mixup | intra-superclass | yes |
| `cutmix_std` | CutMix | uniform random | yes |
| `cutmix_intra` | CutMix | intra-superclass | yes |

Network, schedule, augmentation, lambda distribution and loss weights are identical across
arms. The only variables are the pairing rule and, between ladders, the mixing operator.

Training: 120 epochs, batch 128, SGD + Nesterov (lr 0.1, 5-epoch warmup + cosine),
weight decay 5e-4, label smoothing 0.1, Mixup probability 0.5 with alpha 1.0,
superclass loss weight 0.3, seed 42. Six arms take about 4.5 hours on a V100-32GB.

## Results (last epoch, 10,000 test images)

| Arm | Top-1 | Top-5 | Intra-superclass error share | Cross-superclass errors |
|---|---|---|---|---|
| Random forest (HOG + color histogram) | 0.2823 | 0.5251 | 15.05% | — |
| `base` | 0.7723 | 0.9337 | 35.18% | 1476 |
| `hier` | 0.7763 | 0.9359 | 37.73% | 1393 |
| `mix_std` | 0.7891 | 0.9501 | 38.83% | 1290 |
| `mix_intra` | 0.7827 | 0.9448 | 40.31% | 1297 |
| `cutmix_std` | **0.7932** | 0.9464 | 40.62% | **1228** |
| `cutmix_intra` | 0.7863 | 0.9461 | **42.30%** | 1233 |

Within each ladder (intra-superclass minus standard pairing):

| Ladder | Top-1 | Intra-superclass error share | Cross-superclass errors |
|---|---|---|---|
| Mixup | -0.64 pp | +1.48 pp | +7 images |
| CutMix | -0.69 pp | +1.68 pp | +5 images |

The two ladders agree in sign and magnitude, which is the main evidence that the pairing
rule is independent of the mixing operator.

## What the experiment shows

- **Falsified:** the strong hypothesis that invalid superclass labels would hurt training.
  Intra-superclass pairing is consistently 0.6–0.7 pp *worse* in Top-1.
- **Replicated:** intra-superclass pairing shifts errors into the true superclass
  (+1.5–1.7 pp share) under both Mixup and CutMix.
- **Supported:** the effect does not depend on the mixing operator.
- **Not observed:** cross-superclass errors do not decrease; the extra errors are almost
  all within-superclass. The trade-off is real, and the single-seed limitation is stated
  openly.

Model size: 2,805,944 trainable parameters (421.2M MACs per 32x32 image); the superclass
head adds 5,140 parameters (0.18%).

## Repository layout

```
notebooks/   PyTorch notebook (deliverable) and PaddlePaddle variant
slides/      Self-contained 12-page HTML deck (open in any browser)
docs/        Project overview, speaker notes, deck outline (Chinese)
figures/     fig01–fig10 produced by the notebook
```

## Quick start

1. Obtain CIFAR-100 in folder-per-class PNG form:

   ```
   data/cifar100/train/<superclass>/<fine-class>/*.png
   data/cifar100/test/<superclass>/<fine-class>/*.png
   ```

   or place a `cifar100.zip` with the same structure next to the notebook; it will be
   found and extracted automatically.
2. `pip install -r requirements.txt`
3. Open `notebooks/cifar100_intra_superclass_mixup.ipynb`, select a GPU runtime, run all.

Checkpoints are written every epoch, so an interrupted session resumes automatically.
`pandas` is optional; without it the summary table falls back to plain text.

## Notes

- Single seed (42); no multi-seed error bars. Differences of ~0.6 pp are close to
  run-to-run variance and should be read as a trade-off, not a ranking.
- The data layout and the Python/Paddle backends are platform-agnostic; paths are
  discovered at runtime.
- This project does **not** claim a novel method. Hierarchy-aware Mixup has been explored
  in the literature; the contribution here is an independently observed phenomenon, a
  falsifiable mechanism, and a strictly controlled replication across two operators.

## Author

Gaoyang Wei (@Lingxin449)
