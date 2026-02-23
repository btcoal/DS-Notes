# Regularization in Neural Networks

## early stopping
* monitor validation loss; stop when it stops improving.

## gradient clipping
* clip gradients to max norm to prevent exploding gradients.

## weight decay
* L2 regularization; adds penalty to large weights.

## Label smoothing – prevents overconfidence by softening targets.

## Mixup / CutMix – blends input images (and labels) to improve generalization.

## Dropout 

* Dropout – randomly zeroes activations during training to prevent co-adaptation.
* variants – SpatialDropout, DropPath/Stochastic Depth.
* DropBlock – dropout for contiguous regions in CNNs.

## Data augmentation

* Images
    * Random crops, flips, rotations, color jitter.
* RandAugment, AutoAugment, TrivialAugment.

## Manifold mixup – mixes hidden representations instead of inputs.

## Sharpness-Aware Minimization (SAM) – optimizes for flatter minima.