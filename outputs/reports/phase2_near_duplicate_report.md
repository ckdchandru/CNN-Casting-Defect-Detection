# Phase 2 Near-Duplicate Report

Perceptual-hash grouping (8x8 average hash, exact match,
same class only) - catches augmented siblings that exact-MD5
matching (Phase 1) cannot, since rotation/flip/jitter changes
every byte but leaves the physical part visually identical.

- Total images: 3,921
- Total groups: 2,810
- Groups with >1 image: 637
- Images inside a multi-image group: 1,748
- Largest group: 11 images

## Per-class effective independent count
- defect: 754 groups (from 784 raw images)
- normal: 2,056 groups (from 3,137 raw images)