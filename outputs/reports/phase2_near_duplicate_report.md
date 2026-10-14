# Phase 2 Near-Duplicate Report

Images are grouped by an 8x8 average hash (exact hash match,
same class only). This catches images that look the same but
are not byte-for-byte copies, which the MD5 check in Phase 1
misses. The split is done on these groups.

- Total images: 3,921
- Total groups: 2,810
- Groups with >1 image: 637
- Images inside a multi-image group: 1,748
- Largest group: 11 images

## Per-class effective independent count
- defect: 754 groups (from 784 raw images)
- normal: 2,056 groups (from 3,137 raw images)