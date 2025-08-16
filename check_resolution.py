import cv2
import numpy as np
import os

gt_dir = 'datasets/trackNet/gts'
sample_file = None

for root, dirs, files in os.walk(gt_dir):
    if files:
        sample_file = os.path.join(root, files[0])
        break

if sample_file:
    img = cv2.imread(sample_file)
    print('=== ORIGINAL GROUND TRUTH ANALYSIS ===')
    print('1. File path:', sample_file)
    print('2. Original shape:', img.shape)
    print('3. Original max value:', np.max(img))
    print('4. Original non-zero pixels:', np.count_nonzero(img))
    print('5. Original unique values:', np.unique(img)[:10])
    print('6. Is this 1280x720?', img.shape[1] == 1280 and img.shape[0] == 720)
else:
    print('No ground truth files found')
