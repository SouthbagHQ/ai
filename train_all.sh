#!/bin/bash
set -e

# Train K1
cd /home/josh/Projects/testing_ground/agy/ai/k1
source venv/bin/activate
python train.py

# Train C1
cd /home/josh/Projects/testing_ground/agy/ai/c1
python train.py

# Export both to Numpy
cd /home/josh/Projects/testing_ground/agy/ai
python export_np.py

echo "Training and export complete!"
