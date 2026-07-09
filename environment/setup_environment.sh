#!/bin/bash
# ============================================================================
# One-time environment setup for the vessel segmentation pipeline.
#
# Assumes you already have (mini)conda set up and working.
#
# Usage:
#   bash setup_environment.sh
# ============================================================================

set -e  # stop immediately if any command fails, rather than continuing
        # with a broken partial install

echo "=== Step 0: Making sure conda is usable in this shell ==="
# `conda activate` requires shell functions that `conda init` installs into
# your .bashrc. If this script is run in a shell/job context where that
# hasn't happened (e.g. a fresh session, a non-interactive script, or a
# galyleo-launched job), `conda activate` fails with:
#   CondaError: Run 'conda init' before 'conda activate'
# Sourcing conda.sh directly sidesteps that dependency -- it works whether
# or not `conda init` has been run for this particular shell.
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: 'conda' command not found. Make sure conda/miniconda is on your PATH."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "Using conda at: $CONDA_BASE"

echo ""
echo "=== Step 1: Creating the 'vessel-seg' conda environment ==="
if conda env list | grep -q "vessel-seg"; then
    echo "Environment 'vessel-seg' already exists -- skipping creation."
    echo "(If you want a clean rebuild, run: conda env remove -n vessel-seg)"
else
    conda create -n vessel-seg python=3.11 -y
fi
conda activate vessel-seg

echo ""
echo "=== Step 2: Installing TensorFlow with GPU support ==="
pip install --upgrade pip -q
# Pinned to 2.20, NOT 2.21 -- TensorFlow 2.21.0 has a confirmed upstream bug
# where GPU detection fails with "Cannot dlopen some GPU libraries" even
# though every CUDA library loads fine manually. See:
# https://github.com/tensorflow/tensorflow/issues/113541
pip install 'tensorflow[and-cuda]==2.20.*' -q

echo ""
echo "=== Step 3: Installing the rest of the pipeline's dependencies ==="
pip install albumentations roifile pillow numpy -q

echo ""
echo "=== Step 4: Installing Jupyter ==="
pip install jupyterlab notebook ipywidgets matplotlib -q

# Register this environment as a Jupyter kernel, so it shows up as a
# selectable kernel ("vessel-seg") in the Jupyter interface. 
python -m ipykernel install --user --name vessel-seg --display-name "Python (vessel-seg)"

echo ""
echo "=== Step 5: Fixing a common HPC issue -- TensorFlow can't always find"
echo "    the CUDA libraries it just installed via pip. This bakes the fix"
echo "    into the environment's activation script, so it's automatic every"
echo "    time you 'conda activate vessel-seg' from now on. ==="
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << 'INNER_EOF'
# Point TensorFlow at the CUDA libraries that pip installed inside this
# conda environment (TF does not always find these automatically on HPC
# systems with their own separate system-level CUDA install).
CUDNN_PATH=$(python -c "import nvidia.cudnn as m; print(m.__path__[0])" 2>/dev/null)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH"
INNER_EOF
# Re-activate so the fix takes effect immediately for this session too
conda deactivate
conda activate vessel-seg

echo ""
echo "=== Step 6: Verifying TensorFlow sees the GPU ==="
echo "(This step needs to run somewhere a GPU is actually allocated -- if"
echo " you're running this setup script on the login node or a CPU-only"
echo " node, this will correctly report 0 GPUs; that does NOT mean the"
echo " install failed. Re-check this from your GPU job before training.)"
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs detected: {len(gpus)}')
for g in gpus:
    print(' -', g)
if len(gpus) == 0:
    print('No GPU visible from this session. If you are currently on a')
    print('GPU-allocated node and still see this, common causes are:')
    print(r'  -- check: echo \$LD_LIBRARY_PATH   should include a path')
    print('     containing nvidia/cudnn')
"

echo ""
echo "=== Setup complete ==="
echo "From now on, every new session just needs:"
echo "  conda activate vessel-seg"
echo "(no need to re-run this script -- it only needs to run once)"
