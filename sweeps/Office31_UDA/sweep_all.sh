# NOTE only run this under `src/`

# New sweeps: https://wandb.ai/jyhong/dmtl-Office31_X2X_UDA/sweeps

# Old sweeps: https://wandb.ai/jyhong/dmtl-Office31_X2X_FUDA_fuda/sweeps
# Sweep for Office31 1source vs 3 target (niid with target n_class=15): https://wandb.ai/jyhong/dmtl-Office31_X2X_FUDA_fuda/sweeps
# Sweep for Office31 1source vs 3 target (niid with target n_class=31):
# shotBNFX:

echo current path: $(pwd)
pwd=sweeps/Office31_UDA/*_sweep.yaml

for f in $pwd
do
  echo $f
done

echo "========================================"
echo "TODO: Copy the print paths to all_files."
echo "========================================"

# TODO update the paths
all_files=(
# A2X
sweeps/Office31_UDA/A2X_dann_nuser_sweep.yaml
sweeps/Office31_UDA/A2X_cdan_nuser_sweep.yaml
sweeps/Office31_UDA/A2X_shot_nuser_sweep.yaml
## D2X
#sweeps/Office31_UDA/D2X_dann_nuser_sweep.yaml
#sweeps/Office31_UDA/D2X_cdan_nuser_sweep.yaml
#sweeps/Office31_UDA/D2X_shot_nuser_sweep.yaml
## W2X
#sweeps/Office31_UDA/W2X_dann_nuser_sweep.yaml
#sweeps/Office31_UDA/W2X_cdan_nuser_sweep.yaml
#sweeps/Office31_UDA/W2X_shot_nuser_sweep.yaml
)

#wandb sweep $(pwd)
for f in ${all_files[@]}
do
  echo "==== sweep $f ==="
  wandb sweep $f
done