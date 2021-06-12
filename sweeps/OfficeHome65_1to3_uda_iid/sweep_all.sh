# NOTE only run this under `src/`
# Pretrain using Office65_1to3_uda/*fedavg.sh

# Sweep for Office31 1source vs 3 target with iid users: https://wandb.ai/jyhong/dmtl-OfficeHome65_X2X_1to3/sweeps

echo current path: $(pwd)
pwd=dmtl/experiments/OfficeHome65_1to3_uda_iid/*_sweep.yaml

for f in $pwd
do
  echo $f
done

echo "========================================"
echo "TODO: Copy the print paths to all_files."
echo "========================================"

## TODO update the paths
all_files=(
dmtl/experiments/OfficeHome65_1to3_uda_iid/R2X_dann_nuser_sweep.yaml
dmtl/experiments/OfficeHome65_1to3_uda_iid/R2X_cdan_nuser_sweep.yaml
dmtl/experiments/OfficeHome65_1to3_uda_iid/R2X_shot_nuser_sweep.yaml
)

#wandb sweep $(pwd)
for f in ${all_files[@]}
do
  echo "==== sweep $f ==="
  wandb sweep $f
done