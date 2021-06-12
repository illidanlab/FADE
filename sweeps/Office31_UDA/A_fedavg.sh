# Goal:
#   Train CNN on OfficeHome65A dataset by Single-Task-Learning (STL).
#   All settings follow SHOT using central training.

# TODO use MnistCnnSplit (w/o adv/task predictor) model. but be careful when load.
kwargs="
dataset=Office31
dataset.name=Office31A
logger.wandb.project=dmtl-Office31A_STL
name=stl
model=OfficeCnnSplitAdv
user=generic
user.batch_size=64
user.local_epochs=-1
num_glob_iters=50
n_rep=3
hydra.launcher.n_jobs=1
tele.enable=False
server.num_users=-1
+user.total_local_epochs=38
+eval_freq=10
"
data_kwargs=""
data_kwargs_multi="
dataset.n_user=1
"
#data_kwargs_multi_select="
#dataset.n_user=5
#dataset.n_class=15
#+dataset.class_stride=10
#"
data_kwargs_multi_select="
dataset.n_user=1
"
kwargs_multi="
user.optimizer.learning_rate=0.01,0.001
user.optimizer.name=rmsprop
i_rep=0,1,2,3,4
server=FedAvg
"
# adam works better than rmsprop
kwargs_multi_select="
user.optimizer.learning_rate=0.01
user.optimizer.name=sgd_sch
user.loss=sxent
i_rep=0
server=FedAvg
server.beta=1.
model.mid_dim=256
model.backbone=resnet50
model.freeze_backbone=False
+model.bottleneck_type=dropout
model.disable_bn_stat=True
"
# +model.bottleneck_type=drop

# NOTE: load_model.hash_name has to be updated after training.
load_kwargs="
load_model.do=true
load_model.load=[server]
load_model.hash_name=Office31A/FedAvg/OfficeCnnSplit/3e6393ec7c8e457107995908726fb9e9c9b5ef56/g_0
"
# dropout bottleneck: 47095220f910831acac239dd12b241120a1a093c
# dropout: 3e6393ec7c8e457107995908726fb9e9c9b5ef56 (NEW: no idea what is the difference)
# bn battleneck: 4bae7dd1a1c5bf247a4ca4ce1bf1f2394eb1f34b
#echo =====
#echo Generating dataset
#echo =====
#echo
## python -m dmtl.data.sinusoid_mtl -cn MnistSTL --cfg job
#python -m dmtl.data.extend -cn OfficeHome65 n_user=1 name=OfficeHome65A

#echo =====
#echo Train
#echo =====
#echo
# print the config
#python -m dmtl.mainx $kwargs logger.wandb.group='${server.name}-u${dataset.n_user}-lr${user.optimizer.learning_rate}' $data_kwargs $kwargs_multi_select $data_kwargs_multi_select --cfg job
## TODO add `$load_kwargs` to fine-tune the model.
#python -m dmtl.mainx $kwargs $data_kwargs_multi_select logger.wandb.group='${server.name}-u${dataset.n_user}-lr${user.optimizer.learning_rate}' $data_kwargs $kwargs_multi_select

## # Repeat experiments
##python -m dmtl.mainx $kwargs $data_kwargs_multi_select logger.wandb.group='${server.name}-u${dataset.n_user}-lr${user.optimizer.learning_rate}' $data_kwargs $kwargs_multi_select i_rep=1,2 -m

#echo =====
#echo Check generated files
#echo =====
#echo
#python -m dmtl.mainx $kwargs $kwargs_multi_select $data_kwargs $data_kwargs_multi_select logger.wandb.group='${server.name}-tr${dataset.tasks.1.total_tr_size}-u${dataset.tasks.1.n_user}-lr${user.optimizer.learning_rate}' action=check_files

#echo
#echo ==========================================
#echo "Now copy copy the path to the root of server.pt as the hash name."
#echo ==========================================
#
## TODO after updating the hash name, uncomment below and run evaluation to check the performance.

#echo =====
#echo Eval on Office31A
#echo =====
#
#python -m dmtl.mainx $kwargs $data_kwargs_multi_select $load_kwargs $data_kwargs $kwargs_multi_select action=eval logger.loggers='[]'


all_targets=(
Office31D
Office31W
)

for target in ${all_targets[@]}
do
  echo =====
  echo Eval on ${target}
  echo =====

#  echo With loaded model
#  python -m dmtl.mainx $kwargs $kwargs_multi_select logger.wandb.group='${server.name}-u${dataset.n_user}-lr${user.optimizer.learning_rate}' $data_kwargs  $data_kwargs_multi_select $load_kwargs action=eval dataset.name=$target logger.loggers='[]'


  echo With trained models
  python -m dmtl.mainx $kwargs $kwargs_multi_select logger.wandb.group='${server.name}-u${dataset.n_user}-lr${user.optimizer.learning_rate}' $data_kwargs action=eval dataset.name=$target logger.loggers='[]' $data_kwargs_multi_select i_rep=-1 load_model.do=true load_model.load=[server] load_model.do=true load_model.load=[server] load_model.hash_name=Office31A/FedAvg/OfficeCnnSplit/3e6393ec7c8e457107995908726fb9e9c9b5ef56
  # $data_kwargs_multi_select
done