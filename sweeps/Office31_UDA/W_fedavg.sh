# Goal:
#   Train CNN on OfficeHome65A dataset by Single-Task-Learning (STL).
#   All settings follow SHOT using central training.

# TODO use MnistCnnSplit (w/o adv/task predictor) model. but be careful when load.
kwargs="
dataset=Office31
dataset.name=Office31W
logger.wandb.project=dmtl-Office31A_STL
name=stl
model=OfficeCnnSplitAdv
user=generic
user.batch_size=64
user.local_epochs=-1
num_glob_iters=50
n_rep=5
hydra.launcher.n_jobs=3
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
+model.bottleneck_type=bn
"
# +model.bottleneck_type=drop

# NOTE: load_model.hash_name has to be updated after training.
load_kwargs="
load_model.do=true
load_model.load=[server]
load_model.hash_name=Office31W/FedAvg/OfficeCnnSplit/1e6351a86a07eafba767541fb79aab07bb6eb01f/g_0
"
# dropout bottleneck: 0ff680f5368dcbe84087c270afefc79940dbd08b
# bn battleneck: 1e6351a86a07eafba767541fb79aab07bb6eb01f
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
## print the config
#python -m dmtl.mainx $kwargs logger.wandb.group='${server.name}-u${dataset.n_user}-lr${user.optimizer.learning_rate}' $data_kwargs $kwargs_multi_select $data_kwargs_multi_select --cfg job
##echo
### run sweep
### python -m dmtl.mainx $kwargs $kwargs_multi logger.wandb.group='${server.name}-u${dataset.n_user}-lr${user.optimizer.learning_rate}' $data_kwargs $data_kwargs_multi -m
### Run only with the selected hparam
### TODO add `$load_kwargs` to fine-tune the model.
#python -m dmtl.mainx $kwargs $data_kwargs_multi_select logger.wandb.group='${server.name}-u${dataset.n_user}-lr${user.optimizer.learning_rate}' $data_kwargs $kwargs_multi_select
#
#echo =====
#echo Check generated files
#echo =====
#echo
#python -m dmtl.mainx $kwargs $kwargs_multi_select $data_kwargs $data_kwargs_multi_select logger.wandb.group='${server.name}-tr${dataset.tasks.1.total_tr_size}-u${dataset.tasks.1.n_user}-lr${user.optimizer.learning_rate}' action=check_files
#
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


echo =====
echo Eval on Office31A
echo =====
# generate data
#echo python -m dmtl.data.extend -cn MnistM $data_kwargs $data_kwargs_multi_select dataset.name=MnistMSTL

# TODO the train set may use random crop for augmentation.

python -m dmtl.mainx $kwargs $kwargs_multi_select logger.wandb.group='${server.name}-u${dataset.n_user}-lr${user.optimizer.learning_rate}' $data_kwargs   $load_kwargs action=eval dataset.name=Office31A logger.loggers='[]'
# $data_kwargs_multi_select


echo =====
echo Eval on Office31D
echo =====
# generate data
#echo python -m dmtl.data.extend -cn MnistM $data_kwargs $data_kwargs_multi_select dataset.name=MnistMSTL

# TODO the train set may use random crop for augmentation.

python -m dmtl.mainx $kwargs $kwargs_multi_select logger.wandb.group='${server.name}-u${dataset.n_user}-lr${user.optimizer.learning_rate}' $data_kwargs   $load_kwargs action=eval dataset.name=Office31D logger.loggers='[]'
# $data_kwargs_multi_select