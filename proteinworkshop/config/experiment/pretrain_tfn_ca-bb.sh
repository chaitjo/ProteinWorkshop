HYDRA_FULL_ERROR=1 python proteinworkshop/train.py encoder=tfn task=structure_denoising dataset=afdb_rep_v4 features=ca_bb trainer=gpu_accum scheduler=linear_warmup_cosine_decay callbacks.model_checkpoint.every_n_train_steps=1000 logger=wandb