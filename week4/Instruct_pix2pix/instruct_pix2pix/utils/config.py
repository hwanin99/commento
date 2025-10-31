#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from easydict import EasyDict


def parse_args():
    args = EasyDict()
    args.pretrained_model_name_or_path = "timbrooks/instruct-pix2pix"
    args.revision = None
    args.train_data_dir = "instruct_pix2pix/dataset/metadata.jsonl"
    args.original_image_column ="original_image"
    args.edited_image_column = "edited_image"
    args.edit_prompt_column = "edit_prompt"
    args.val_image_url = None
    args.validation_prompt = None
    args.num_validation_images = 4
    args.validation_epochs = 1
    args.max_train_samples = None
    args.output_dir = "instruct_pix2pix_model"
    args.logging_dir = "logs"
    args.cache_dir = "cache"
    args.seed = 42
    args.resolution = 256
    args.center_crop = False
    args.random_flip = True
    args.train_batch_size = 16
    args.num_train_epochs = 100
    args.max_train_steps = None
    args.gradient_accumulation_steps = 1
    args.gradient_checkpointing = True
    args.learning_rate = 1e-4
    args.scale_lr = False
    args.lr_scheduler = "constant"
    args.lr_warmup_steps = 500
    args.conditioning_dropout_prob = None
    args.use_8bit_adam = True
    args.allow_tf32 = True
    args.use_ema = True
    args.non_ema_revision = None
    args.dataloader_num_workers = 0
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_weight_decay = 1e-2
    args.adam_epsilon = 1e-08
    args.max_grad_norm = 1.0
    args.push_to_hub = True
    args.hub_token = "Change to your HuggingFace API Key"
    args.hub_model_id = None
    args.mixed_precision = None
    args.report_to = "tensorboard"
    args.local_rank = 1
    args.checkpointing_steps = 500
    args.checkpoints_total_limit = None
    args.resume_from_checkpoint = None
    args.enable_xformers_memory_efficient_attention = True

    # args = EasyDict()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

