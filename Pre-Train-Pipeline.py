import torch
from TrainConfig import *
from Trainer import *

ADAM_W = 'AdamW'

configs = []

config_nos = 1

for i in range(config_nos):
    configs.append(TrainConfig(
                                # Model:
                                tokens_batch_size = 512 * 8 * 1,
                                batch_size = 8,
                                dec_context_size = 512,
                                batch_overlap = 0,
                                betas = (0.9, 0.95),
                                vocab_size = 12000,
                                d_model = 786,
                                num_heads = 12,
                                num_decoder_blocks = 12,
                                pos_enc_dropout = 0,
                                drop_prob = 0,
                                weight_decay = None,
                                d_feedfwd = 512 * 4,
                                mask_attention = True,
                                pre_norm = True,

                                # Data Loader and Checkpointing:
                                x_data_loader_dtype = torch.int32,
                                y_data_loader_dtype = torch.int64,
                                load_check_point = False,
                                checkpoint_path = './CheckPoints/',
                                checkpoint_name = '',
                                checkpoint_save_iter = None,
                                num_iters = None,
                                eval_val_set = True,
                                val_eval_iters = None,
                                val_eval_interval = None,
                                
                                # Optimization:
                                optimizer_name = None,
                                max_lr = None,
                                min_lr = None,
                                model_name = 'Model-Name',
                                warmup_iters = None,
                                clip_grad_norm = 1.0,

                                # Training Files:
                                replacements = {},
                                file_name = "All Files",
                                file_path = "./Training_Docs/",
                                vocab_path = "./Tokenizer/",
                                load_merge_info_name = '',
                                load_vocab_name = '',
                                data_path = './Data/',
                                train_shard_names = ['shard1', 'shard2'],
                                val_name = 'val_name',
                                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            ))

config_nos = config_nos

pipe_indx = 0
cfg_no = -1

# ========================================================================================================================

# MAIN MODEL
# VOCAB SIZE = 12000

cfg_no += 1
configs[cfg_no].optimizer_name = ADAM_W
configs[cfg_no].max_lr = 2e-4
configs[cfg_no].min_lr = 2e-5
configs[cfg_no].vocab_size = 12000
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-' + str(configs[cfg_no].vocab_size) + '-CFG-' + str(cfg_no) + '-Complete-Sanskrit'
configs[cfg_no].warmup_iters = 5000
configs[cfg_no].weight_decay = 1e-2
configs[cfg_no].tokens_batch_size = 512 * 8 * 1
configs[cfg_no].batch_overlap = 0
configs[cfg_no].batch_size = 8
assert configs[cfg_no].tokens_batch_size % (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
configs[cfg_no].gradient_accum_iters = configs[cfg_no].tokens_batch_size // (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size)
configs[cfg_no].checkpoint_save_iter = 50000
configs[cfg_no].num_iters = 300005
configs[cfg_no].val_eval_iters = 100
configs[cfg_no].val_eval_interval = 1000

# ========================================================================================================================

# Batch Size Experiment
# Batch Size = 128

# VOCAB SIZE = 12000
# Batches in 1 Epoch = 780 (Simulated Using Gradient Accumulation)

cfg_no += 1
configs[cfg_no].optimizer_name = ADAM_W
configs[cfg_no].max_lr = 9e-4
configs[cfg_no].min_lr = 2e-5
configs[cfg_no].vocab_size = 12000
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-' + str(configs[cfg_no].vocab_size) + '-CFG-' + str(cfg_no) + '-Complete-Sanskrit'
configs[cfg_no].warmup_iters = 300
configs[cfg_no].weight_decay = 1e-2
configs[cfg_no].tokens_batch_size = 512 * 64 * 2
configs[cfg_no].batch_overlap = 0
configs[cfg_no].batch_size = 64
assert configs[cfg_no].tokens_batch_size % (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
configs[cfg_no].gradient_accum_iters = configs[cfg_no].tokens_batch_size // (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size)
configs[cfg_no].checkpoint_save_iter = 1250
configs[cfg_no].num_iters = 18755
configs[cfg_no].val_eval_iters = 25
configs[cfg_no].val_eval_interval = 125

# ========================================================================================================================

# Batch Size Experiment
# Batch Size = 64

# VOCAB SIZE = 12000
# Batches in 1 Epoch = 1560

cfg_no += 1
configs[cfg_no].optimizer_name = ADAM_W
configs[cfg_no].max_lr = 6e-4
configs[cfg_no].min_lr = 2e-5
configs[cfg_no].vocab_size = 12000
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-' + str(configs[cfg_no].vocab_size) + '-CFG-' + str(cfg_no) + '-Complete-Sanskrit'
configs[cfg_no].warmup_iters = 625
configs[cfg_no].weight_decay = 1e-2
configs[cfg_no].tokens_batch_size = 512 * 64 * 1
configs[cfg_no].batch_overlap = 0
configs[cfg_no].batch_size = 64
assert configs[cfg_no].tokens_batch_size % (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
configs[cfg_no].gradient_accum_iters = configs[cfg_no].tokens_batch_size // (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size)
configs[cfg_no].checkpoint_save_iter = 2500
configs[cfg_no].num_iters = 37505
configs[cfg_no].val_eval_iters = 25
configs[cfg_no].val_eval_interval = 250

# ========================================================================================================================

# Batch Size Experiment
# Batch Size = 32

# VOCAB SIZE = 12000
# Batches in 1 Epoch = 3121

cfg_no += 1
configs[cfg_no].optimizer_name = ADAM_W
configs[cfg_no].max_lr = 3.5e-4
configs[cfg_no].min_lr = 2e-5
configs[cfg_no].vocab_size = 12000
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-' + str(configs[cfg_no].vocab_size) + '-CFG-' + str(cfg_no) + '-Complete-Sanskrit'
configs[cfg_no].warmup_iters = 1250
configs[cfg_no].weight_decay = 1e-2
configs[cfg_no].tokens_batch_size = 512 * 32 * 1
configs[cfg_no].batch_overlap = 0
configs[cfg_no].batch_size = 32
assert configs[cfg_no].tokens_batch_size % (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
configs[cfg_no].gradient_accum_iters = configs[cfg_no].tokens_batch_size // (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size)
configs[cfg_no].checkpoint_save_iter = 5000
configs[cfg_no].num_iters = 75005
configs[cfg_no].val_eval_iters = 50
configs[cfg_no].val_eval_interval = 500

# ========================================================================================================================

# Batch Size Experiment
# Batch Size = 16

# VOCAB SIZE = 12000
# Batches in 1 Epoch = 6244

cfg_no += 1
configs[cfg_no].optimizer_name = ADAM_W
configs[cfg_no].max_lr = 2e-4
configs[cfg_no].min_lr = 2e-5
configs[cfg_no].vocab_size = 12000
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-' + str(configs[cfg_no].vocab_size) + '-CFG-' + str(cfg_no) + '-Complete-Sanskrit'
configs[cfg_no].warmup_iters = 2500
configs[cfg_no].weight_decay = 1e-2
configs[cfg_no].tokens_batch_size = 512 * 16 * 1
configs[cfg_no].batch_overlap = 0
configs[cfg_no].batch_size = 16
assert configs[cfg_no].tokens_batch_size % (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
configs[cfg_no].gradient_accum_iters = configs[cfg_no].tokens_batch_size // (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size)
configs[cfg_no].checkpoint_save_iter = 10000
configs[cfg_no].num_iters = 150005
configs[cfg_no].val_eval_iters = 100
configs[cfg_no].val_eval_interval = 1000

# ========================================================================================================================

# Vocab Size Experiment
# VOCAB SIZE = 16000

cfg_no += 1
configs[cfg_no].optimizer_name = ADAM_W
configs[cfg_no].max_lr = 2e-4
configs[cfg_no].min_lr = 2e-5
configs[cfg_no].vocab_size = 16000
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-' + str(configs[cfg_no].vocab_size) + '-CFG-' + str(cfg_no) + '-Complete-Sanskrit'
configs[cfg_no].warmup_iters = 5000
configs[cfg_no].weight_decay = 1e-2
configs[cfg_no].tokens_batch_size = 512 * 8 * 1
configs[cfg_no].batch_overlap = 0
configs[cfg_no].batch_size = 8
assert configs[cfg_no].tokens_batch_size % (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
configs[cfg_no].gradient_accum_iters = configs[cfg_no].tokens_batch_size // (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size)
configs[cfg_no].checkpoint_save_iter = 10000
configs[cfg_no].num_iters = 300005
configs[cfg_no].val_eval_iters = 100
configs[cfg_no].val_eval_interval = 1000

configs[cfg_no].load_merge_info_name = '16000'
configs[cfg_no].load_vocab_name = '16000'

configs[cfg_no].train_shard_names = ['16000']
configs[cfg_no].val_name = '16000'

# ========================================================================================================================

# Vocab Size Experiment
# VOCAB SIZE = 24000

cfg_no += 1
configs[cfg_no].optimizer_name = ADAM_W
configs[cfg_no].max_lr = 2e-4
configs[cfg_no].min_lr = 2e-5
configs[cfg_no].vocab_size = 24000
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-' + str(configs[cfg_no].vocab_size) + '-CFG-' + str(cfg_no) + '-Complete-Sanskrit'
configs[cfg_no].warmup_iters = 5000
configs[cfg_no].weight_decay = 1e-2
configs[cfg_no].tokens_batch_size = 512 * 8 * 1
configs[cfg_no].batch_overlap = 0
configs[cfg_no].batch_size = 8
assert configs[cfg_no].tokens_batch_size % (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
configs[cfg_no].gradient_accum_iters = configs[cfg_no].tokens_batch_size // (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size)
configs[cfg_no].checkpoint_save_iter = 10000
configs[cfg_no].num_iters = 300005
configs[cfg_no].val_eval_iters = 100
configs[cfg_no].val_eval_interval = 1000

configs[cfg_no].load_merge_info_name = '24000'
configs[cfg_no].load_vocab_name = '24000'

configs[cfg_no].train_shard_names = ['24000']
configs[cfg_no].val_name = '24000'

# ========================================================================================================================

# Vocab Size Experiment
# VOCAB SIZE = 33000

cfg_no += 1
configs[cfg_no].optimizer_name = ADAM_W
configs[cfg_no].max_lr = 2e-4
configs[cfg_no].min_lr = 2e-5
configs[cfg_no].vocab_size = 33000
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-Complete-Sanskrit'
configs[cfg_no].warmup_iters = 5000
configs[cfg_no].weight_decay = 1e-2
configs[cfg_no].tokens_batch_size = 512 * 8 * 1
configs[cfg_no].batch_overlap = 0
configs[cfg_no].batch_size = 8
assert configs[cfg_no].tokens_batch_size % (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
configs[cfg_no].gradient_accum_iters = configs[cfg_no].tokens_batch_size // (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size)
configs[cfg_no].checkpoint_save_iter = 50000
configs[cfg_no].num_iters = 300005
configs[cfg_no].val_eval_iters = 100
configs[cfg_no].val_eval_interval = 1000

configs[cfg_no].load_merge_info_name = '33000'
configs[cfg_no].load_vocab_name = '33000'

configs[cfg_no].train_shard_names = ['33000']
configs[cfg_no].val_name = '33000'

# ========================================================================================================================

# Vocab Size Experiment
# VOCAB SIZE = 43008

cfg_no += 1
configs[cfg_no].optimizer_name = ADAM_W
configs[cfg_no].max_lr = 2e-4
configs[cfg_no].min_lr = 2e-5
configs[cfg_no].vocab_size = 43008
configs[cfg_no].model_name = 'Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-Complete-Sanskrit'
configs[cfg_no].warmup_iters = 5000
configs[cfg_no].weight_decay = 1e-2
configs[cfg_no].tokens_batch_size = 512 * 8 * 1
configs[cfg_no].batch_overlap = 0
configs[cfg_no].batch_size = 8
assert configs[cfg_no].tokens_batch_size % (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
configs[cfg_no].gradient_accum_iters = configs[cfg_no].tokens_batch_size // (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size)
configs[cfg_no].checkpoint_save_iter = 50000
configs[cfg_no].num_iters = 300005
configs[cfg_no].val_eval_iters = 100
configs[cfg_no].val_eval_interval = 1000

configs[cfg_no].load_merge_info_name = '43008'
configs[cfg_no].load_vocab_name = '43008'

configs[cfg_no].train_shard_names = ['43008']
configs[cfg_no].val_name = '43008'

# ========================================================================================================================

for i in range(config_nos):
    trainer = Trainer(configs[i])
    trainer.train()
    del trainer