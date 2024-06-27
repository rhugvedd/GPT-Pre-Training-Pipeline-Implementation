import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import *
from BPETokenizer import BPETokenizer
from DataLoader import DataLoader
import datetime
import time

"""
Hyperparameters' List:
"""
"""
Optimization and Model:
"""
tokens_batch_size = 32768 # No. of tokens in one gradient accumulation iteration
batch_size = 4
dec_context_size = 1024
batch_overlap = 960 #TODO: Check This
betas = (0.92, 0.96)
assert tokens_batch_size % (batch_size * dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
vocab_size = 12288 # TODO: Make this a good no. as a power of 2.
d_model = 786
num_heads = 12
num_decoder_blocks = 10 # TODO: Make this a good no. as a power of 2.
pos_enc_dropout = 0
drop_prob = 0
weight_decay = 0.1
d_feedfwd = d_model * 4
mask_attention = True
pre_norm = True
"""
Data Loader and Checkpointing:
"""
x_data_loader_dtype = torch.int32
y_data_loader_dtype = torch.int64
load_check_point = True
model_name = 'Component-' + str(vocab_size) + '-' + str(dec_context_size) + '-' + str(num_heads) + '-' + str(num_decoder_blocks)
checkpoint_path = './CheckPoints/26-06-24 Component-CS-1024/'
checkpoint_name = ''
checkpoint_save_iter = 100
num_iters = 3000
eval_val_set = True
val_eval_iters = 100
val_eval_interval = 50
max_lr = 5e-4
min_lr = 1e-5
warmup_iters = 200
replacements = {}
File = "Components.txt"
FilePath = "./Data/" + File
VocabPath = "./Vocab/"
Load_MergeInfo_Name = 'Component_MergeInfo-12288-2024-06-26 19-16-53'
Load_Vocab_Name = 'Component_Vocab-12288-2024-06-26 19-16-53'
data_path = './Data Tensors/'
train_name = 'Components-Train-2024-06-26 19-41-58'
val_name = 'Components-Val-2024-06-26 19-41-58'
"""
List Ends
"""

"""
Calculated Parameters:
"""
# TODO: Check This
gradient_accum_iters = tokens_batch_size // (batch_size * dec_context_size)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
"""
List Ends
"""

#====================================================================================================================
@torch.no_grad()
def estimate_val_loss():
    EmbeddingLearner.eval()
    loss_values = torch.zeros(val_eval_iters)
    data_loader.shuffle('val', reset_batch_index = True)

    for loss_iter in range(val_eval_iters):
        print('-', end="")
        
        X, Y = data_loader.get_val_batch(device)
        
        preds = EmbeddingLearner(X)
        
        B, T, C = preds.shape
        preds = preds.view(B*T, C)
        Y = Y.view(B*T)
        loss = F.cross_entropy(preds, Y)

        loss_values[loss_iter] = loss.item()

    val_loss = loss_values.mean().item()

    EmbeddingLearner.train()
    
    return val_loss

#====================================================================================================================

print("Initializing")
print(f"Total Tokens Batch Size: {tokens_batch_size}")
print(f"Iterations for Gradient Accumulation: {gradient_accum_iters}")

torch.cuda.empty_cache()

EmbeddingLearner = Decoder(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    context_size=dec_context_size,
                    pos_enc_dropout=pos_enc_dropout,
                    num_decoder_blocks=num_decoder_blocks,
                    num_heads=num_heads,
                    drop_prob=drop_prob,
                    d_feedfwd=d_feedfwd,
                    pre_norm=pre_norm,
                    mask_attention=mask_attention
                )

torch.set_float32_matmul_precision('high')
m = EmbeddingLearner.to(device)

# print("Model Compiling Started.")
# EmbeddingLearner = torch.compile(EmbeddingLearner)
# print("Compiling Done.")
print(f"No. of Parameters: {sum(p.numel() for p in m.parameters()) / 1e6} M parameters\n")

#====================================================================================================================

print("Loading Data")
data_loader = DataLoader(data_path)
data_loader.load_data   (
                            batch_size = batch_size,
                            context_size = dec_context_size,
                            train_val = 'train',
                            name = train_name,
                            batch_overlap = batch_overlap,
                            x_dtype = x_data_loader_dtype,
                            y_dtype = y_data_loader_dtype
                        )

if eval_val_set:
    data_loader.load_data   (
                                batch_size = batch_size,
                                context_size = dec_context_size,
                                train_val = 'val',
                                name = val_name,
                                batch_overlap = batch_overlap,
                                x_dtype = x_data_loader_dtype,
                                y_dtype = y_data_loader_dtype
                            )
print("Data Loading Complete\n")

#====================================================================================================================

print("Configuring Optimzer and Learning Rate Scheduler")

iters = torch.arange(num_iters + 1)
cosine_lr = (max_lr * ((iters < warmup_iters) * (iters + 1) / warmup_iters)) + ((min_lr + (0.5 * (max_lr - min_lr) * (1 + torch.cos((iters - warmup_iters) * torch.pi / (num_iters - warmup_iters))))) * (iters >= warmup_iters))

decay_params = [param for param in EmbeddingLearner.parameters() if ((param.requires_grad and param.dim()) >= 2)]
no_decay_params = [param for param in EmbeddingLearner.parameters() if ((param.requires_grad and param.dim()) < 2)]

try:
    from torch.optim import FusedAdamW
    fused_available = True
except ImportError:
    fused_available = False

fused_available = fused_available and (device == torch.device('cuda'))

print(f"Number of parameter tensors with weight decay: {len(decay_params)}, totaling {sum(p.numel() for p in decay_params):,} parameters")
print(f"Number of parameter tensors without weight decay: {len(no_decay_params)}, totaling {sum(p.numel() for p in no_decay_params):,} parameters")

optimizer = torch.optim.AdamW(   
                                [
                                    {'params': decay_params, 'weight_decay': weight_decay},
                                    {'params': no_decay_params, 'weight_decay': 0.0}
                                ],
                                lr = cosine_lr[0],
                                betas = betas,
                                eps = 1e-8,
                                fused = fused_available
                            )

st_iter = 0
val_losses = []
total_loss_list = [] 
total_norm_list = []

print("Configuration Complete\n")

#====================================================================================================================

if load_check_point:
    print("\nLoading Checkpoint")
    checkpoint = torch.load(checkpoint_path + checkpoint_name + '.pth')
    EmbeddingLearner.load_state_dict(checkpoint['model_state_dict'])   
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print('\nLoaded Checkpoint: ' + checkpoint_path + checkpoint_name)
    
    st_iter = checkpoint['iter'] + 1
    # loss = checkpoint['loss']

    print(f'Starting Iter for Training: {st_iter}')

    lr = optimizer.param_groups[0]['lr']
    print("Learning rate of loaded model:", lr)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = cosine_lr[st_iter]
    
    print("Setting learning rate to:", optimizer.param_groups[0]['lr'])

#====================================================================================================================   

torch.cuda.empty_cache()
torch.set_default_dtype(torch.float32)
print("Computing Started")

for iter in range(st_iter, num_iters + 1):
    st_time = time.time()

    # optimizer.zero_grad(set_to_none=True)
    cumulative_loss = 0.0
    optimizer.zero_grad()

    for mini_iter in range(gradient_accum_iters):

        train_x, train_y = data_loader.get_train_batch(device)
        batch_no = data_loader.train_batch_index

        # TODO: Check whether we can implement this with fp16
        # with torch.autocast(device_type=device, dtype=torch.float16):
        preds = EmbeddingLearner(train_x)

        B, T, C = preds.shape
        preds = preds.view(B*T, C)
        train_y = train_y.view(B*T)

        loss = F.cross_entropy(preds, train_y)

        loss = loss / gradient_accum_iters
        cumulative_loss += loss.detach()
        loss.backward()

    # TODO: Observe this norm and it should be stable, and not climbing linearly or having spikes.
    norm = torch.nn.utils.clip_grad_norm_(EmbeddingLearner.parameters(), 1.0)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cosine_lr[iter]

    optimizer.step()
    if device == torch.device('cuda'): 
        torch.cuda.synchronize() 
    
    total_loss_list.append(cumulative_loss.item())
    total_norm_list.append(norm.item())

    if eval_val_set and ((iter % val_eval_interval == 0) or (iter == num_iters - 1)):
        print("\nEvaluating Val Loss")
        val_loss = estimate_val_loss()
        val_losses.append(val_loss)
        print(f"\nIter {iter}: Val loss: {val_loss:.4f}\n")
    
    if (iter % checkpoint_save_iter == 0) and (iter != 0):
        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':', '-')

        torch.save  ({
                        'iter': iter,
                        'num_iters': num_iters,
                        'model_state_dict': EmbeddingLearner.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_losses': val_losses,
                        'total_loss_list': total_loss_list,
                        'total_norm_list': total_norm_list,
                        'tokens_batch_size': tokens_batch_size,
                        'batch_size': batch_size,
                        'vocab_size': vocab_size,
                        'd_model': d_model,
                        'num_heads': num_heads,
                        'dec_context_size': dec_context_size,
                        'num_decoder_blocks': num_decoder_blocks,
                        'pos_enc_dropout': pos_enc_dropout,
                        'drop_prob': drop_prob,
                        'weight_decay': weight_decay,
                        'd_feedfwd': d_feedfwd,
                        'pre_norm': pre_norm,
                        'checkpoint_save_iter': checkpoint_save_iter,
                        'val_eval_iters': val_eval_iters,
                        'val_eval_interval': val_eval_interval,
                        'min_lr': min_lr,
                        'max_lr': max_lr,
                        'warmup_iters': warmup_iters,
                        'betas': betas,
                        'replacements': replacements,
                        'File': File,
                        'FilePath': FilePath,
                        'VocabPath': VocabPath,
                        'Load_MergeInfo_Name': Load_MergeInfo_Name,
                        'Load_Vocab_Name': Load_Vocab_Name,
                        'data_path': data_path,
                        'train_name': train_name,
                        'val_name': val_name,
                        'mask_attention': mask_attention
                    }, checkpoint_path + model_name + '-Iter-' + str(iter) + '-' + date_time + '.pth')
        print("Checkpoint Saved")

    time_taken = time.time() - st_time
    token_throughput = batch_size * dec_context_size * gradient_accum_iters / time_taken

    print(f"Iter: {iter:4d} | Loss: {cumulative_loss.item():.5f} | Norm: {norm:.4f} | Batch No: ({batch_no:4d}/{data_loader.train_num_batches:4d}) | Token Throughput: {token_throughput:.2f} | Time: {time_taken*1000:.2f}ms | LR: {cosine_lr[iter]:.3e}")


print("Training Complete")