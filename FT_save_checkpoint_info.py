import torch

# Define the tasks dynamically
TASK_NAMES = ['Task1', 'Task2', 'Task3']

checkpoints_names = ['./CheckPoints/Name.pth']

all_tot_loss = []
all_val_loss = []
all_norms = []

tot_loss_file = open("./Total_Loss.csv", "w")
val_loss_file = open("./Val_Loss.csv", "w")
norm_file = open("./Norms.csv", "w")

task_eval_files = {}
task_metrics = {}

for task in TASK_NAMES:
    task_eval_files[task] = open(f"./{task}_Eval.csv", "w")
    task_metrics[task] = {
        'acc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

for checkpoints_name in checkpoints_names:
    checkpoint = torch.load(checkpoints_name)

    model_name = checkpoint['train_config'].model_name if checkpoint['train_config'].model_name is not None else 'None'
    optimizer_name = checkpoint['train_config'].optimizer_name if checkpoint['train_config'].optimizer_name is not None else 'None'
    max_lr = checkpoint['train_config'].max_lr if checkpoint['train_config'].max_lr is not None else 'None'
    min_lr = checkpoint['train_config'].min_lr if checkpoint['train_config'].min_lr is not None else 'None'
    warmup_iters = checkpoint['train_config'].warmup_iters if checkpoint['train_config'].warmup_iters is not None else 'None'
    tokens_batch_size = checkpoint['train_config'].tokens_batch_size
    weight_decay = checkpoint['train_config'].weight_decay

    print(model_name)
    print(optimizer_name)
    print(f"max_lr: {max_lr}")
    print(f"min_lr: {min_lr}")
    print(f"warmup_iters: {warmup_iters}")
    print(f"Tokens Batch Size: {tokens_batch_size}")
    print(f"Weight Decay: {weight_decay}")
    print('================================================================')

    identifier = f"{optimizer_name}-{max_lr}-{min_lr}-{tokens_batch_size}-{weight_decay}-{checkpoints_name[14:26]}"
    
    all_tot_loss.append([identifier] + checkpoint['total_loss_list'])
    all_val_loss.append([identifier] + checkpoint['val_losses'])
    all_norms.append([identifier] + checkpoint['total_norm_list'])

    for task in TASK_NAMES:
        if task in checkpoint:
            task_eval = checkpoint[task]
            task_metrics[task]['acc'].append(identifier)
            task_metrics[task]['precision'].append(identifier)
            task_metrics[task]['recall'].append(identifier)
            task_metrics[task]['f1'].append(identifier)

            for metric in task_eval:
                task_metrics[task]['acc'].append(metric[0])
                task_metrics[task]['precision'].append(metric[1])
                task_metrics[task]['recall'].append(metric[2])
                task_metrics[task]['f1'].append(metric[3])

    print('Done')
    del checkpoint

all_tot_loss = list(zip(*all_tot_loss))
all_val_loss = list(zip(*all_val_loss))
all_norms = list(zip(*all_norms))

for sub_list in all_tot_loss:
    for item in sub_list:
        tot_loss_file.write(str(item) + ',')
        
    tot_loss_file.write('\n')
tot_loss_file.close()

for sub_list in all_val_loss:
    for item in sub_list:
        val_loss_file.write(str(item) + ',')
        
    val_loss_file.write('\n')
val_loss_file.close()

for sub_list in all_norms:
    for item in sub_list:
        norm_file.write(str(item) + ',')

    norm_file.write('\n')
norm_file.close()

for task in TASK_NAMES:
    for acc, precision, recall, f1 in zip(task_metrics[task]['acc'], task_metrics[task]['precision'], task_metrics[task]['recall'], task_metrics[task]['f1']):
        task_eval_files[task].write(f"{acc},{precision},{recall},{f1},\n")
    task_eval_files[task].close()
