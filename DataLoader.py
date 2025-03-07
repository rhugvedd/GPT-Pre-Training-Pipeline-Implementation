import torch
from BPETokenizer import *
from datetime import datetime
import pandas as pd
import gc
from collections import OrderedDict
import os

PRE_TRAIN = "Pre-Train"
FINE_TUNE = "SemAnlg"

class DataLoader:
    def __init__    (
                        self,
                        save_path: str
                    ):
        super(DataLoader, self).__init__()

        self.save_path = save_path
        self.train_batch_index = 0
        self.val_batch_index = 0

    def extractData_MultipleDocs(
                                    self,
                                    file_path: str, 
                                    train_split_percentage: float, 
                                    vocab_path: str, 
                                    merge_info_name: str, 
                                    vocab_name: str,
                                    replacements: dict,
                                    without_new_line: bool,
                                    skip_first_chunk_in_line: bool,    
                                    train_name: str, 
                                    val_name: str,
                                    remove_last_n_merge_info: bool
                                ):

        Tokenizer = BPETokenizer()
        Tokenizer.load(vocab_path, merge_info_name, vocab_name)

        # This is done to remove the special tokens from the vocab when tokenizing for pre-training.
        if remove_last_n_merge_info:
            Tokenizer.MergeInfo = self.remove_last_n_elements_from_dict(Tokenizer.MergeInfo, len(Tokenizer.special_tokens))

        print("Tokenizing Files")

        CompleteTokens = []

        for filename in os.listdir(file_path):
            sans_file_path = os.path.join(file_path, filename)
            
            Tokens = Tokenizer.Encode(sans_file_path, WithoutNewLine = without_new_line, SkipFirstChunkInLine = skip_first_chunk_in_line, Replacements = replacements)
            Tokens = [t for sub_t in Tokens for t in sub_t]
            
            CompleteTokens += Tokens
            CompleteTokens += [Tokenizer.special_tok('<sep>')]

            print(f"Tokens in {filename}: {len(Tokens)}")
        
        print(f"Total Tokens: {len(CompleteTokens)}")

        data = torch.tensor(CompleteTokens)
        split_thres = int(train_split_percentage * len(data))

        if split_thres == 0 or split_thres == len(data):
            raise ValueError("Train split percentage results in an empty train or validation set. Please adjust the split percentage.")

        train_data = data[:split_thres]
        val_data = data[split_thres:]

        print(f"Train Tokens: {len(train_data)}")
        print(f"Val Tokens: {len(val_data)}")

        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':', '-')

        torch.save(train_data, self.save_path + train_name + date_time + '.pt')
        torch.save(val_data, self.save_path + val_name + date_time + '.pt')

        print(f"Saved Succesfuly to {save_path}")

    def remove_last_n_elements_from_dict(self, d, n):
        od = OrderedDict(d)
        
        num_elements_to_keep = len(od) - n
        
        od = OrderedDict(list(od.items())[:num_elements_to_keep])
        
        return dict(od)
  
    def clean_text(self, text):
        
        text = re.sub(r'[0-9\.\|"।]', '', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'(\r\n){2,}', '\n', text)

        return text
    
    def shuffle (
                    self,
                    split: str,
                    reset_batch_index: bool
                ):
        
        if split == 'train':
            print("SHUFFLING TRAIN BATCHES")
            shuffle_indices = torch.randperm(self.x_train.size(0))
            self.x_train = self.x_train.index_select(0, shuffle_indices)
            self.y_train = self.y_train.index_select(0, shuffle_indices)

            try:
                self.mask_train = self.mask_train.index_select(0, shuffle_indices)
            except:
                print("Mask not available!")

            if reset_batch_index:
                self.train_batch_index = 0

        elif split == 'val':
            print("SHUFFLING VAL BATCHES")
            shuffle_indices = torch.randperm(self.x_val.size(0))
            self.x_val = self.x_val.index_select(0, shuffle_indices)
            self.y_val = self.y_val.index_select(0, shuffle_indices)
            
            try:
                self.mask_val = self.mask_val.index_select(0, shuffle_indices)
            except:
                print("Mask not available!")

            if reset_batch_index:
                self.val_batch_index = 0
        else:
            raise ValueError("Wrong split name!")

    def load_fine_tune_data (   
                                self,
                                batch_size: int,
                                x_train_name: str,
                                y_train_name: str,
                                mask_train_name: str = None,
                                x_val_name: str = None,
                                y_val_name: str = None,
                                mask_val_name: str = None
                            ):

        self.x_train = torch.load(self.save_path + x_train_name + '.pt')
        self.y_train = torch.load(self.save_path + y_train_name + '.pt')
        
        self.train_num_batches = self.x_train.size(0) // batch_size

        train_nos = self.train_num_batches * batch_size

        self.x_train = self.x_train[: train_nos].view(self.train_num_batches, batch_size, self.x_train.size(1))
        self.y_train = self.y_train[: train_nos].view(self.train_num_batches, batch_size, self.y_train.size(1))
        
        if mask_train_name != None:
            self.mask_train = torch.load(self.save_path + mask_train_name + '.pt')
            self.mask_train = self.mask_train[: train_nos].view(self.train_num_batches, batch_size, self.mask_train.size(1))

        if x_val_name != None:
            self.x_val = torch.load(self.save_path + x_val_name + '.pt')
            self.val_num_batches = self.x_val.size(0) // batch_size
            val_nos = self.val_num_batches * batch_size
            self.x_val = self.x_val[: val_nos].view(self.val_num_batches, batch_size, self.x_val.size(1))

        if y_val_name != None:
            self.y_val = torch.load(self.save_path + y_val_name + '.pt')
            self.val_num_batches = self.y_val.size(0) // batch_size
            val_nos = self.val_num_batches * batch_size
            self.y_val = self.y_val[: val_nos].view(self.val_num_batches, batch_size, self.y_val.size(1))

        if mask_val_name != None:
            self.mask_val = torch.load(self.save_path + mask_val_name + '.pt')
            self.val_num_batches = self.mask_val.size(0) // batch_size
            val_nos = self.val_num_batches * batch_size
            self.mask_val = self.mask_val[: val_nos].view(self.val_num_batches, batch_size, self.mask_val.size(1))

    def load_shard  (
                        self,
                        shard_name: str,
                        train_val: str
                    ):
        batch_toks = self.batch_size * self.context_size

        if not (self.batch_overlap >= 0 and self.batch_overlap <= self.context_size and (self.batch_overlap % 1) == 0):
            raise ValueError("'batch_overlap' must be between 0 and 'context_size'") 

        data = torch.load(self.save_path + shard_name + '.pt')

        x_data = data[:-1]
        y_data = data[1:]

        batch_non_overlap = self.context_size - self.batch_overlap

        print([(batch_jump + batch_st, batch_jump + batch_st + batch_toks) for batch_jump in range(0, len(data) - (batch_toks * 2), batch_toks) for batch_st in range(0, self.context_size, batch_non_overlap)][-1])

        x_data = torch.stack([x_data[batch_jump + batch_st : batch_jump + batch_st + batch_toks] for batch_jump in range(0, len(data) - (batch_toks * 2), batch_toks) for batch_st in range(0, self.context_size, batch_non_overlap)], dim = 0)
        y_data = torch.stack([y_data[batch_jump + batch_st : batch_jump + batch_st + batch_toks] for batch_jump in range(0, len(data) - (batch_toks * 2), batch_toks) for batch_st in range(0, self.context_size, batch_non_overlap)], dim = 0)
        
        x_data = x_data.view(-1, self.batch_size, self.context_size)
        y_data = y_data.view(-1, self.batch_size, self.context_size)

        x_data = x_data.to(self.x_dtype)
        y_data = y_data.to(self.y_dtype)

        num_batches = x_data.size(0)

        if train_val == 'train':
            self.x_train = x_data
            self.y_train = y_data
            self.train_num_batches = num_batches
            print(f"Loaded 'Train' Shard - {shard_name}")
        elif train_val == 'val':
            self.x_val = x_data
            self.y_val = y_data
            self.val_num_batches = num_batches
            print(f"Loaded 'Val' Shard - {shard_name}")
        else:
            raise ValueError('Wrong split name! Expected "train" or "val".')

        gc.collect()

    def load_data   (
                        self, 
                        batch_size: int,
                        context_size: int,
                        train_shard_names: list,
                        batch_overlap: float, # Should be between 0 and context size.
                        x_dtype: torch.dtype,
                        y_dtype: torch.dtype,
                        val_name = None,
                        load_shard_indx = 0,
                        load_train_batch_indx = 0
                    ):

        self.batch_size = batch_size
        self.context_size = context_size
        self.train_shard_names = train_shard_names
        self.batch_overlap = batch_overlap
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

        self.shard_indx = load_shard_indx
        self.tot_shards = len(train_shard_names)

        self.load_shard(train_shard_names[self.shard_indx], train_val = 'train')
        
        self.train_batch_index = load_train_batch_indx

        print(f'Batch Index of First Batch to load: {self.train_batch_index}')

        if val_name != None:
            self.load_shard(val_name, train_val = 'val')

    def get_ft_train_batch  (
                                self, 
                                device: torch.device
                            ):
        batch_x = self.x_train[self.train_batch_index].to(device)
        batch_y = self.y_train[self.train_batch_index].to(device)

        self.train_batch_index = (self.train_batch_index + 1) % self.train_num_batches

        return batch_x, batch_y

    def get_train_batch (
                            self, 
                            device: torch.device
                        ):
        batch_x = self.x_train[self.train_batch_index].to(device)
        batch_y = self.y_train[self.train_batch_index].to(device)

        self.train_batch_index = (self.train_batch_index + 1) % self.train_num_batches

        if self.train_batch_index == (self.train_num_batches - 1):
            print(f"\nLoaded last batch of the Shard - {self.train_shard_names[self.shard_indx]}\n")

            self.train_batch_index = 0
            self.shard_indx = (self.shard_indx + 1) % self.tot_shards
            
            self.load_shard(self.train_shard_names[self.shard_indx], train_val = 'train')

        return batch_x, batch_y

    def get_val_batch   (
                            self,
                            device: torch.device
                        ):
        
        batch_x = self.x_val[self.val_batch_index].to(device)
        batch_y = self.y_val[self.val_batch_index].to(device)

        self.val_batch_index = (self.val_batch_index + 1) % self.val_num_batches
        
        return batch_x, batch_y
    
    def get_train_mask  (
                            self,
                            device: torch.device
                        ):
                        
        return self.mask_train[self.train_batch_index - 1 if self.train_batch_index != 0 else self.train_num_batches - 1].to(torch.int32).to(device)

    def get_val_mask    (
                            self,
                            device: torch.device
                        ):

        return self.mask_val[self.val_batch_index - 1 if self.val_batch_index != 0 else self.val_num_batches - 1].to(torch.int32).to(device)

    def set_indx (
                    self,
                    batch_index: int,
                    train_val: str
                 ):

        if train_val == 'train':
            self.train_batch_index = batch_index
        elif train_val == 'val':
            self.val_batch_index = batch_index
        else:
            raise ValueError('Wrong split name! Expected "train" or "val".')

    def reset   (
                    self,
                    train_or_val: str
                ):

        if train_or_val == 'train':
            self.train_batch_index = 0
        elif train_or_val == 'val':
            self.val_batch_index = 0
        else:
            raise ValueError('Wrong split name! Expected "train" or "val".')

if __name__ == '__main__':
    # data_extraction_task = PRE_TRAIN
    # data_extraction_task = FINE_TUNE

    if data_extraction_task == PRE_TRAIN:
        save_path = './Data/'
        file_path = './Training_Docs/Sanskrit_Corpus/Final Corpus/'
        vocab_size = 12000
        train_name = f'Final-{vocab_size}-Sanskrit-Train'
        val_name = f'Final-{vocab_size}-Sanskrit-Val'
        train_split_percentage = 0.98

        vocab_path = './Final Tokenizer/'
        merge_info_name = 'Final-Corpus-Tokenizer-Merge-Info-NL-12000-2024-08-31 03-04-04'
        vocab_name = 'Final-Corpus-Tokenizer-Vocab-NL-12000-2024-08-31 03-04-04'
        replacements = {}

        without_new_line = False
        skip_first_chunk_in_line = False
        remove_last_n_merge_info = True

    elif data_extraction_task == FINE_TUNE:
        csv_path = ''

        vocab_path = ''
        vocab_size = None
        merge_info_name = ''
        vocab_name = ''
        replacements = {}
        without_new_line = None
        skip_first_chunk_in_line = None

        shuffle = None
        train_split_percentage = None
        train_name = f''
        val_name = f''
        save_path = ''
        max_pad_len = None
        truncation = None

    data_loader = DataLoader(
                                save_path = save_path
                            )

    if data_extraction_task == PRE_TRAIN:
        data_loader.extractData_MultipleDocs(
                                        file_path = file_path,
                                        train_split_percentage = train_split_percentage,
                                        vocab_path = vocab_path,
                                        merge_info_name = merge_info_name,
                                        vocab_name = vocab_name,
                                        replacements = replacements,
                                        without_new_line = without_new_line,
                                        skip_first_chunk_in_line = skip_first_chunk_in_line,
                                        train_name = train_name,
                                        val_name = val_name,
                                        remove_last_n_merge_info = remove_last_n_merge_info
                                    )

    elif data_extraction_task == FINE_TUNE:
        data_loader.extract_Analogies_data  (
                                                csv_path = csv_path,
                                                vocab_path = vocab_path, 
                                                merge_info_name = merge_info_name, 
                                                vocab_name = vocab_name,
                                                replacements = replacements,
                                                without_new_line = without_new_line,
                                                skip_first_chunk_in_line = skip_first_chunk_in_line,
                                                shuffle = shuffle,
                                                train_split_percentage = train_split_percentage,
                                                train_name = train_name,
                                                max_pad_len = max_pad_len,
                                                truncation = truncation,
                                                val_name = val_name
                                            )