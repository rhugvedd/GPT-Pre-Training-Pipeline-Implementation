import torch
from BPETokenizer import *
from datetime import datetime
import pandas as pd

class DataLoader:
    def __init__    (
                        self,
                        save_path: str
                    ):
        super(DataLoader, self).__init__()

        self.save_path = save_path
        self.train_batch_index = 0
        self.val_batch_index = 0

    def extractData_toMem(
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
                            val_name: str
                        ):
        
        Tokenizer = BPETokenizer()
        Tokenizer.load(vocab_path, merge_info_name, vocab_name)

        print("Tokenizing Data")

        Tokens = Tokenizer.Encode(file_path, WithoutNewLine = without_new_line, SkipFirstChunkInLine = skip_first_chunk_in_line, Replacements = replacements)
        Tokens = [t for sub_t in Tokens for t in sub_t]

        print(f"Total Tokens: {len(Tokens)}")

        data = torch.tensor(Tokens)
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

        print("Saved Succesfuly to {save_path}")

    def extract_QAdata_toMem(
                                self,
                                file_path: str,
                                vocab_path: str, 
                                merge_info_name: str, 
                                vocab_name: str,
                                replacements: dict,
                                without_new_line: bool,
                                skip_first_chunk_in_line: bool,
                                max_len: int,
                                truncation: bool,
                                shuffle: bool,
                                train_split_percentage: float,
                                context_available: bool,
                                to_lower_case: bool,
                                train_name: str,
                                val_name:str
                            ):

        data = pd.read_csv(file_path)
        
        if to_lower_case:
            data = data.map(lambda x: x.lower() if isinstance(x, str) else x)

        data = data.dropna(subset = ['Question', 'Answer'])

        Questions = data['Question'].tolist()
        Answers = data['Answer'].tolist()
        
        if context_available:
            Contexts = []
            Contexts = ["" if pd.isna(context) else context for context in data['Contexts']]

        Tokenizer = BPETokenizer()
        Tokenizer.load(vocab_path, merge_info_name, vocab_name)

        x_data, y_data, mask = Tokenizer.EncodeQuestionAnswer  (
                                                                    Questions = Questions,
                                                                    Answers = Answers,
                                                                    MaxLen = max_len,
                                                                    Truncation = truncation,
                                                                    WithoutNewLine = without_new_line,
                                                                    SkipFirstChunkInLine = skip_first_chunk_in_line,
                                                                    Replacements = replacements,
                                                                    Contexts = Contexts if context_available else None
                                                                )

        shuffle_idices = torch.randperm(x_data.size(0))
        
        x_data = x_data.index_select(0, shuffle_idices)
        y_data = y_data.index_select(0, shuffle_idices)
        mask = mask.index_select(0, shuffle_idices)

        split_thres = int(train_split_percentage * x_data.size(0))

        x_train = x_data[:split_thres]
        x_val = x_data[split_thres:]

        y_train = y_data[:split_thres]
        y_val = y_data[split_thres:]

        mask_train = mask[:split_thres]
        mask_val = mask[split_thres:]

        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':', '-')

        torch.save(x_train, self.save_path + 'X-' + train_name + date_time + '.pt')
        torch.save(x_val, self.save_path + 'X-' + val_name + date_time + '.pt')

        torch.save(y_train, self.save_path + 'Y-' + train_name + date_time + '.pt')
        torch.save(y_val, self.save_path + 'Y-' + val_name + date_time + '.pt')
        
        torch.save(mask_train, self.save_path + 'Mask-' + train_name + date_time + '.pt')
        torch.save(mask_val, self.save_path + 'Mask-' + val_name + date_time + '.pt')

        print("Saved Successfully..!!")
        
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

            if reset_batch_index:
                self.train_batch_index = 0

        elif split == 'val':
            print("SHUFFLING VAL BATCHES")
            shuffle_indices = torch.randperm(self.x_val.size(0))
            self.x_val = self.x_val.index_select(0, shuffle_indices)
            self.y_val = self.y_val.index_select(0, shuffle_indices)
            
            if reset_batch_index:
                self.val_batch_index = 0
        else:
            raise ValueError("Wrong split name!")

    def load_fine_tune_data (   
                                self,
                                batch_size: int,
                                x_train_name: str,
                                y_train_name: str,
                                mask_train_name: str,
                                x_val_name: str = None,
                                y_val_name: str = None,
                                mask_val_name: str = None
                            ):

        self.x_train = torch.load(self.save_path + x_train_name + '.pt')
        self.y_train = torch.load(self.save_path + y_train_name + '.pt').to(torch.int64)
        self.mask_train = torch.load(self.save_path + mask_train_name + '.pt')

        self.train_num_batches = self.x_train.size(0) // batch_size

        train_nos = self.train_num_batches * batch_size

        self.x_train = self.x_train[: train_nos].view(self.train_num_batches, batch_size, self.x_train.size(1))
        self.y_train = self.y_train[: train_nos].view(self.train_num_batches, batch_size, self.y_train.size(1))
        self.mask_train = self.mask_train[: train_nos].view(self.train_num_batches, batch_size, self.mask_train.size(1), self.mask_train.size(1))

        if x_val_name != None:
            self.x_val = torch.load(self.save_path + x_val_name + '.pt')
            self.val_num_batches = self.x_val.size(0) // batch_size
            val_nos = self.val_num_batches * batch_size
            self.x_val = self.x_val[: val_nos].view(self.val_num_batches, batch_size, self.x_val.size(1))

        if y_val_name != None:
            self.y_val = torch.load(self.save_path + y_val_name + '.pt').to(torch.int64)
            self.val_num_batches = self.y_val.size(0) // batch_size
            val_nos = self.val_num_batches * batch_size
            self.y_val = self.y_val[: val_nos].view(self.val_num_batches, batch_size, self.y_val.size(1))

        if mask_val_name != None:
            self.mask_val = torch.load(self.save_path + mask_val_name + '.pt')
            self.val_num_batches = self.mask_val.size(0) // batch_size
            val_nos = self.val_num_batches * batch_size
            self.mask_val = self.mask_val[: val_nos].view(self.val_num_batches, batch_size, self.mask_val.size(1))


    def clean_text(self, text):
        
        text = re.sub(r'[0-9\.\|"।]', '', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'(\r\n){2,}', '\n', text)

        return text
    
    def load_data   (
                        self, 
                        batch_size: int,
                        context_size: int,
                        train_val: str,
                        name: str,
                        batch_overlap: float, # Should be between 0 and context size.
                        x_dtype: torch.dtype,
                        y_dtype: torch.dtype
                    ):

        batch_toks = batch_size * context_size

        if not (batch_overlap >= 0 and batch_overlap <= context_size and (batch_overlap % 1) == 0):
            raise ValueError("'batch_overlap' must be between 0 and 'context_size'") 

        data = torch.load(self.save_path + name + '.pt')

        x_data = data[:-1]
        y_data = data[1:]

        batch_non_overlap = context_size - batch_overlap

        # print([(batch_jump + batch_st, batch_jump + batch_st + batch_toks) for batch_jump in range(0, len(data) - (batch_toks * 2), batch_toks) for batch_st in range(0, context_size, batch_non_overlap)])

        x_data = torch.stack([x_data[batch_jump + batch_st : batch_jump + batch_st + batch_toks] for batch_jump in range(0, len(data) - (batch_toks * 2), batch_toks) for batch_st in range(0, context_size, batch_non_overlap)], dim = 0)
        y_data = torch.stack([y_data[batch_jump + batch_st : batch_jump + batch_st + batch_toks] for batch_jump in range(0, len(data) - (batch_toks * 2), batch_toks) for batch_st in range(0, context_size, batch_non_overlap)], dim = 0)
        
        x_data = x_data.view(-1, batch_size, context_size)
        y_data = y_data.view(-1, batch_size, context_size)

        x_data = x_data.to(x_dtype)
        y_data = y_data.to(y_dtype)

        num_batches = x_data.size(0)

        if train_val == 'train':
            self.x_train = x_data
            self.y_train = y_data
            self.train_num_batches = num_batches
        elif train_val == 'val':
            self.x_val = x_data
            self.y_val = y_data
            self.val_num_batches = num_batches
        else:
            raise ValueError('Wrong split name! Expected "train" or "val".')

    def get_train_batch (
                            self, 
                            device: torch.device
                        ):
        batch_x = self.x_train[self.train_batch_index].to(device)
        batch_y = self.y_train[self.train_batch_index].to(device)

        self.train_batch_index = (self.train_batch_index + 1) % self.train_num_batches
        
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
                        
        return self.mask_train[self.train_batch_index - 1 if self.train_batch_index != 0 else self.train_num_batches - 1].to(device)

    def set_indx (
                    self,
                    batch_index: int,
                    train_val: str
                 ):

        if train_or_val == 'train':
            self.train_batch_index = batch_index
        elif train_or_val == 'val':
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
    """
    Parameters' Start:

    Pre-Training:
    """
    # save_path = './Data Tensors/'
    # file_path = './Data/Boylestad-Malvino - Pre-Train Data.txt'
    # train_name = 'Boylestad-Malvino - Train-Data-10240-'
    # val_name = 'Boylestad-Malvino - Val-Data-10240-'
    # train_split_percentage = 0.9

    # vocab_path = './Vocab/'
    # merge_info_name = 'Electronics_MergeInfo-10240-2024-06-17 22-32-43'
    # vocab_name = 'Electronics_Vocab-10240-2024-06-17 22-32-43'
    # replacements = {}

    # without_new_line = True
    # skip_first_chunk_in_line = False

    # save_path = './Data Tensors/'
    # file_path = './Data/Components.txt'
    # train_name = 'Components-Train-'
    # val_name = 'Components-Val-'
    # train_split_percentage = 0.9

    # vocab_path = './Vocab/'
    # merge_info_name = 'Electronics_MergeInfo-10408-2024-06-26 19-16-53'
    # vocab_name = 'Electronics_Vocab-10408-2024-06-26 19-16-53'
    # replacements = {}

    # without_new_line = False
    # skip_first_chunk_in_line = False
    """
    Fine Tuning:
    """
    save_path = './Fine Tune Data/'
    train_name = 'FT-Components-Train-'
    val_name = 'FT-Components-Val-'
    train_split_percentage = 1
    
    vocab_path = './Vocab/'
    merge_info_name = 'Electronics_MergeInfo-10408-2024-06-26 19-16-53'
    vocab_name = 'Electronics_Vocab-10408-2024-06-26 19-16-53'
    replacements = {}

    without_new_line = False
    skip_first_chunk_in_line = False

    QA_file_path = './Data/Components-QA-Pairs.csv'
    context_available = False
    max_len = 128
    truncation = False
    shuffle = True
    to_lower_case = True
    """
    Hyper Parameters End
    """

    data_loader = DataLoader(
                                save_path = save_path
                            )

    # data_loader.extractData_toMem(
    #                                 file_path = file_path,
    #                                 train_split_percentage = train_split_percentage,
    #                                 vocab_path = vocab_path,
    #                                 merge_info_name = merge_info_name,
    #                                 vocab_name = vocab_name,
    #                                 replacements = replacements,
    #                                 without_new_line = without_new_line,
    #                                 skip_first_chunk_in_line = skip_first_chunk_in_line,
    #                                 train_name = train_name,
    #                                 val_name = val_name
    #                             )

    data_loader.extract_QAdata_toMem(
                                        file_path=QA_file_path,
                                        vocab_path=vocab_path,
                                        merge_info_name=merge_info_name,
                                        vocab_name=vocab_name,
                                        replacements=replacements,
                                        without_new_line=without_new_line,
                                        skip_first_chunk_in_line=skip_first_chunk_in_line,
                                        max_len=max_len,
                                        truncation=truncation,
                                        shuffle=shuffle,
                                        train_split_percentage=train_split_percentage,
                                        context_available=context_available,
                                        to_lower_case=to_lower_case,
                                        train_name=train_name,
                                        val_name=val_name
                                    )