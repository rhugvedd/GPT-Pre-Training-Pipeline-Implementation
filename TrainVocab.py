import torch
import time

from BPETokenizer import *

"""
Following is a List of Hyperparameters:
"""
VocabSize = (1024 * 12)
# File = "Final - Pre-Train Data.txt"
File = "Components.txt"
FilePath = "./Data/"
replacements = {}
VocabPath = "./Vocab/"
Load_MergeInfo_Name = ''
Load_Vocab_Name = ''
Save_MergeInfo_Name = 'Component_MergeInfo-' + str(VocabSize) + '-'
Save_Vocab_Name = 'Component_Vocab-' + str(VocabSize) + '-'
SkipFirstChunkInLine = False
WithoutNewLine = True
Tokenization_File = './Tokenizer Output/Tokenized-' + str(VocabSize) + '.json'
"""
List Ends
"""

StartTime = time.time()

Tokenizer = BPETokenizer()

Tokenizer.load("./Vocab/", Load_MergeInfo_Name, Load_Vocab_Name)

TextTokens, Vocab = Tokenizer.TrainVocab(
                                            FilePath + File, 
                                            VocabSize, 
                                            PrintStat = True, 
                                            PrintStatsEvery_Token = 1, 
                                            WithoutNewLine = WithoutNewLine, 
                                            SkipFirstChunkInLine = SkipFirstChunkInLine, 
                                            Replacements = replacements
                                        )

Tokenizer.save(VocabPath, Save_MergeInfo_Name, Save_Vocab_Name)

Tokenizer.PrintTokenizedText(TextTokens, SaveFilePath = Tokenization_File)

print(f"Time Taken: {time.time() - StartTime} s")