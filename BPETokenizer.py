import regex as re
import pickle
import datetime
import json
from collections import Counter
import itertools

class BPETokenizer:
    def __init__(self):
        self.MergeInfo = {}
        self.Vocab = [i for i in range(256)]
        self.special_tokens = ('<bos>', '<eos>', '<sep2>', '<sep>', '<pad>')

    def special_tok(self, tok):
        if tok in self.special_tokens:
            return(self.Vocab[-(len(self.special_tokens) - self.special_tokens.index(tok))])
        else:
            raise ValueError(f"Special Token '{tok}' not found in the Tokenizer")

    def GetBigramStats(self, Text, BigramStats):
        for Bigram in zip(Text, Text[1:]):
            BigramStats[Bigram] = BigramStats.get(Bigram, 0) + 1

    def Merge(self, TextTokens, TokensToMerge, NewToken):
        Indx = 0

        while Indx < len(TextTokens) - 1:
            if TextTokens[Indx] == TokensToMerge[0] and TextTokens[Indx + 1] == TokensToMerge[1]:
                TextTokens[Indx: Indx + 2] = [NewToken]

            Indx+=1

        return TextTokens

    def GetCleanedText(self, Text, WithoutNewLine, SkipFirstChunkInLine, Replacements):
        
        print('Cleaning and splitting text.')

        for old, new in Replacements.items():
            Text = Text.replace(old, new)

        if WithoutNewLine and SkipFirstChunkInLine:
            CleanedText = ' '.join([line[line.find(" ") + 1:] for line in Text.splitlines()])
        elif WithoutNewLine and (not SkipFirstChunkInLine):
            CleanedText = ' '.join([line for line in Text.splitlines()])
        elif (not WithoutNewLine) and SkipFirstChunkInLine:
            CleanedText = '\n'.join([line[line.find(" ") + 1:] for line in Text.splitlines()])
        elif (not WithoutNewLine) and (not SkipFirstChunkInLine):
            CleanedText = '\n'.join([line for line in Text.splitlines()])
        
        SPLIT_PATTERN = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

        SplitText = re.findall(GPT_SPLIT_PATTERN, CleanedText)

        return SplitText

    def TrainVocab(self, FilePath, VocabSize, PrintStat, PrintStatsEvery_Token, WithoutNewLine, SkipFirstChunkInLine, Replacements, RemoveSpecialTok):
        """
        This function is implemented in such a way that     
        it builds upon the existing vocabulary and does 
        not start the training from tha scratch. 
        This is implemented in such a way so that this 
        function can be called multiple times to train 
        a larger vocabulary size as needed by requirements 
        in further control flow.
        """

        if RemoveSpecialTok:
            print('Removing Special Tokens from the Vocabulary End')
            if len(self.Vocab) > 256:
                self.Vocab = self.Vocab[:-len(self.special_tokens)]

        Text = open(FilePath, "r", encoding="utf-8").read()
        return self.TrainVocab_fromText(Text, VocabSize, PrintStat, PrintStatsEvery_Token, WithoutNewLine, SkipFirstChunkInLine, Replacements)
            
    def count_bigrams(self, segment):
        return Counter(zip(segment, segment[1:]))

    def TrainVocab_fromText(self, Text, VocabSize, PrintStat, PrintStatsEvery_Token, WithoutNewLine, SkipFirstChunkInLine, Replacements): 
        
        TextTokens = self.EncodeFromText(Text, WithoutNewLine, SkipFirstChunkInLine, Replacements)

        Token = 0
        if(PrintStat):
            print(f"Vocab Training Started:\nTokens Trained:")
        
        while(len(self.Vocab) < (VocabSize - len(self.special_tokens))):

            Stats = Counter(itertools.chain.from_iterable(zip(seg, seg[1:]) for seg in TextTokens))

            Bigram = max(Stats, key = lambda item: Stats.get(item))
            NewToken = self.Vocab[-1] + 1
            self.MergeInfo[Bigram] = NewToken
            self.Vocab.append(NewToken)

            TextTokens = [self.Merge(seg, Bigram, NewToken) for seg in TextTokens]

            if PrintStat and (Token % PrintStatsEvery_Token == 0): 
                print(len(self.Vocab), end = " - Repetitions of Bigram: ")
                print(Stats[Bigram])

            Token += 1

            if len(self.Vocab) % 1000 == 0:
                self.save('./Tokenizer/', 'Final-Corpus-Tokenizer-Merge-Info-NL-' + str(len(self.Vocab)) + '-', 'Final-Corpus-Tokenizer-Vocab-NL-' + str(len(self.Vocab)) + '-', Save_SpecialTok = False)
                
                self.PrintTokenizedText(TextTokens, SaveFilePath = './Tokenizer/Tokenized-Final-Corpus-' + str(len(self.Vocab)) + '.json')

        return TextTokens, self.Vocab

    def Encode(self, FilePath, WithoutNewLine, SkipFirstChunkInLine, Replacements):
        Text = open(FilePath, "r", encoding="utf-8").read()

        return self.EncodeFromText(Text, WithoutNewLine, SkipFirstChunkInLine, Replacements)

    def DecodeVocab(self, SaveFilePath = None):
        
        DecodedBytes = {}
        
        for i in range(256):
            DecodedBytes[i] = bytes([i])

        for (m0, m1), mgd in self.MergeInfo.items():
            DecodedBytes[mgd] = DecodedBytes[m0] + DecodedBytes[m1]

        TokenizedText = [DecodedBytes[tok].decode("utf-8", errors="replace") for tok in self.Vocab[:-len(self.special_tokens)]]

        if SaveFilePath != None:
            with open(SaveFilePath, 'w', encoding='utf-8') as file:
                json.dump(TokenizedText, file, ensure_ascii=False, indent=4)

                print(f"Vocab saved to: {SaveFilePath}")

    def EncodeFromText(self, Text, WithoutNewLine, SkipFirstChunkInLine, Replacements):

        Text = self.GetCleanedText(Text, WithoutNewLine, SkipFirstChunkInLine, Replacements)

        Encoded = []

        if self.MergeInfo == {}:
            for seg in Text:
                Tokens = list(seg.encode("utf-8"))
                Encoded.append(Tokens)
                print('Vocab is Empty!!')
        else:
            print(f'Encoding from Vocab - (Size: {len(self.Vocab)})')
            for seg in Text:
                Tokens = list(seg.encode("utf-8"))

                while len(Tokens) >= 2:
                    Stats = {}
                    self.GetBigramStats(Tokens, Stats)
                    Bigram = min(Stats, key = lambda p: self.MergeInfo.get(p, float("inf")))

                    if Bigram not in self.MergeInfo:
                        break
                    
                    NewToken = self.MergeInfo[Bigram]
                    Tokens = self.Merge(Tokens, Bigram, NewToken)
                
                Encoded.append(Tokens)
        
        return Encoded

    def Decode(self, Tokens):
        
        DecodedBytes = {}
        
        for i in range(256):
            DecodedBytes[i] = bytes([i])

        for (m0, m1), mgd in self.MergeInfo.items():
            DecodedBytes[mgd] = DecodedBytes[m0] + DecodedBytes[m1]
        
        # These two approaches need to be switch according to the form in which Tokens is passed. Better logic can be implemented here and is a part of the To-Do list of the project.
        Bytes = [b"".join(DecodedBytes[idx] for idx in Tokens)]
        # Bytes = [b"".join(DecodedBytes[idx] for idx in seg) for seg in Tokens]
        
        return "".join(Chunk.decode("utf-8", errors="replace") for Chunk in Bytes)

    def PrintTokenizedText(self, Tokens, SaveFilePath = None):
        
        DecodedBytes = {}
        
        for i in range(256):
            DecodedBytes[i] = bytes([i])

        for (m0, m1), mgd in self.MergeInfo.items():
            DecodedBytes[mgd] = DecodedBytes[m0] + DecodedBytes[m1]
        
        TokenizedText = [[DecodedBytes[idx].decode("utf-8", errors="replace") for idx in seg] for seg in Tokens]

        if SaveFilePath != None:
            with open(SaveFilePath, 'w', encoding='utf-8') as file:
                json.dump(TokenizedText, file, ensure_ascii=False, indent=4)

                print(f"Tokenized text saved to: {SaveFilePath}")

        # print(TokenizedText)
    
    def save(self, path, MergeInfo_Name, Vocab_Name, Save_SpecialTok):
        
        if Save_SpecialTok:
            for i in range(len(self.special_tokens)):
                self.Vocab.append(len(self.Vocab))

        if(path[-1] != '/'): path += '/'
        
        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':', '-')
        
        with open(path + MergeInfo_Name + date_time + '.pkl', 'wb') as file:
            pickle.dump(self.MergeInfo, file)

        with open(path + Vocab_Name + date_time + '.pkl', 'wb') as file:
            pickle.dump(self.Vocab, file)

        print("Contents saved successfully.")
        
    def load(self, path, MergeInfo_Name, Vocab_Name):
        
        if(path[-1] != '/'): path += '/'

        with open(path + MergeInfo_Name + '.pkl', 'rb') as file:
            self.MergeInfo = pickle.load(file)
        
        with open(path + Vocab_Name + '.pkl', 'rb') as file:
            self.Vocab = pickle.load(file)

        print('Loaded Vocabulary!')
        print(f'Vocab Size: {len(self.Vocab)}')
        print(f'MergeInfo Size: {len(self.MergeInfo)}')