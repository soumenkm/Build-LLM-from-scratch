import re

class Tokenizer:
    
    def __init__(self, corpus_file_path):
        self.corpus_file_path = corpus_file_path
        self.pattern = r"(--|[,.?_:;\"'()!]|\s)" 
        
        with open(self.corpus_file_path, "r") as f:
            self.raw_text = f.read()
        print(f"Number of characters in corpus: {len(self.raw_text)}")
        
        self.tokens = self.tokenize(self.raw_text)
        print(f"Number of tokens in corpus: {len(self.tokens)}")

        self.unique_tokens = sorted(list(set(self.tokens)))
        self.unique_tokens.extend(["<|unk|>", "<|endoftext|>"])
        print(f"Number of unique tokens in vocabulary: {len(self.unique_tokens)}")
        
        self.vocab = {j:i for i,j in enumerate(self.unique_tokens)}
        self.inv_vocab = {j:i for i,j in self.vocab.items()}
    
    def tokenize(self, text):
        tokens = re.split(self.pattern, text)
        tokens = [i.strip() for i in tokens if i.strip()]
        return tokens
    
    def encode(self, text):
        tokens = self.tokenize(text)
        token_ids = [self.vocab[i] if i in self.unique_tokens else self.vocab["<|unk|>"] for i in tokens]
        return token_ids
    
    def decode(self, token_ids):
        tokens = [self.inv_vocab[i] for i in token_ids]
        text = " ".join(tokens)
        text = re.sub(r"\s+(--|-|[_\(\)':;,.?!])", r"\1", text)
        text = re.sub(r"(--|-|[_\(\)'])\s+", r"\1", text)
        return text

tokenizer = Tokenizer("/raid/speech/soumen/build-llm/tokenizer/corpus/the-verdict.txt")
text='''Hello there, how are you? <|endoftext|> I am fine, you?'''
    
en = tokenizer.encode(text)
de = tokenizer.decode(en)
print(text)
print(de)