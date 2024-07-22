import re
from pathlib import Path
from typing import List

class Tokenizer:
    def __init__(self, corpus_file_path: Path) -> None:
        self.pattern = r"(--|[,.?_:;\"'()!]|\s)" 
        with open(corpus_file_path, "r") as f:
            raw_text = f.read()
        tokens = self.tokenize(raw_text)
        unique_tokens = sorted(list(set(tokens)))
        unique_tokens.extend(["<|endoftext|>"])
        
        self.vocab = {j:i for i,j in enumerate(unique_tokens)}
        self.inv_vocab = {j:i for i,j in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def tokenize(self, text: str) -> List[str]:
        tokens = re.split(self.pattern, text)
        tokens = [i.strip() for i in tokens if i.strip()]
        return tokens
        
    def encode(self, text: str) -> List[str]:
        tokens = self.tokenize(text)
        token_ids = []
        for token in tokens:
            if token in self.vocab.keys():
                token_id = self.vocab[token]
            else:
                token_id = self.vocab["<|endoftext|>"]
            token_ids.append(token_id)
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inv_vocab[i] for i in token_ids]
        text = " ".join(tokens)
        text = re.sub(r"\s+(--|-|[_\(\)':;,.?!])", r"\1", text) # Added to fix whitespace
        text = re.sub(r"(--|-|[_\(\)'])\s+", r"\1", text) # Added to fix whitespace
        return text

if __name__ == "__main__":
    import random
    random.seed(42) # For reproducing the results
    current_dir = Path.cwd() 
    tokenizer = Tokenizer(Path(current_dir, "tokenizer/corpus/pg1513.txt"))

    print(f"Size of vocabulary: {tokenizer.vocab_size}")
    print(f"Random 10 tokens from vocabulary: ", 
          random.sample(list(zip(tokenizer.vocab.items())), k=10))
    print(f"<|endoftext|> token id: ", tokenizer.vocab["<|endoftext|>"])

    orig_text = "I love machine learning. Do you also love ML?"
    token_ids = tokenizer.encode(text=orig_text)
    dec_text = tokenizer.decode(token_ids=token_ids)
    print(orig_text)
    print(token_ids)
    print(dec_text)