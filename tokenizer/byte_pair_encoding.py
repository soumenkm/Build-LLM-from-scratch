import tiktoken

enc = tiktoken.get_encoding("gpt2")
bpe_ranks = enc._mergeable_ranks
print(f"Sample vocabulary: {list(bpe_ranks.items())[40000:40010]}")
print(f"Length of vocab: {enc.n_vocab}")
print(f"Special tokens: {enc._special_tokens}")

text = "Hello there! How are you doing today? <|endoftext|> Do you like movies?"
en = enc.encode(text, allowed_special={'<|endoftext|>'})
de = enc.decode(en)
print(text)
print(en)
print(de)
