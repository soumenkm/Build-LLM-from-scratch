import tiktoken, random
random.seed(42)

tokenizer = tiktoken.get_encoding("gpt2") # similar to tokenizer = Tokenizer(...)

print(f"Size of vocabulary: {tokenizer.n_vocab}")
print("Random 10 tokens from vocabulary: ", 
    random.sample(list(zip(tokenizer._mergeable_ranks.items())), k=10))
print("Special tokens: ", tokenizer._special_tokens)
print("<|endoftext|> token ID: ", tokenizer.eot_token) # index starts from 0!

text = "Hello there! How are you doing today? <|endoftext|> Do you like movies?"
en1 = tokenizer.encode(text, disallowed_special=(tokenizer.special_tokens_set - {'<|endoftext|>'}))
de1 = tokenizer.decode(en1)
en2 = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
de2 = tokenizer.decode(en2)
print(text)
print([tokenizer.decode([i]) for i in en1])
print(de1)
print([tokenizer.decode([i]) for i in en2])
print(de2)
