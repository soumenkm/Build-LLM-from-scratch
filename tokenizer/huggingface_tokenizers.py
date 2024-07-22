import tokenizers as tn
import random
random.seed(42)

model = tn.models.BPE()
tokenizer = tn.Tokenizer(model)

tokenizer.decoder = tn.decoders.BPEDecoder()
tokenizer.pre_tokenizer = tn.pre_tokenizers.Whitespace()
trainer = tn.trainers.BpeTrainer(vocab_size=1000,
                                 show_progress=True,
                                 special_tokens=["<|endoftext|>"])
tokenizer.train(files=["tokenizer/corpus/pg1513.txt"],
                trainer=trainer)

print(f"Size of vocabulary: {tokenizer.get_vocab_size()}")
print(f"Random 10 tokens from vocabulary: ", 
        random.sample(list(zip(tokenizer.get_vocab().items())), k=10))
print(f"<|endoftext|> token id: ", tokenizer.token_to_id("<|endoftext|>"))

text = "I love machine learning. <|endoftext|> Do you also love ML?"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)

print(text)
print(encoded.tokens)
print(encoded.ids)
print(decoded)
