import tokenizers as tn

tokenizer = tn.Tokenizer(tn.models.WordPiece())
tokenizer.decoder = tn.decoders.WordPiece()
tokenizer.pre_tokenizer = tn.pre_tokenizers.Whitespace()
print(tokenizer.model)
print(tokenizer.decoder)
print(tokenizer.pre_tokenizer)

trainer = tn.trainers.WordPieceTrainer(vocab_size=1000,
                                 show_progress=False,
                                 special_tokens=["<|endoftext|>", "<|unk|>","<|pad|>","<|mask|>"])

tokenizer.train(files=["/raid/speech/soumen/build-llm/corpus/the-verdict.txt"],
                trainer=trainer)

print(f"First 10 items in vocabulary: {list(tokenizer.get_vocab().items())[0:10]}")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
print(f"Token to id: {tokenizer.token_to_id('colour')}")
print(f"ID to token: {tokenizer.id_to_token(0)}")

text = "Hello there! How are you doing today? <|endoftext|> I like movies"
encoded = tokenizer.encode(text, is_pretokenized=False)
print(encoded.tokens)
print(encoded.ids)

decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
print(decoded)