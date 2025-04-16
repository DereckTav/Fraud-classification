from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

folder = "corpus_tokenizer\\"

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=True) #all other cases are true strip_accents, etc...
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=50000, special_tokens=special_tokens)
tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wiki_corpus_files\\wikipedia_corpus.txt"], trainer=trainer)

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
)

encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens)
print(encoding.type_ids)

tokenizer.decoder = decoders.WordPiece(prefix="##")
tokenizer.save("tokenizer.json")
# new_tokenizer = Tokenizer.from_file("tokenizer.json")