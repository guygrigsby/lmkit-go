#!/usr/bin/env python3
"""Generate the byte-level BPE equivalence fixture from the reference Python
tokenizer. Run in a venv with `tokenizers`:
    python gen_encodings.py  # reads tokenizer.json, writes encodings.json
"""
import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
SAMPLES = [
    "Hello, world!",
    "  leading and   multiple   spaces ",
    "trailing space ",
    "no_prefix",
    "Punctuation?! (parens) — em-dash, and 'quotes'.",
    "Numbers 12345 and 6.78 mixed with words2024.",
    "Unicode: café, naïve, résumé, 日本語, emoji 🙂.",
    "Newlines\nand\ttabs\tin text.",
    "A longer sentence of clean English prose, the kind the corpus is full of, "
    "to exercise common merges over ordinary words and spacing.",
    "ALLCAPS and MixedCase and snake_case and kebab-case.",
    # special/added token cases
    "<|endoftext|>",
    "hello <|endoftext|> world",
    "<|im_start|>user\nhi<|im_end|>",
]
out = [{"text": s, "ids": tok.encode(s).ids} for s in SAMPLES]
with open("encodings.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=0)
print(f"wrote encodings.json: {len(out)} cases")
