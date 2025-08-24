from transformers import AutoTokenizer
from pprint import pprint

def show_encoding(tok, text, name=""):
    enc = tok(text, return_tensors=None)
    print(f"\n=== {name} ===")
    print("Text:", text)
    print("Tokens:", tok.convert_ids_to_tokens(enc["input_ids"]))
    print("Input IDs:", enc["input_ids"])
    print("Attention mask:", enc["attention_mask"])

def show_batch(tok, texts, name=""):
    enc = tok(texts, return_tensors=None, padding=True, truncation=True, max_length=16)
    print(f"\n=== Batch ({name}) ===")
    print("Texts:", texts)
    print("Input IDs:")
    pprint(enc["input_ids"])
    print("Attention mask:")
    pprint(enc["attention_mask"])
    print("Decode[0]:", tok.decode(enc["input_ids"][0]))

def main():
    en_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ar_tok = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")

    # Single text examples
    show_encoding(en_tok, "The model is amazing!", "English / single")
    show_encoding(ar_tok, "النموذج رائع!", "Arabic / single")

    # Batch encoding
    show_batch(en_tok, ["Transformers are powerful.", "Tokenization matters a lot."], "English / batch")
    show_batch(ar_tok, ["هذه تجربة ترميز.", "المحولات قوية جدًا لمعالجة اللغة."], "Arabic / batch")

    # Special tokens
    print("\n=== Special Tokens (EN) ===")
    print(en_tok.special_tokens_map)
    print("Vocab size (EN):", en_tok.vocab_size)

if __name__ == "__main__":
    main()
