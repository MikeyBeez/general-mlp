"""Character-level corpus: train on Shakespeare + Twain, test on Dickens (held-out author)."""

import os
import re
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SMALL_TRAIN = ["shakespeare.txt", "twain_huck.txt", "twain_sawyer.txt"]
SMALL_TEST = ["dickens_two_cities.txt", "dickens_great_expectations.txt"]
BIG_TRAIN = ["gut_100.txt", "twain_huck.txt", "twain_sawyer.txt", "gut_86.txt", "gut_1837.txt",
             "gut_3176.txt", "gut_245.txt", "gut_3177.txt", "gut_119.txt"]
BIG_TEST = SMALL_TEST + ["dickens_gut_1023.txt", "dickens_gut_580.txt", "dickens_gut_730.txt",
                         "dickens_gut_766.txt", "dickens_gut_963.txt"]
if os.environ.get("GMLP_CORPUS", "small") == "big":
    TRAIN_FILES, TEST_FILES = BIG_TRAIN, BIG_TEST
else:
    TRAIN_FILES, TEST_FILES = SMALL_TRAIN, SMALL_TEST


def strip_gutenberg(text):
    start = re.search(r"\*\*\* START OF [^\n]*\*\*\*", text)
    end = re.search(r"\*\*\* END OF [^\n]*\*\*\*", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]
    return text


def clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "--").replace("–", "-")
    text = re.sub(r"[^\x20-\x7e\n]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def load_file(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8", errors="replace") as f:
        return clean(strip_gutenberg(f.read()))


class Corpus:
    def __init__(self, val_fraction=0.05):
        train_texts = [load_file(n) for n in TRAIN_FILES]
        test_texts = [load_file(n) for n in TEST_FILES]
        chars = sorted(set("".join(train_texts + test_texts)))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab = len(chars)
        train_ids, val_ids = [], []
        for text in train_texts:
            ids = self.encode(text)
            cut = int(len(ids) * (1 - val_fraction))
            train_ids.append(ids[:cut])
            val_ids.append(ids[cut:])
        self.train = torch.cat(train_ids)
        self.val = torch.cat(val_ids)
        self.test = torch.cat([self.encode(t) for t in test_texts])

    def encode(self, text):
        return torch.tensor([self.stoi[c] for c in text if c in self.stoi], dtype=torch.long)

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def batch(self, split, batch_size, ctx, device, generator=None):
        data = {"train": self.train, "val": self.val, "test": self.test}[split]
        idx = torch.randint(0, len(data) - ctx - 1, (batch_size,), generator=generator)
        x = torch.stack([data[i : i + ctx] for i in idx])
        y = torch.stack([data[i + 1 : i + ctx + 1] for i in idx])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def summary(self):
        return {
            "vocab": self.vocab,
            "train_chars": len(self.train),
            "val_chars": len(self.val),
            "test_chars": len(self.test),
        }


if __name__ == "__main__":
    c = Corpus()
    print(c.summary())
    x, y = c.batch("test", 1, 120, "cpu")
    print(repr(c.decode(x[0].tolist())))
