
from src.utils import TOKENIZER

output = [13, 285, 253, 187, 187, 25645, 457, 84, 2589, 369, 417, 773, 328, 6937, 1020, 668, 762, 253, 16650, 10737, 15, 4031, 15, 387, 187, 187, 883, 15, 535, 50272, 510, 3286, 1302, 671, 1119, 326, 253, 3257, 457, 84, 2589, 369, 417, 187, 187, 1628, 328, 6937, 1020,
          668, 762, 253, 16650, 10737, 15, 4031, 15, 387, 1249, 15, 380, 1302, 1119, 326, 253, 187, 187, 25645, 457, 84, 2589, 369, 773, 328, 6937, 1020, 668, 762, 253, 16650, 10737, 15, 4031, 15, 387, 2145, 15, 187, 187, 510, 1302, 671, 1119, 326, 253, 3257, 457, 84, 2589, 369]
WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
print(f'\nLoading tokenizer {WORD_NAME}...')
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

print(tokenizer.tokenizer.decode(output))
