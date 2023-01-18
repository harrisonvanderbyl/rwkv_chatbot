import tqdm
from bs4 import BeautifulSoup
from sys import argv
import requests
import torch
from src.rwkv import RWKV
model = RWKV("RWKV-3-Pile-20220720-10704.pth",
             mode="pytorch(cpu/gpu)", useGPU=False, dtype=torch.float32)
article = argv[-1]

# download article

text = requests.get(article).text
# convert html to text

soup = BeautifulSoup(text, 'html.parser')
text = soup.get_text()


splits = text.split("\n")
splits = [split for split in splits if len(split.strip()) > 15]

o, start = model.loadContext("\n",
                             f"Does the following excerpt answer the question: {input('What question would you like answered?')}? \n")

results = []

for split in tqdm.tqdm(splits):
    mystart = start
    model.setState(mystart)
    model.loadContext("\n", split+"\nYes/No: ")
    likelyhood = model.forward()["logits"][model.tokenizer.encode("Yes")[0]]
    results += [(likelyhood, split)]

results = sorted(results, key=lambda x: x[0], reverse=True)
print("\n\n")

for result in results[:10]:
    print(result[1])
    print("Likelyhood: ", result[0])
    print("")
