from rwkvstic.load import RWKV
instruct = "https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-Instruct-test1-20230124.pth"
model = RWKV(
    "RWKV-4-Pile-3B-20221005-7348.pth"
)

while 1:
    model.resetState()
    model.loadContext(newctx=f"Q: {input('Question:')}\n\nA:")
    print(model.forward(stopStrings=["<|endoftext|>"], number=100)["output"])
