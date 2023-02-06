from rwkvstic.load import RWKV
import inquirer
links = {
    "3B": "https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-Instruct-test1-20230124.pth",
    "7B": "https://huggingface.co/BlinkDL/rwkv-4-pile-7b/resolve/main/RWKV-4-Pile-7B-Instruct-test1-20230124.pth",
    "14B": "https://huggingface.co/BlinkDL/rwkv-4-pile-14b/resolve/main/RWKV-4-Pile-14B-20230115-5775.pth"
}
model = RWKV(
    links[inquirer.prompt([inquirer.List("model_size", message="Model size", choices=[
                          "3B", "7B", "14B"])])["model_size"]]
)

while 1:
    model.resetState()
    model.loadContext(newctx=f"Q: {input('Question:')}\n\nA:")
    print(model.forward(stopStrings=["<|endoftext|>"], number=100)["output"])
