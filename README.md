# rwkv_chatbot

rwkv_chatbot

This is an inference engine for RWKV 4.

I have ported the main inference engine to pypi.
https://github.com/harrisonvanderbyl/rwkvstic

Please `pip install rwkvstic`

run
`python3 runOptimised.py`
for having a look at running
or
`python3 ./multi.py`

to run a discord bot or for a chat-gpt like react-based frontend, and a simplistic chatbot backend server

To load a model, just download it and have it in the root folder of this project.

When you run the program, you will be prompted on what file to use,

You will also be prompted for the backend to use, such as jax,pytorch, or tensorflow, and a few more in there

You can also use export-torchscript option to export a .pt file that can be loaded in as a prebuilt torchscript file.

You may also be promped specify the float mode to use: fp64, fp32, fp16, or bf16 (some may not work on your device, and inference speed varies)

When using cuda, you can optionally specify another dtype to use during vector operations.
