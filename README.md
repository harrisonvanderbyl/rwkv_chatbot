# rwkv_chatbot
rwkv_chatbot

This is an inference engine for RWKV 4.

run 
`python3 run.py`
for having a look at running
or
`TOKEN='discordtoken' python3 chat.py`
to run a discord bot using RWKV

To load a model, just download it and have it in the root folder of this project.

When you run the program, you will be prompted on what file to use,
what device (cpu,gpu) to use.

You will also be able to specify how many layers of the model you want to run on the available gpus.
Any unallocated layers will be loaded dynamically onto the nth-1 cuda device

You can also specify the float mode to use: fp32, fp16, or bf16 (some may not work on your device, and inference speed varies)

