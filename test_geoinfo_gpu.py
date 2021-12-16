#import torch
import wandb
import os

print(os.getcwd())
wandb.init(name="test", dir=os.getcwd())

gpu = torch.cuda.get_device_name(0)
wandb.log({"gpu": gpu})
print(gpu)
for i in range(1, 10):
    wandb.log({f"print_{i}":f"This is {i}"})

wandb.finish()
