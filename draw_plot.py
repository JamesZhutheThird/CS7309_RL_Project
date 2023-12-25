import pdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import os

for env_name in ["VideoPinball-ramNoFrameskip-v4", "BreakoutNoFrameskip-v4", "PongNoFrameskip-v4", "BoxingNoFrameskip-v4"]:


    log_path= os.path.join("results/DQN/", f"{env_name}/log.txt")

    '''
    Model: DQN
    Environment: Boxing 
    ========================================
    Step: 16000 | Average Reward: nan | Time: 24.70
    Step: 32000 | Average Reward: 2.062 | Time: 50.52
    Step: 48000 | Average Reward: 2.062 | Time: 77.12
    ...
    '''
    step_list = []
    reward_list = []

    for line in open(log_path, "r"):
        if line.startswith("Step"):
            step = int(line.split("|")[0].split(":")[1])
            reward = float(line.split("|")[1].split(":")[1])
            time = float(line.split("|")[2].split(":")[1])
            step_list.append(step)
            reward_list.append(reward)

    plt.figure(figsize=(10, 5))
    plt.plot(step_list, reward_list)
    plt.xlabel("Training Steps")
    plt.ylabel("Average Reward")
    plt.title(f"Reward vs. Steps for DQN on {env_name}")
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{:,.0f}".format(x / 1000) + "k"))
    plt.savefig(log_path.replace("log.txt", f"Rewards_DQN_{env_name.removesuffix('NoFrameskip-v4')}.pdf"), bbox_inches="tight")
    plt.close()