import random

import soccer_twos
from soccer_twos import EnvType


env = soccer_twos.make(
    base_port=8500,
    # render=True,
    watch=True,
    flatten_branched=True,
    # time_scale=1,
    variation=EnvType.team_vs_policy,
    single_player=True,
    opponent_policy=lambda *_: 0,
)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space)

step = 0
team0_reward = 0
team1_reward = 0
env.reset()

env.env_channel.set_parameters(
    # ball_state={
    #     "position": [-14, 0],
    #     "velocity": [10, 0],
    # },
    players_states={
        1: {"rotation_y": 45, "position": [-14, 1.5],},
        # 1: {
        #     # "rotation_y": 45,
        #     "position": [-6, -1.5],
        # },
        # 2: {
        #     # "rotation_y": 45,
        #     "position": [6, 1.5],
        # },
        # 3: {
        #     # "rotation_y": 45,
        #     "position": [1, 1],
        # },
    },
)


while True:
    obs, reward, done, info = env.step(
        26
        # {
        #     0: 0,
        #     1: 0,
        #     2: 0,
        #     3: 0,
        #     # 0: env.action_space.sample(),
        #     # 1: env.action_space.sample(),
        #     # 2: env.action_space.sample(),
        #     # 3: env.action_space.sample(),
        # }
    )

    if step == 30:
        print("updating policy")
        env.set_opponent_policy(lambda *_: env.action_space.sample())

    # team0_reward += reward[0] + reward[1]
    # team1_reward += reward[2] + reward[3]
    step += 1
    if done:
        # if max(done.values()):  # if any agent is done
        # print(info[0]["player_info"]["position"])
        # print(info[1]["player_info"]["position"])
        # print(info[2]["player_info"]["position"])
        # print(info[3]["player_info"]["position"])
        # print("Total Reward: ", team0_reward, " x ", team1_reward)
        step = 0
        team0_reward = 0
        team1_reward = 0
        env.reset()
