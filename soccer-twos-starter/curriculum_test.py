import time
import yaml

import soccer_twos
from soccer_twos import EnvType
from utils import sample_pos_vel, sample_player

current = 0
with open("curriculum.yaml") as f:
    curriculum = yaml.load(f, Loader=yaml.FullLoader)
tasks = curriculum["tasks"]
config_fns = {
    "none": lambda *_: None,
    "random_players": lambda env: env.set_policies(
        lambda *_: env.action_space.sample()
    ),
}


def reset_env(_env):
    print("task: ", tasks[current]["name"])
    _env.reset()
    print("appling config", tasks[current]["config_fn"])
    config_fns[tasks[current]["config_fn"]](env)
    print("setting parameters")
    _env.env_channel.set_parameters(
        ball_state=sample_pos_vel(tasks[current]["ranges"]["ball"]),
        players_states={
            player: sample_player(tasks[current]["ranges"]["players"][player])
            for player in tasks[current]["ranges"]["players"]
        },
    )


env = soccer_twos.make(
    base_port=8500,
    render=True,
    # watch=True,
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
reset_env(env)

time.sleep(2)
print("go")
while True:
    obs, reward, done, info = env.step(
        0
        # env.action_space.sample()
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
        reset_env(env)

        if current < len(tasks) - 1:
            current += 1
            reset_env(env)
