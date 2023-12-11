import os
import time
import soccer_twos
import time
import numpy as np
from torch.distributions import Normal
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

from . import ppo

from clearml import Task
from clearml import Logger

global logger


def obs_convert(obs):
    """
    對 observation 做處理，讓每個位置的數值都乘上距離
    """
    output_shape = (3, 14*7)
    # obs 是一個 list，裡面有四個元素，每個元素都是一個 336 維的一維陣列
    new_obs_all = np.zeros((len(obs), output_shape[0], output_shape[1]))
    for player in range(len(obs)):
        # 對每一個 player 的 obs 做處理
        one_obs = obs[player]
        one_obs = np.array(one_obs).reshape(42, 8)
        
        # 將每一列的前七項乘上最後一項
        new_obs = one_obs[:, :-1] * one_obs[:, -1].reshape(-1, 1)
        
        # 重新排序，依照第 t-2 秒的前11項(0~10)、第 t-2 秒的後3項(34~36)、第 t-1 秒的前11項(11~21)、第 t-1 秒的後3項(37~39)、第 t 秒的前11項(22~33)、第 t 秒的後3項(40~42)
        new_index = list(range(0, 11)) + list(range(34, 37)) + list(range(11, 22)) + list(range(37, 40)) + list(range(22, 34)) + list(range(40, 42))
        new_obs = new_obs[new_index]
        # print(new_obs.shape)
                
        # 將 obs[player] 轉成一維陣列
        new_obs_array = new_obs.reshape(output_shape)
        # new_obs_array = new_obs_array.reshape(1, -1)
        # 把所有 player 的 obs 都放在 new_obs_all 裡面
        new_obs_all[player] = new_obs_array
    return new_obs_all


def environment():
    env = soccer_twos.make(
        render=False,
        # watch=True,
        flatten_branched=True,
        # time_scale=1,
        variation=soccer_twos.EnvType.multiagent_player,
        opponent_policy=lambda *_: 0,
        base_port=60005,
    )
    return env

def restart_env(env):
    # 初始化環境
    env.reset()
    # 初始動作，取得資訊
    state, _, _, info = env.step({0: 0, 1: 0, 2: 0, 3: 0})
    state = obs_convert(state)
    ball_before = info[0]["ball_info"]["position"]

    # 初始化與 reward 相關的list
    total_reward = [0, 0, 0, 0]
    stand_counter = [0, 0, 0, 0]

    return env, total_reward, stand_counter, state, ball_before, 0


def face_ball(obs, ratio=0.5):
    player_list = range(len(obs))
    # 處理 obs
    # obs = obs_convert(obs)
    reward_list = []
    # 為了讓整體reward不要太大或太小
    time_ratio = [x * 0.01 for x in [0.5, 0.7, 1]]    
    # 對每一個球員
    for player in player_list:
        reward = 0
        # 取出某一個球員的 obs
        player_obs = obs[player]
        # 對每一個時間點 t
        for i in range(len(player_obs)):
            t = player_obs[i]
            # 將資料 reshape 成 14 * 7 的矩陣
            # 第 t 秒時的所有lidar
            lidars = t.reshape(14, -1)

            # 如果前方 9 個 lidar 有照到球，依照角度給予不同的 reward
            if len(list(filter(lambda dist: dist > 0, lidars[:11, 0]))) == 0: 
                # 如果正對球 0 1 2，就加分 0.5 (每條正對的lidar都加)
                reward += (0.5 * ratio * time_ratio[i]) * len(list(filter(lambda dist: dist > 0, lidars[:3, 0])))
                # 如果側對球 3 4 5 6，就加分 0.2 (每條側對的lidar都加)
                reward += (0.2 * ratio * time_ratio[i]) * len(list(filter(lambda dist: dist > 0, lidars[3:7, 0])))
                # 如果斜對球 7 8 9 10，就加分 0.04
                reward += (0.04 * ratio * time_ratio[i]) * len(list(filter(lambda dist: dist > 0, lidars[7:11, 0])))
            else:
                # 如果前方 11 個 lidar 都沒有照到球，就扣分 -3
                reward -= 3 * ratio * time_ratio[i]
            
        reward_list.append(reward)
    return reward_list


# face_ball(obs_convert(obs))


def still_stand(actions, stand_counter, stand_step=15, ratio=0.5):
    ratio *= 0.01
    reward_list = [0, 0, 0, 0]
    # 判斷是否有人停在原地
    for id, action in actions.items():
        if action == 0:
            stand_counter[id] += 1
        elif action != 0 and stand_counter[id] > 0:
            stand_counter[id] = 0
    # 如果有人連續 15 步停在原地，就給 punishment    -5
    for id, counter in enumerate(stand_counter):
        if counter >= stand_step:
            reward_list[id] = -3 * ratio
            stand_counter[id] = 0
    return reward_list, stand_counter


def back_our_goal(obs, ratio=0.5):
    reward_list = []
    for player in range(len(obs)):
        reward = 0
        player_obs = obs[player]
        t = player_obs[2]
        t = t.reshape(14, -1)
        # 如果背面的三個 lidar 有任何一條還看的到我方球門，就加分
        if any(lidar[1] > 0 for lidar in t[11:]):
            reward += 1 * ratio
        else:
            reward -= 1 * ratio
        reward_list.append(reward)
    return reward_list

# back_our_goal(obs_convert(obs))

def touch_ball(ball, info, ratio=0.5):
    player_posi = {
        0: info[0]["player_info"]["position"],
        1: info[1]["player_info"]["position"],
        2: info[2]["player_info"]["position"],
        3: info[3]["player_info"]["position"],
    }
    reward_list = [0]*len(info)
    ball_new = info[0]["ball_info"]["position"]
    for player in range(len(info)):
        # 如果有球動了，且原球跟球員距離小於 1.414，就加分
        if np.linalg.norm(ball - ball_new) > 0.1 and np.linalg.norm(ball - player_posi[player]) < 1.414:
            reward_list[player] = 3 * ratio
    return reward_list

# print(touch_ball(ball_before, info))

def close_ball(obs, ratio=0.25):
    reward_list = []
    for player in range(len(obs)):
        reward = 0
        player_obs = obs[player]
        t_1 = player_obs[1]
        t = player_obs[2]

        t_1_ball = t_1[0]
        t_ball = t[0]
        
        # 有接近球就加分
        if t_ball < t_1_ball:
            reward += 1 * ratio
        # 遠離球就扣分
        elif t_ball > t_1_ball:
            reward -= 0.5 * ratio
        reward_list.append(reward)
    return reward_list

# close_ball(obs_convert(obs))

def reward_counter(next_state, ball_before, info, stand_counter, action_dict):
    reward_list = np.zeros(4)

    # 面對球加分 +0.5 * ratio 側對球加分 +0.05 沒面對球扣分 -2 * ratio
    # reward_list = np.sum([reward_list, face_ball(next_state, ratio=0.3)], axis=0)

    # 碰球檢查 +3 * ratio
    # reward_list = np.sum([reward_list, touch_ball(ball_before, info, ratio=0.5)], axis=0)
    # ball_before = info[0]["ball_info"]["position"]
    
    # 靠近球加分 +1 * ratio  遠離球扣分 -0.5 * ratio
    # reward_list = np.sum([reward_list, close_ball(next_state, ratio=0.25)], axis=0)

    # 背對我方球門(面向敵方球門) 加分 +1 * ratio
    # reward_list = np.sum([reward_list, back_our_goal(next_state, ratio=1)], axis=0)

    # punishment
    # 如果有人連續 5 步停在原地，就給 punishment -3
    stand_punish, stand_counter = still_stand(action_dict, stand_counter, stand_step=5, ratio=0.5)
    reward_list = np.sum([reward_list, stand_punish], axis=0)

    return reward_list, ball_before, stand_counter


def train(agent):
    play_time = 0
    env = environment()
    update_timestep = 3
    best_reward = -999
        
    # restart 環境
    env, total_reward, stand_counter, state, ball_before, game_step = restart_env(env)
    play_time = 1

    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }

    total_r = 0


    while True:
        # step_reward
        reward = 0
        action = agent.take_action(state[0].reshape(1,-1))

        action_dict = {
            0: action, 
            1: 0,
            2: 0,
            3: 0,
            }

        next_state, r, done, info = env.step(action_dict)
        if done["__all__"] == False:
            done_num = 0
        else:
            done_num = 1
        # 轉換 observation 的格式
        next_state = obs_convert(next_state)

        reward += r[0]*20
        
        player_to_ball = int((info[0]["ball_info"]["position"][0]-info[0]["player_info"]["position"][0])**2+(info[0]["ball_info"]["position"][1]-info[0]["player_info"]["position"][1])**2)**0.5
        reward += 1/(player_to_ball*50+1)



        total_r += reward
        
        

        # 遊戲的總 reward
        # total_reward = np.sum([total_reward, reward_list], axis=0)

        transition_dict['states'].append(state[0])
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state[0])
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done_num)

        state = next_state

        # 更新梯度
        if play_time % update_timestep == 0:
            agent.learn(transition_dict)
            
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }


            play_time += 1


        if done["__all__"]: # 每1000個畫面就會done
            print("play time:{} reward:{}".format(play_time, total_r))
            logger.report_scalar(title="reward",series="reward", value=total_r, iteration=play_time)
            play_time += 1

            if total_r > best_reward:
                best_reward = total_r
                print("save best agent")
                torch.save(agent.actor.state_dict(), "./pt6/best_reward.pt")
                


            # restart 環境
            env, total_reward, stand_counter, state, ball_before, game_step = restart_env(env)     
            total_r = 0     


def main():
    task = Task.init(project_name="soccer", task_name="phase 1-6")
    logger = Logger.current_logger()

    env = soccer_twos.make(
        render=False,
        flatten_branched=True,
        variation=soccer_twos.EnvType.multiagent_player,
    )
    env.reset()
    print("Observation Space: ", env.observation_space.shape)
    print("Action Space: ", env.action_space)
    obs, reward, done, info = env.step({0: 0, 1: 0, 2: 0, 3: 0})
    print("info", info)
    print("done", done)
    env.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_episodes = 100  # 总迭代次数
    gamma = 0.9  # 折扣因子
    actor_lr = 1e-3  # 策略网络的学习率
    critic_lr = 1e-2  # 价值网络的学习率

        

    agent = ppo.PPO(n_states=294,  # 状态数
                n_hiddens=512,  # 隐含层数
                n_actions=env.action_space.n,  # 动作数
                actor_lr=actor_lr,  # 策略网络学习率
                critic_lr=critic_lr,  # 价值网络学习率
                lmbda = 0.95,  # 优势函数的缩放因子
                epochs = 10,  # 一组序列训练的轮次
                eps = 0.2,  # PPO中截断范围的参数
                gamma=gamma,  # 折扣因子
                device = device
                )

    train(agent)


if __name__ == '__main__':
    main()