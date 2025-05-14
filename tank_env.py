import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete
import numpy as np
import cv2
import time

class TankEnv(gym.Env):
    def __init__(self, max_steps = 1000):
        super().__init__() 
        # 연속형 환경 관측
        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=(128, 128, 2), dtype=np.uint8),  # RGB 이미지
            "sensors": Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)  # 4개의 센서 값
        })
        # 이산형 행동 출력
        self.action_space = MultiDiscrete([5, 10])
        self.steps = 0
        self.max_steps = max_steps
        self.weight_bins = np.linspace(0.05, 0.5, 10)

        print('Tank Env initialized')
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 시뮬레이터 초기화 및 초기 관측값 반환
        # options를 통해서 각종 자료를 flask 서버에서 넘겨보자 
        image = options['image']  # 더미 이미지
        sensors = options['sensor_data']  # 더미 센서 데이터
        self.step_count = 0
        print('Environment has been reset')
        return {"image": image, "sensors": sensors}, {}
    
    def step(self, action, option=None):
        new_data = data_stack.pop()
        striked = option
        image = new_data['image']
        sensors = new_data['sensor_data']
        self.step_count += 1
        reward = 0
        reward -= 0.02 
        if striked == 'enemy':
            reward += 10
            terminated = True
        elif striked != None:
            reward -= 1.8
            terminated = True
        if self.step_count >= self.max_steps:
            truncated = True
            reward -= 1
        info = {}
        print('Step finished')
        return {"image": image, "sensor_data": sensors}, reward, terminated, truncated, info
