from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import gymnasium as gym

# 커스텀 피처 추출기 (이전 질문 참조)
class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_image = torch.zeros(1, 3, 128, 128) # 여기는 원래 이미지 데이터로 (batch, channel, height, width)
            n_flatten = self.cnn(sample_image).shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        image = observations["image"].permute(0, 3, 1, 2).float() / 255.0 # 여긴 입력 이미지의 형태를 잘 보고 결정하자자
        image_features = self.cnn(image)
        sensor_features = self.mlp(observations["sensors"])
        combined = torch.cat([image_features, sensor_features], dim=1)
        return self.linear(combined)
    
# 커스텀 정책 정의
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.features_extractor = CustomFeaturesExtractor(observation_space)