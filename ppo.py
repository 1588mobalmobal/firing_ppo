from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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
            sample_image = torch.zeros(1, 3, 84, 84)
            n_flatten = self.cnn(sample_image).shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        image = observations["image"].permute(0, 3, 1, 2).float() / 255.0
        image_features = self.cnn(image)
        sensor_features = self.mlp(observations["sensors"])
        combined = torch.cat([image_features, sensor_features], dim=1)
        return self.linear(combined)

# 환경 및 모델
env = CustomEnv()
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = PPO(
    policy="MultiInputActorCriticPolicy",
    env=env,
    policy_kwargs={"features_extractor_class": CustomFeaturesExtractor},
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=1,
    tensorboard_log="./tb_logs/",
)

# 학습
eval_env = CustomEnv()
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
eval_callback = EvalCallback(eval_env, best_model_save_path="./best_model/", eval_freq=10000)
model.learn(total_timesteps=100000, callback=eval_callback)
model.save("ppo_multidiscrete")