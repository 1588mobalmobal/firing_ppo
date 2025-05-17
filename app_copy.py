import os
import math
import numpy as np
from collections import deque
import threading
from flask import Flask, request, jsonify, send_file, render_template
from utils import shared_data
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
import segformer_b0 as seg
import path_finding as pf
import firing as fire

app = Flask(__name__)

# Segmentation 모델 선언
seg_model, image_processor = seg.init_model()
# 난수생성기 
rng = np.random.default_rng(seed=13)
# 학습 기록을 위한 카운터 변경
episode_counter = 0
# 연산 인디케이터 
initiating = False
on_step = False
striked_target = None
striked_buffer = 0
# 발포 여부
fired = False
ready_to_shot = False
# 전차 크기 정의 (x: 5미터, z: 11미터)
VEHICLE_WIDTH = int(5.0)
VEHICLE_LENGTH = int(11.0)
# 월드 크기 정의
WORLD_SIZE = 300  # 300x300 미터
# 적 감지 여부
enemy_detected = False
detected_buffer = 0
destination_buffer = 0
enemy_list = []
# 초기화
grid = pf.Grid(width=WORLD_SIZE, height=WORLD_SIZE)
pathfinding = pf.Pathfinding()
nav_config = pf.NavigationConfig()
nav_controller = pf.NavigationController(nav_config, pathfinding, grid)
obstacles_list = []
# 저장 관련 변수 
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)
latest_result = os.path.join(result_dir, "latest_result.png")

# 강화학습 관련 변수
device = None
is_env_start = False
is_episode_done = False
step_check = False
prev_data = None
prev_result = None
env = None
model = None
rollout_buffer = None
n_steps = 256
total_steps = 1024
step_counter = 0
is_bc_collecting = True
bc_dataset = []
training = False
# 타이밍 동기화를 위한 스택
data_stack = deque()
stack_lock = threading.Lock()
action_lock = threading.Lock()
obs_lock = threading.Lock()

command_to_number = {'Q': 0, 'E' : 1, 'R': 2, 'F': 3, 'FIRE': 4}
number_to_command = {0: 'Q', 1 : 'E', 2: 'R', 3: 'F', 4: 'FIRE'}
weight_bins = np.linspace(0.05, 0.5, 10)

#####################################################################################################
# 강화학습 관련 클래스 선언
#####################################################################################################
class TankEnv(gym.Env):
    def __init__(self, max_steps = 1000):
        super().__init__() 
        # 연속형 환경 관측
        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=(128, 128, 2), dtype=np.uint8),  # RGB 이미지
            "sensor_data": Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)  # 9개의 센서 값
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
        if options:
            image = options['image']  # 더미 이미지
            sensor_data = options['sensor_data']  # 더미 센서 데이터
        self.step_count = 0
        print('Environment has been reset')
        return {"image": image, "sensor_data": sensor_data}, {}
    
    def step(self, action):
        data = data_stack.pop()
        new_data = data['data']
        striked = data['striked_target']
        image = new_data['image']
        sensor_data = new_data['sensor_data']
        self.step_count += 1
        reward = 0
        reward -= 0.02 

        terminated = False
        truncated = False

        if striked == 'enemy':
            reward += 10
            terminated = True
        elif striked != None:
            reward -= 1.8
            terminated = True
        if self.step_count >= self.max_steps:
            terminated = True
            reward -= 1
        info = {}
        print('Step finished')
        return {"image": image, "sensor_data": sensor_data}, reward, terminated, truncated, info
    
# 커스텀 피처 추출기 (이전 질문 참조)
class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_image = torch.zeros(1, 2, 128, 128) # 여기는 원래 이미지 데이터로 (batch, channel, height, width)
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
        image = observations["image"].permute(0, 3, 1, 2).float() / 255.0 # 여긴 입력 이미지의 형태를 잘 보고 결정하자
        image_features = self.cnn(image)
        sensor_features = self.mlp(observations["sensor_data"])
        combined = torch.cat([image_features, sensor_features], dim=1)
        return self.linear(combined)
    
# 커스텀 DummyVecEnv
class CustomDummyVecEnv(DummyVecEnv):
    def reset(self, seed=None, options=None):
        # 배치 차원 포함한 버퍼 초기화
        self.buf_obs = {
            key: np.zeros((self.num_envs,) + self.observation_space[key].shape, dtype=self.observation_space[key].dtype)
            for key in self.observation_space.spaces.keys()
        }
        infos = []
        for env_idx, env in enumerate(self.envs):
            obs, info = env.reset(seed=seed, options=options)
            for key in self.buf_obs:
                self.buf_obs[key][env_idx] = obs[key]
            infos.append(info)
        # print(f"Reset buf_obs: image={self.buf_obs['image'].shape}, sensor_data={self.buf_obs['sensor_data'].shape}")
        return self.buf_obs, infos[0] if infos else {}

    def step_async(self, actions):
        self.step_results = []
        for env_idx, env in enumerate(self.envs):
            # Call env.step() directly, store results
            result = env.step(actions[env_idx])
            self.step_results.append(result)

    def step_wait(self):
        self.buf_obs = {
            key: np.zeros((self.num_envs,) + self.observation_space[key].shape, dtype=self.observation_space[key].dtype)
            for key in self.observation_space.spaces.keys()
        }
        rewards, dones, infos = [], [], []
        for i, (obs, rew, terminated, truncated, info) in enumerate(self.step_results):
            done = terminated or truncated
            for key in self.buf_obs:
                self.buf_obs[key][i] = obs[key]
            rewards.append(rew)
            dones.append(done)
            infos.append(info)
        # print(f"Step buf_obs: image={self.buf_obs['image'].shape}, sensor_data={self.buf_obs['sensor_data'].shape}, image_nonzero={np.any(self.buf_obs['image'])}")
        return self.buf_obs, np.array(rewards), np.array(dones), infos
    
# PPO 초기화
def initialize_ppo():
    global model, env, rollout_buffer, device
    env = TankEnv(total_steps)
    env = CustomDummyVecEnv([lambda: env])
    rollout_buffer = DictRolloutBuffer(
        buffer_size=n_steps,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=1,
    )
    model = PPO(
        policy=MultiInputActorCriticPolicy,
        env=env,
        policy_kwargs={"features_extractor_class": CustomFeaturesExtractor},
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        verbose=1,
    )
#####################################################################################################

#####################################################################################################
# 각도 변환용
def change_degree(my_d):
    if my_d > 180:
        direction = -(360-my_d)
    else:
        direction = my_d
    return direction

# 상대좌표
def get_target_coord(now_x, now_y, turret_x, distance):
    rad = math.radians(turret_x)
    enemy_x = math.sin(rad) * distance + now_x
    enemy_y = math.cos(rad) * distance + now_y
    return enemy_x, enemy_y

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)
    filtered_results = []
    return (filtered_results), 200

@app.route('/latest_result')
def get_latest_result():
    if os.path.exists(latest_result):
        return send_file(latest_result, mimetype='image/png')
    else:
        return jsonify({"error": "No result available"}), 404


# Flask 라우팅
@app.route('/info', methods=['POST'])
def info():
    global striked_target
    global striked_buffer
    global is_env_start
    global is_episode_done
    global training
    if training:
        return jsonify({"status": "success", "control": ""})
    data = request.get_json()
    shared_data.set_data(data)
    # 세그멘테이션, 깊이 2채널 128 * 128 이미지 받아오기
    # 환경이 리셋되어 있지 않으면 리셋을 수행합니다.
    if not(is_env_start):
        image = seg.get_depth_and_class(seg_model, image_processor)
        # 현재 전차 제원을 받아옵니다.
        x, y, z = data['playerPos']['x'], data['playerPos']['y'], data['playerPos']['z']
        speed, t_x, t_y  = data['playerSpeed'], data['playerTurretX'], data['playerTurretY']
        b_x, b_y, b_z = data['playerBodyX'] ,data['playerBodyY'], data['playerBodyZ']
        env.reset(options={'image':image, 'sensor_data':[x,y,z,speed,t_x,t_y,b_x,b_y,b_z]})
        is_env_start = True

    # 관측된 포탄 낙하 결과를 반영하여 에피소드를 리셋합니다.
    if is_episode_done:
        is_episode_done = False
        striked_target = None
        is_env_start = False
        data_stack.clear()
        return jsonify({"status": "success", "control": "reset"})
    return jsonify({"status": "success", "control": ""})


@app.route('/update_position', methods=['POST'])
def update_position():
    data = request.get_json()
    if not data or "position" not in data:
        return jsonify({"status": "ERROR", "message": "위치 데이터 누락"}), 400
    result = nav_controller.update_position(data["position"])
    if result["status"] == "ERROR":
        return jsonify(result), 400
    return jsonify(result)

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles_list
    # data = request.get_json()
    try:
        # obstacles = data["obstacles"]
        # for obstacle in obstacles:
        #     x_min = float(obstacle["x_min"])
        #     x_max = float(obstacle["x_max"])
        #     z_min = float(obstacle["z_min"])
        #     z_max = float(obstacle["z_max"])
        #     grid.set_obstacle(x_min, x_max, z_min, z_max)
        #     obstacles_list.append({
        #         "x_min": x_min,
        #         "x_max": x_max,
        #         "z_min": z_min,
        #         "z_max": z_max
        #     })
        # # print(f"Obstacles Updated: {obstacles_list}")
        return jsonify({"status": "OK"})
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error in /update_obstacle: {e}")
        return jsonify({"status": "ERROR", "message": "Invalid obstacle data"}), 400

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    # if not data or "destination" not in data:
    #     return jsonify({"status": "ERROR", "message": "목적지 데이터 누락"}), 400
    # result = nav_controller.set_destination(data["destination"])
    # print('Destination:' , data["destination"], type(data["destination"]))
    # if result["status"] == "ERROR":
    #     return jsonify(result), 400
    return jsonify(data['destination'])

@app.route('/get_move', methods=['GET'])
def get_move():
    # global enemy_detected
    # global enemy_list
    # global destination_buffer
    # if enemy_detected:
    #     data = shared_data.get_data()
    #     if enemy_list == None:
    #         print('Stop the tank')
    #         return jsonify({"move": "STOP"})
    #     enemies = len(enemy_list)
    #     if enemies == 1:
    #         # 사정거리 안에 있으면 그 자리에서 멈춰서 쏘자
    #         distance = enemy_list[0]['distance']
    #         if distance < 105:
    #             print('Stop the tank')
    #             return jsonify({"move": "STOP"})
    #         else:
    #             x = data['playerPos']['x']
    #             y = data['playerPos']['y']
    #             z = data['playerPos']['z']
    #             turret_x = data['playerTurretX']
    #             enemy_x, enemy_z = get_target_coord(x, z, turret_x, distance)
    #             if destination_buffer == 0:
    #                 nav_controller.set_destination(f'{enemy_x},{y},{enemy_z}')
    #                 print(f'Destination has been changed: {enemy_x},{y},{enemy_z}')
    #                 destination_buffer += 1
    #             else:
    #                 destination_buffer += 1
    #                 if destination_buffer > 16:
    #                     destination_buffer = 0
    #             command = nav_controller.get_move()
    #             print(f'Moving Command: {command}')
    #             return jsonify(command)
    #     else:
    #         target_id = 0
    #         target_distance = 1000
    #         for i, enemy in enumerate(enemy_list):
    #             if enemy.get['distance'] < target_distance:
    #                 target_id = i
    #                     # 사정거리 안에 있으면 그 자리에서 멈춰서 쏘자
    #         distance = enemy_list[target_id]['distance']
    #         if distance < 100:
    #             print('Stop the tank')
    #             return jsonify({"move": "STOP"})
    #         else:
    #             x = data['playerPos']['x']
    #             y = data['playerPos']['y']
    #             z = data['playerPos']['z']
    #             turret_x = data['playerTurretX']
    #             enemy_x, enemy_z = get_target_coord(x, z, turret_x, distance)
    #             if destination_buffer == 0:
    #                 nav_controller.set_destination(f'{enemy_x},{y},{enemy_z}')
    #                 print(f'Destination has been changed: {enemy_x},{y},{enemy_z}')
    #                 destination_buffer += 1
    #             else:
    #                 destination_buffer += 1
    #                 if destination_buffer > 16:
    #                     destination_buffer = 0
    #             command = nav_controller.get_move()
    #             print(f'Moving Command: {command}')
    #             return jsonify(command)
    # else:
    #     command = nav_controller.get_move()
    #     print(f'Moving Command: {command}')
        return jsonify({"move":"STOP"})

@app.route('/visualization', methods=['GET'])
def get_visualization():
    try:
        return send_file("path_visualization.html")
    except FileNotFoundError:
        return jsonify({"status": "ERROR", "message": "Visualization file not found. Please set a destination first."}), 404

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    if training:
        return jsonify({"status": "OK", "message": "Bullet impact data received"})
    global striked_target
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400
    striked_target = data.get('hit')
    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/get_action', methods=['GET'])
def get_action():
    global prev_data
    global prev_result
    global step_check
    global striked_target
    global step_counter
    global n_steps
    global device
    global is_episode_done
    global bc_dataset
    global training
    # 사격 제원 산출
    if training:
        return jsonify({"turret": "Q", "weight": 0.0})
    data = shared_data.get_data()
    context = fire.Initialize(data)
    turret = fire.TurretControl(context)
    result = turret.normal_control()
    # 학습을 위한 이미지와 정보 산출
    image = seg.get_depth_and_class(seg_model, image_processor)
    x, y, z = data['playerPos']['x'], data['playerPos']['y'], data['playerPos']['z']
    speed, t_x, t_y  = data['playerSpeed'], data['playerTurretX'], data['playerTurretY']
    b_x, b_y, b_z = data['playerBodyX'] ,data['playerBodyY'], data['playerBodyZ']
    origin_data = {'image':image, 'sensor_data':[x,y,z,speed,t_x,t_y,b_x,b_y,b_z]}
    data = {'image': torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device),
            'sensor_data': torch.tensor([x,y,z,speed,t_x,t_y,b_x,b_y,b_z], dtype=torch.float32).unsqueeze(0).to(device)}
    data_stack.append({'data':origin_data, 'striked_target':striked_target})
    
    # 행동 전달 및 스텝 단계 이행
    if step_check:
        # 여기서 변화한 상태와 보상 및 종료여부를 가져오지롱
        new_obs, reward, done, info = env.step(prev_result[0])
        print(new_obs, type(new_obs))
        rollout_buffer.add(
            obs=prev_data,
            action=prev_result[0].cpu(),
            reward=np.array([reward]),
            # done=np.array([done]),
            value=prev_result[1].cpu(),
            log_prob=prev_result[2].cpu(),
            episode_start=np.array([False])
        )
        step_counter += 1

        # if step_counter % n_steps == 0:
        #     with torch.no_grad():
        #         next_value = model.policy.predict_values(new_obs)
        #     rollout_buffer.compute_returns_and_advantage(last_values=next_value, dones=np.array([done]))
        #     print('Training...')
        #     training = True
        #     model.train()
        #     print('Training finished...')
        #     training = False
        #     rollout_buffer.reset()
        
        # 학습 종료
        if step_counter >= total_steps:
            done = True
        
        # 에피소드 리셋
        if done:
            with torch.no_grad():
                next_value = model.policy.predict_values(new_obs)
            rollout_buffer.compute_returns_and_advantage(last_values=next_value, dones=np.array([done]))
            print('Training...')
            training = True
            model.train()
            print('Training finished...')
            training = False
            rollout_buffer.reset()
            is_episode_done = True
            model.save("ppo_custom_model")
            # obs, _ = env.reset()
            # with obs_lock:
            #     prev_data = obs
        step_check = False
        print('👍 reward:' , reward)
        return jsonify({"turret": "Q", "weight": 0.0})
    # 환경 수집 및 행동 출력 수행
    else: # 그럼 여기서 확률적 행동을 산출해야겠지?
        prev_data = origin_data
        with torch.no_grad():
            action, value, log_prob = model.policy(data)
        print(f'Action: {action}, value {value}, log_prob {log_prob}')
        # print(action.detach().cpu().numpy(), type(action.detach().cpu().numpy()))
        # current_action = {'turret': command_to_number[action[0]], 'weight' : action[1]} # 모델 출력 값
        if is_bc_collecting:
            print(result)
            action_1 = result[0]
            action_2 = (result[1] // 0.05) * 0.05
            bc_dataset.append({"obs": data, "action": result})
        else:
            action_1 = number_to_command[action.detach().cpu().numpy()[0][0]]
            action_2 = weight_bins[action.detach().cpu().numpy()[0][0]]
        command = {"turret": action_1, "weight": action_2} # 규칙 기반 출력 값
        print(f"🔫 Action Command: {command}")
        prev_result = [action, value, log_prob]
        step_check = True
        return jsonify(command)

@app.route('/init', methods=['GET'])
def init():
    global rng
    global initiating
    if initiating:
        return jsonify({"status": "Calculating", "message": "Some Calculation are going on..."}), 102
    initiating = True
    curriculum = 40
    while True:
        random_coord = rng.integers(low=60, high=240, size=4)
        x = int(random_coord[0])
        z = int(random_coord[1])
        
        des_x = rng.integers(low= x - curriculum + 40, high= x + curriculum + 40, size = 1)[0]
        des_z = rng.integers(low= z - curriculum + 40, high= z + curriculum + 40, size = 1)[0]
        
        distance = np.sqrt((x - des_x) ** 2 + (z - des_z) ** 2)
        if (distance > 40) and (des_x > 5 and des_x < 295 and des_z > 5 and des_z < 295):
            break
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": int(x),  #Blue Start Position
        "blStartY": 10,
        "blStartZ": int(z),
        "rdStartX": int(des_x), #Red Start Position
        "rdStartY": 10,
        "rdStartZ": int(des_z),
        "trackingMode": True,
        "detactMode": False,
        "logMode": True,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveStereoCamera": True,
        "saveLog": False,
        "saveLidarData": False
    }
    print("🛠️ Initialization config sent via /init:", config["blStartX"], config["blStartZ"], config["rdStartX"], config["rdStartZ"])
    initiating = False
    # send_reset_message()
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():

    return jsonify({"control": ""})
    
def init_device():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    init_device()
    initialize_ppo()
    app.run(host='0.0.0.0', port=5055, debug=True)

