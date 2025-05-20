import os
import math
import numpy as np
import pandas as pd
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
from stable_baselines3.common import utils
import segformer_b0 as seg
import path_finding as pf
import firing as fire

app = Flask(__name__)

# Segmentation 모델 선언
# seg_model, image_processor = seg.init_model()
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
n_steps = 512
batch_size = 128
total_steps = 100000
step_counter = 0
is_bc_collecting = True
target_bc_count = 20000
bc_dataset = []
training = False
# 타이밍 동기화를 위한 스택
data_stack = deque()
training_lock = threading.Lock()
# 사격 버퍼
firing_buffer = 0
prev_command = None
# 이동 관련 변수
final_destination = [0,0]

command_to_number = {'W': 0, 'S' : 1, 'A': 2, 'D': 3}
number_to_command = {0: 'W', 1 : 'S', 2: 'A', 3: 'D'}
# weight_bins = [0.0, 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ]
weight_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]

#####################################################################################################
# 강화학습 관련 클래스 선언
#####################################################################################################
class TankEnv(gym.Env):
    def __init__(self, max_steps = 1000):
        super().__init__() 
        # 연속형 환경 관측
        self.observation_space = Dict({

            "sensor_data": Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)  # 9개의 센서 값
        })
        # 이산형 행동 출력
        self.action_space = MultiDiscrete([4, 11])
        self.steps = 0
        self.max_steps = max_steps
        # self.weight_bins = np.linspace(0.05, 0.5, 10)

        print('Tank Env initialized')
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 시뮬레이터 초기화 및 초기 관측값 반환
        # options를 통해서 각종 자료를 flask 서버에서 넘겨보자
        if options:

            sensor_data = options['sensor_data']  # 더미 센서 데이터
        self.step_count = 0
        print('Environment has been reset')
        return {"sensor_data": sensor_data}, {}
    
    def step(self, action):
        data = data_stack.pop()
        new_data = data['data']
        result = data['result']
        # striked = data['striked_target']

        sensor_data = new_data['sensor_data']

        self.step_count += 1
        reward = 0

        terminated = False
        truncated = False

        if result:
            reward += 10
            terminated = True
        if self.step_count >= self.max_steps:
            terminated = True
            reward -= 1
        info = {}
        return {"sensor_data": sensor_data}, reward, terminated, truncated, info
    
# 커스텀 피처 추출기 (이전 질문 참조)
class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        sensor_features = self.mlp(observations["sensor_data"])
        return self.linear(sensor_features)
    
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
        return self.buf_obs.copy(), infos[0] if infos else {}

    def step_async(self, actions):
        self.step_results = []
        for env_idx, env in enumerate(self.envs):
            # Call env.step() directly, store results
            result = env.step(actions[env_idx])
            self.step_results.append(result)

    def step_wait(self):
        self.buf_obs = {
            
            "sensor_data": np.zeros((self.num_envs, 11), dtype=np.float32)
        }
        rewards, dones, infos = [], [], []
        for i, (obs, rew, terminated, truncated, info) in enumerate(self.step_results):
            done = terminated or truncated
            
            self.buf_obs["sensor_data"][i] = np.copy(obs["sensor_data"])
            rewards.append(rew)
            dones.append(done)
            infos.append(info)
        return self.buf_obs.copy(), np.array(rewards), np.array(dones), infos
    
# PPO 초기화
def initialize_ppo():
    global model, env, rollout_buffer, device
    env = TankEnv(total_steps)
    env = CustomDummyVecEnv([lambda: env])
    model = PPO(
        policy=MultiInputActorCriticPolicy,
        env=env,
        policy_kwargs={"features_extractor_class": CustomFeaturesExtractor},
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        verbose=1,
        device=device,
    )
    # Explicitly set logger
    model._logger = utils.configure_logger(verbose=model.verbose, tensorboard_log=None, tb_log_name="PPO")
    return model, env
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
    global is_env_start
    global is_episode_done
    global training
    global firing_buffer
    global is_bc_collecting
    global final_destination
    if training:
        return jsonify({"status": "success", "control": ""})
    data = request.get_json()
    shared_data.set_data(data)
    # 세그멘테이션, 깊이 2채널 128 * 128 이미지 받아오기
    # 환경이 리셋되어 있지 않으면 리셋을 수행합니다.
    if not(is_env_start):

        # 현재 제원을 받아옵니다.
        x, y, z = data['playerPos']['x'], data['playerPos']['y'], data['playerPos']['z']
        speed, t_x, t_y  = data['playerSpeed'], data['playerTurretX'], data['playerTurretY']
        b_x, b_y, b_z = data['playerBodyX'] ,data['playerBodyY'], data['playerBodyZ']
        d_x, d_z = rng.integers(low=60, high=240, size=2)
        sensor_data_for_reset = np.array([x,y,z,speed,t_x,t_y,b_x,b_y,b_z, d_x, d_z])
        env.reset(options={'sensor_data': sensor_data_for_reset})
        is_env_start = True
        firing_buffer = 0
        dest = nav_controller.set_destination(f'{d_x},10,{d_z}')
        print(f'Initiated Destination: {dest}')


    if len(bc_dataset) == target_bc_count:
        sensors = []
        actions = []
        print('Saving..')
        for i in bc_dataset:
            sensors.append(i['sensor_data'])
            actions.append(i['action'])
        np.savez('sensors.npz', sensors)
        np.savez('actions.npz', actions)
        
    if len(bc_dataset) > target_bc_count:
        is_bc_collecting = False

    # 종료 시 결과를 반영하여 에피소드를 리셋합니다.
    if is_episode_done:
        is_episode_done = False
        nav_controller.completed = False
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
    data = request.get_json()
    try:
        obstacles = data["obstacles"]
        for obstacle in obstacles:
            x_min = float(obstacle["x_min"])
            x_max = float(obstacle["x_max"])
            z_min = float(obstacle["z_min"])
            z_max = float(obstacle["z_max"])
            grid.set_obstacle(x_min, x_max, z_min, z_max)
            obstacles_list.append({
                "x_min": x_min,
                "x_max": x_max,
                "z_min": z_min,
                "z_max": z_max
            })
        # print(f"Obstacles Updated: {obstacles_list}")
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
    global prev_data
    global prev_result
    global step_check
    global step_counter
    global n_steps
    global device
    global is_episode_done
    global bc_dataset
    global training
    global prev_command
    if training:
        return jsonify({"move": "W", "weight": 0.00})
    command = nav_controller.get_move()
    result = command['completed']
    sim_data = shared_data.get_data()
    x, y, z = sim_data['playerPos']['x'], sim_data['playerPos']['y'], sim_data['playerPos']['z']
    speed, t_x, t_y  = sim_data['playerSpeed'], sim_data['playerTurretX'], sim_data['playerTurretY']
    b_x, b_y, b_z = sim_data['playerBodyX'] ,sim_data['playerBodyY'], sim_data['playerBodyZ']
    d_x, d_y = nav_controller.destination[0], nav_controller.destination[1]
    data = {
            'sensor_data': torch.tensor([x,y,z,speed,t_x,t_y,b_x,b_y,b_z, d_x, d_y], dtype=torch.float32).unsqueeze(0).to(device)}
    data_np = {'sensor_data': data['sensor_data'].cpu()}
    data_stack.append({'data':data_np, 'result':result})

    # 행동 전달 및 스텝 단계 이행
    if step_check:
        # 여기서 변화한 상태와 보상 및 종료여부를 가져오지롱
        new_obs, reward, done, info = env.step(prev_result[0])
        model.rollout_buffer.add(
            obs=prev_data,
            action=prev_result[0].cpu(),
            reward=np.array([reward]),
            value=prev_result[1].cpu(),
            log_prob=prev_result[2].cpu(),
            episode_start=np.array([done])
        )
        step_counter += 1

        # 에피소드 리셋
        if step_counter % n_steps == 0 and model.rollout_buffer.pos >= n_steps:
            with training_lock:
                training = True
                print('Training...')
                new_obs_torch = {
                    
                    'sensor_data': torch.tensor(new_obs['sensor_data'], dtype=torch.float32).to(device)

                }
                with torch.no_grad():
                    next_value = model.policy.predict_values(new_obs_torch)
                model.rollout_buffer.compute_returns_and_advantage(last_values=next_value.cpu(), dones=np.array([done]))
                
                model.train()
                print('Training finished...')
                training = False
                model.rollout_buffer.reset()
                model.save("ppo_custom_model")

        if step_counter >= total_steps:
            model.save("ppo_custom_model_final")
            print("Learning completed")
            return {"status": "Learning completed"}, 200
        # 에피소드 리셋
        if done:
            is_episode_done = True

        step_check = False
        print('👍 reward:' , reward)
        return jsonify({"move": "W", "weight": 0.0})
    # 환경 수집 및 행동 출력 수행
    else: # 그럼 여기서 확률적 행동을 산출해야겠지?
        prev_data = data_np
        with torch.no_grad():
            action, value, log_prob = model.policy(data)
        print(f"Log Prob: {log_prob} / destination: {nav_controller.destination} / position: {x:.2f}, {z:.2f}")
        if is_bc_collecting:
            action_1 = command['move']
            action_2 = round((command['weight'] // 0.1) * 0.1, 1)
            action_1_idx = command_to_number[action_1]
            action_2_idx = weight_bins.index(action_2)

            sensor_np = data['sensor_data'].cpu()
            action_np = np.array([action_1_idx, action_2_idx])
            # 리스트 저장
            bc_dataset.append({

                "sensor_data": sensor_np,
                "action": action_np
            }, )
        else:
            action_1 = number_to_command[action.detach().cpu().numpy()[0][0]]
            action_2 = weight_bins[action.detach().cpu().numpy()[0][1]]
        command = {"move": action_1, "weight": action_2} # 규칙 기반 출력 값
        print(f"🚒🚒🚒 Move Command: {command} / bc_data: {len(bc_dataset)}")
        prev_result = [action, value, log_prob]
        step_check = True
        prev_command = action_1
        return jsonify(command)


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
    # global prev_data
    # global prev_result
    # global step_check
    # global striked_target
    # global step_counter
    # global n_steps
    # global device
    # global is_episode_done
    # global bc_dataset
    # global training
    # global firing_buffer
    # global prev_command
    # # 제원 산출
    # if training:
    #     return jsonify({"turret": "Q", "weight": 0.05})
    # data = shared_data.get_data()
    # context = fire.Initialize(data)
    # turret = fire.TurretControl(context)
    # result = turret.normal_control()
    # # 학습을 위한 이미지와 정보 산출
    # image = seg.get_depth_and_class(seg_model, image_processor)
    # x, y, z = data['playerPos']['x'], data['playerPos']['y'], data['playerPos']['z']
    # speed, t_x, t_y  = data['playerSpeed'], data['playerTurretX'], data['playerTurretY']
    # b_x, b_y, b_z = data['playerBodyX'] ,data['playerBodyY'], data['playerBodyZ']
    # # origin_data = {'image':image, 'sensor_data':[x,y,z,speed,t_x,t_y,b_x,b_y,b_z]}
    # data = {'image': torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device),
    #         'sensor_data': torch.tensor([x,y,z,speed,t_x,t_y,b_x,b_y,b_z], dtype=torch.float32).unsqueeze(0).to(device)}
    # data_np = {'image': data['image'].cpu(), 'sensor_data': data['sensor_data'].cpu()}
    # data_stack.append({'data':data_np, 'striked_target':striked_target})
    
    # # 행동 전달 및 스텝 단계 이행
    # if step_check:
    #     # 여기서 변화한 상태와 보상 및 종료여부를 가져오지롱
    #     new_obs, reward, done, info = env.step(prev_result[0])
    #     model.rollout_buffer.add(
    #         obs=prev_data,
    #         action=prev_result[0].cpu(),
    #         reward=np.array([reward]),
    #         value=prev_result[1].cpu(),
    #         log_prob=prev_result[2].cpu(),
    #         episode_start=np.array([done])
    #     )
    #     step_counter += 1

        
    #     # 에피소드 리셋
    #     if step_counter % n_steps == 0 and model.rollout_buffer.pos >= n_steps:
    #         with training_lock:
    #             print('Training...')
    #             training = True
    #             print(f'High: pos: {model.rollout_buffer.pos} / counter: {step_counter} / capa: {model.rollout_buffer.full}')
    #             new_obs_torch = {
    #                 'image': torch.tensor(new_obs['image'], dtype=torch.float32).to(device),
    #                 'sensor_data': torch.tensor(new_obs['sensor_data'], dtype=torch.float32).to(device)

    #             }
    #             with torch.no_grad():
    #                 next_value = model.policy.predict_values(new_obs_torch)
    #             model.rollout_buffer.compute_returns_and_advantage(last_values=next_value.cpu(), dones=np.array([done]))
    #             print(f"obs_image={model.rollout_buffer.observations['image'].shape} / obs_sensor={model.rollout_buffer.observations['sensor_data'].shape}")
    #             model.train()
    #             print('Training finished...')
    #             training = False
    #             model.rollout_buffer.reset()
    #             model.save("ppo_custom_model")
    #             # obs, _ = env.reset()
    #             # with obs_lock:
    #             #     prev_data = obs

    #     if step_counter >= total_steps:
    #         model.save("ppo_custom_model_final")
    #         print("Learning completed")
    #         return {"status": "Learning completed"}, 200
    #     # 에피소드 리셋
    #     if done:
    #         is_episode_done = True

    #     step_check = False
    #     print('👍 reward:' , reward)
    #     return jsonify({"turret": "Q", "weight": 0.00})
    # # 환경 수집 및 행동 출력 수행
    # else: # 그럼 여기서 확률적 행동을 산출해야겠지?
    #     prev_data = data_np
    #     with torch.no_grad():
    #         action, value, log_prob = model.policy(data)
    #     # current_action = {'turret': command_to_number[action[0]], 'weight' : action[1]} # 모델 출력 값
    #     if is_bc_collecting:
    #         action_1 = result[0]
    #         action_2 = round((result[1] // 0.05) * 0.05, 2)
    #         action_1_idx = command_to_number[action_1]
    #         action_2_idx = weight_bins.index(action_2)
    #         image_np = data['image'].cpu()
    #         sensor_np = data['sensor_data'].cpu()
    #         action_np = np.array([action_1_idx, action_2_idx])
    #         # 리스트 저장
    #         bc_dataset.append({
    #             "image": image_np,
    #             "sensor_data": sensor_np,
    #             "action": action_np
    #         }, )
    #     else:
    #         action_1 = number_to_command[action.detach().cpu().numpy()[0][0]]
    #         action_2 = weight_bins[action.detach().cpu().numpy()[0][1]]
    #     command = {"turret": action_1, "weight": action_2} # 규칙 기반 출력 값
    #     print(f"🔫 Action Command: {command} / bc_data: {len(bc_dataset)}")
    #     prev_result = [action, value, log_prob]
    #     step_check = True
    #     prev_command = action_1
    #     return jsonify(command)
    return jsonify({"turret": "Q", "weight": 0.00})


@app.route('/init', methods=['GET'])
def init():
    global rng
    global initiating
    global step_counter
    global final_destination
    global is_env_start
    if initiating:
        return jsonify({"status": "Calculating", "message": "Some Calculation are going on..."}), 102
    initiating = True
    is_env_start = False
    # curriculum = step_counter * 0.01
    while True:
        random_coord = rng.integers(low=60, high=240, size=4)
        x = int(random_coord[0])
        z = int(random_coord[1])
        des_x = int(random_coord[2])
        des_z = int(random_coord[3])
        # des_x = rng.integers(low= x - 30, high= x  + 30, size = 1)[0]
        # des_z = rng.integers(low= z , high= z , size = 1)[0]

        distance = np.sqrt((x - des_x) ** 2 + (z - des_z) ** 2)
        if (distance > 40) and (des_x > 5 and des_x < 295 and des_z > 5 and des_z < 295):
            break

    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": int(x),  #Blue Start Position
        "blStartY": 10,
        "blStartZ": int(z),
        "rdStartX": 200, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 200,
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
    return device


if __name__ == '__main__':
    init_device()
    initialize_ppo()
    app.run(host='0.0.0.0', port=5057, debug=True)

