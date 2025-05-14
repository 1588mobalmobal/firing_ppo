from flask import Flask, request, jsonify, send_file, render_template
import os
import segformer_b0 as seg
import path_finding as pf
import firing as fire
from utils import shared_data
from queue import Queue
import math
import numpy as np
from tank_env import TankEnv

app = Flask(__name__)

# Segmentation 모델 선언
seg_model, image_processor = seg.init_model()
# 난수생성기 
rng = np.random.default_rng(seed=13)
# 학습 기록을 위한 카운터 변경
episode_counter = 0
# 연산 인디케이터 
calculating = False
striked_target = None
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
env = TankEnv(max_steps=1000)
is_env_start = False
# 타이밍 동기화를 위한 큐 
data_queue = Queue()

command_to_number = {'Q': 0, 'E' : 1, 'R': 2, 'F': 3, 'FIRE': 4}

class Episode:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, env, now_state):
        self.states = []
        self.actions = []
        self.actions_log_probability = []
        self.values = []
        self.rewards = []
        self.done = False
        self.episode_reward = 0
        self.state, self.info = env.reset(options=now_state) # state는 np.array / info는 딕셔너리



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
    global fired
    global is_env_start
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
    if striked_target:
        fired = False
        striked_target = None
        is_env_start = False
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
    global striked_target
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400
    striked_target = data.get('hit')
    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})


@app.route('/get_action', methods=['GET'])
def get_action():
    global fired
    global ready_to_shot
    # 사격 후 대기
    if fired:
        print('Waiting to observe a bullet')
        return jsonify({"turret": "", "weight": 0.0})
    # 시작하자마자 발포하는 것을 방지하기 위한 버퍼
    if not(ready_to_shot):
        ready_to_shot = True
        print('Set to ready')
        return jsonify({"turret": "", "weight": 0.0})
    # 사격 제원 산출
    data = shared_data.get_data()
    context = fire.Initialize(data)
    turret = fire.TurretControl(context)
    result = turret.normal_control()
    # 학습을 위한 이미지와 정보 산출
    image = seg.get_depth_and_class(seg_model, image_processor)
    x, y, z = data['playerPos']['x'], data['playerPos']['y'], data['playerPos']['z']
    speed, t_x, t_y  = data['playerSpeed'], data['playerTurretX'], data['playerTurretY']
    b_x, b_y, b_z = data['playerBodyX'] ,data['playerBodyY'], data['playerBodyZ']
    data = {'image':image, 'sensor_data':[x,y,z,speed,t_x,t_y,b_x,b_y,b_z]}
    # 큐에 저장
    data_queue.put(data)
    if result == None:
        return jsonify({"turret": "", "weight": 0.0})
    command = {"turret": result[0], "weight": result[1]}
    action = [command_to_number[result[0]], result[1]]
    if command['turret'] == 'FIRE':
        fired = True
        ready_to_shot = False
    print(f"🔫 Action Command: {command}")
    return jsonify(command)

@app.route('/init', methods=['GET'])
def init():
    global rng
    global calculating
    if calculating:
        return jsonify({"status": "Calculating", "message": "Some Calculation are going on..."}), 102
    calculating = True
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
    calculating = False
    # send_reset_message()
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    # print("🚀 /start command received")
    
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5052, debug=True)