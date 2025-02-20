import torch
import torch.nn as nn
import numpy as np
import pickle
import time
from quadrotor import Quadrotor   
from parameters import *          


class AMPCPredictor(nn.Module):

    def __init__(self, input_dim):
        super(AMPCPredictor, self).__init__()
        self.fc1 = nn.Linear(13, 100)   
        self.fc2 = nn.Linear(100, 100) 
        self.fc3 = nn.Linear(100, 4)    

    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01) 
        x = nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)  
        x = self.fc3(x)  
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AMPCPredictor(13).to(device)
model.load_state_dict(torch.load('AMPC/ampc_control.pth', map_location=device))
model.eval()  


scaler_data = torch.load('AMPC/scalers_ampc.pth', map_location=device)
state_scaler = scaler_data['state_scaler_ampc']
cost_scaler = scaler_data['cost_scaler_ampc']


def ampc_controller(state):

    state_scaled = state_scaler.transform(state.reshape(1, -1))
 
    state_tensor = torch.FloatTensor(state_scaled).to(device)

    with torch.no_grad():
        control_scaled = model(state_tensor)
    control_scaled_np = control_scaled.cpu().numpy()

    control = cost_scaler.inverse_transform(control_scaled_np)
    return control.flatten() 



if __name__ == "__main__":
 
    obstacles = [
        (np.array([4.0, 4.0, 7.0]), 2.0), 
        (np.array([8.0, 8.0, 2.0]), 2.0), 
        (np.array([6.0, 8.0, 3.0]), 2.0), 
        (np.array([4.0, 4.0, 1.0]), 2.0), 
        (np.array([8.0, 2.0, 2.0]), 2.0), 
        (np.array([2.0, 8.0, 5.0]), 2.0)
    ]
    

    quad = Quadrotor()


    path = []
    times = []
    inside_obstacle_points = [] 

    cur_time = 0
    total_time = 2.3


    target_point = np.array([10.0, 10.0, 10.0])
    tolerance = 0.5 

    while cur_time < total_time:
    

        state = np.concatenate(quad.get_state())
        

        start = time.time()

        thrust = ampc_controller(state)
        times.append(time.time() - start)
        

        print(f"Time: {cur_time:.2f}s")
        pos, quat, vel, a_rate = quad.get_state()
        print("States:")
        print(f"  Position: {pos}")
        print(f"  Quaternion: {quat}")
        print(f"  Velocity: {vel}")
        print(f"  Angular Rate: {a_rate}")
        print("Control Inputs (Thrusts):", thrust)
        print("-" * 50)
        

        quad.update(thrust, dt=DT)
        path.append(quad.pos)

        for center, radius in obstacles:
            if np.linalg.norm(quad.pos - center) <= radius+0.5:
                inside_obstacle_points.append(quad.pos.copy())
                break  

        cur_time += DT
        

        distance_to_target = np.linalg.norm(quad.pos - target_point)
        if distance_to_target <= tolerance:
            print(f" (: {distance_to_target:.4f} )")
            print(f": {cur_time:.2f} ")
            break


    if inside_obstacle_points:
        print("\n")
        for point in inside_obstacle_points:
            print(point)
    else:
        print("\n")
    
    # 保存仿真数据，包括路径、计算时间以及进入障碍物内部的状态点
    with open(FILE_NAME, 'wb') as file:
        path_array = np.array(path)
        times_array = np.array(times)
        data = {
            'path': path_array,
            'times': times_array,
            'inside_obstacle_points': inside_obstacle_points 
        }
        print("Max processing time: {:.4f}s".format(times_array.max()))
        print("Min processing time: {:.4f}s".format(times_array.min()))
        print("Mean processing time: {:.4f}s".format(times_array.mean()))
        pickle.dump(data, file)
