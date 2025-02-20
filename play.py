import pickle
import numpy as np
from quadrotor import Quadrotor
from controller_nn import Controller_nn
from parameters import get_reference_fixed, N, DT
import psutil
import time

def load_state(file_name):
    with open(file_name, 'rb') as f:
        file_data = pickle.load(f)
    return file_data

def use_state(file_name):

    file_data = load_state(file_name)


    total_exec_time = 0  
    total_cpu_usage = 0  
    total_psutil_calls = 0  

    num_samples = len(file_data)

    for sample_idx, sample in enumerate(file_data):
        current_state = sample['state']

        quad = Quadrotor()
        quad.pos = current_state[0:3]
        quad.quat = current_state[3:7]
        quad.vel = current_state[7:10]
        quad.a_rate = current_state[10:13]

        controller = Controller_nn(quad, obstacles=None, n_nodes=N, use_nn=True, 
                                   model_path='SANMPC_Neural_Model/neural_value_function_model.pth',
                                   use_sensitivity=True, model_path_sensitivity='SANMPC_Neural_Model/neural_sensitivity_model.pth',
                                   dt=DT, gamma=0.3)

        x_ref = get_reference_fixed(
            time=0.0,
            x0=current_state,
            n=N,
            dt=DT,
            target=np.array([10.0, 10.0, 10.0])
        )

        cpu_before = psutil.cpu_percent(interval=0.01)  
        start_time = time.time()  

        control_input, cost_over_30, cost_over_last_25, state_step_5 = controller.compute_control_signal(x_ref)
        
        end_time = time.time()  
        cpu_after = psutil.cpu_percent(interval=0.01)  

    
        exec_time = end_time - start_time
        cpu_usage = cpu_after - cpu_before

        total_exec_time += exec_time

        if cpu_usage >= 0:
            total_cpu_usage += cpu_usage
            total_psutil_calls += 1  

        print(f"Sample {sample_idx + 1}: Execution Time = {exec_time:.4f}s")

    avg_exec_time = total_exec_time / num_samples if num_samples > 0 else 0
    print(f"\nAverage Execution Time for {num_samples} samples: {avg_exec_time:.4f} s")
    avg_cpu = total_cpu_usage / total_psutil_calls if total_psutil_calls > 0 else 0
    print(f"CPU:{avg_cpu:.4f} %")



if __name__ == "__main__":
    file_name = "data_test_jetson.txt"  
    use_state(file_name)
