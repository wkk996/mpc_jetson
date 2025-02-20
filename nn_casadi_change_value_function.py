import torch
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler


class CostPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super(CostPredictor, self).__init__()
        self.fc1 = torch.nn.Linear(13, 32)  
        self.fc2 = torch.nn.Linear(32, 32)  
        self.fc3 = torch.nn.Linear(32, 32)  
        self.fc4 = torch.nn.Linear(32, 1)   

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  
        x = torch.tanh(self.fc2(x))  
        x = torch.tanh(self.fc3(x))  
        x = self.fc4(x)              
        return x


def load_model(model_path='SANMPC_Neural_Model/neural_value_function_model.pth'):
    model = CostPredictor(13)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  
    return model


def get_model_parameters(model):
    state_dict = model.state_dict()
    
    W1 = state_dict['fc1.weight'].detach().numpy()
    b1 = state_dict['fc1.bias'].detach().numpy()
    W2 = state_dict['fc2.weight'].detach().numpy()
    b2 = state_dict['fc2.bias'].detach().numpy()
    W3 = state_dict['fc3.weight'].detach().numpy()
    b3 = state_dict['fc3.bias'].detach().numpy()
    W4 = state_dict['fc4.weight'].detach().numpy()
    b4 = state_dict['fc4.bias'].detach().numpy()
    
    return W1, b1, W2, b2, W3, b3, W4, b4



def create_casadi_nn(input_dim, W1, b1, W2, b2, W3, b3, W4, b4, state_scaler, cost_scaler):

    x = ca.MX.sym('x', 1, 13) 
    

    x_normalized = (x - state_scaler.mean_.reshape(1, -1)) / state_scaler.scale_.reshape(1, -1)



    W1_T = W1.T
    W2_T = W2.T
    W3_T = W3.T
    W4_T = W4.T
    

    W1_casadi = ca.MX(W1_T)
    b1_casadi = ca.MX(b1).T
    layer1 = ca.tanh(ca.mtimes(x_normalized, W1_casadi) + b1_casadi)


    W2_casadi = ca.MX(W2_T)
    b2_casadi = ca.MX(b2).T
    layer2 = ca.tanh(ca.mtimes(layer1, W2_casadi) + b2_casadi)

    W3_casadi = ca.MX(W3_T)
    b3_casadi = ca.MX(b3).T
    layer3 = ca.tanh(ca.mtimes(layer2, W3_casadi) + b3_casadi)


    W4_casadi = ca.MX(W4_T)
    b4_casadi = ca.MX(b4).T
    output = ca.mtimes(layer3, W4_casadi) + b4_casadi


    output_normalized = output * cost_scaler.scale_.reshape(1, -1) + cost_scaler.mean_.reshape(1, -1)


    return x, output_normalized




def pytorch_to_casadi(model_path='SANMPC_Neural_Model/neural_value_function_model.pth',state_scaler=None, cost_scaler=None):

    model = load_model(model_path)
    scalers = torch.load('SANMPC_Neural_Model/scalers_value_fuction.pth')
    state_scaler = scalers['state_scaler']
    cost_scaler = scalers['cost_scaler']



    W1, b1, W2, b2, W3, b3, W4, b4 = get_model_parameters(model)
    

    x_casadi, output_casadi = create_casadi_nn(3, W1, b1, W2, b2, W3, b3, W4, b4, state_scaler,cost_scaler)
    
    nn_function = ca.Function('nn_function', [x_casadi], [output_casadi])
    
    return nn_function




def pytorch_nn(state_input, model_path='SANMPC_Neural_Model/neural_value_function_model.pth', cost_scaler=None):
    model = load_model(model_path)
    scalers = torch.load('SANMPC_Neural_Model/scalers_value_fuction.pth')
    state_scaler = scalers['state_scaler']
    cost_scaler = scalers['cost_scaler']

    state_input = state_input.reshape(1, -1) 
    state_input_scaled = state_scaler.transform(state_input)
    
 
    state_tensor = torch.FloatTensor(state_input_scaled)
    with torch.no_grad():
        output = model(state_tensor)
    
 
    if cost_scaler:
        output_reshaped = output.numpy().reshape(-1, 1)  
        output = cost_scaler.inverse_transform(output_reshaped)
    
    return output



def casadi_nn(state_input, model_path='SANMPC_Neural_Model/neural_value_function_model.pth', cost_scaler=None):
    nn_function = pytorch_to_casadi(model_path, cost_scaler)
    output_value = nn_function(state_input) 
    return output_value




def test_nn_models():
    model_path = 'SANMPC_Neural_Model/neural_value_function_model.pth'
    state_input =  np.array([
        0,   
        0,   
        0,  
        1,   
        0,   
        0,   
        0,  
        0,   
        0,   
        0,  
        0,   
        0,   
    ], dtype=np.float32)




    pytorch_output = pytorch_nn(state_input, model_path)
    





    print(f"PyTorch NN Prediction: {pytorch_output}")



if __name__ == "__main__":
    test_nn_models()