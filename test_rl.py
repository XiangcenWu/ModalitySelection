from RL.Env import Env
from RL.Agent import Agent
from Training import data_spilt
from monai.transforms import *
from monai.networks.nets import SwinUNETR
import torch



seg_list, rl_list, holdout_list = data_spilt('/raid/candi/xiangcen/miami-data/miama_h5', 925, 300, 100)


device = 'cuda:0'
seg_model = SwinUNETR(
    img_size = (128, 128, 32),
    in_channels = 2,
    out_channels = 1,
    depths = (2, 2, 2, 2),
    num_heads = (3, 6, 12, 24),
    drop_rate = 0.,
    attn_drop_rate = 0.,
    dropout_path_rate = 0.,
    downsample="mergingv2",
    use_v2=True,
)
seg_model.load_state_dict(torch.load("/raid/candi/xiangcen/trained_models/SegModels/swinunetr.ptm", map_location=device, weights_only=True))
seg_model.eval()



step_per_patient = 60
step_to_train = 40
batch_size = 10
n_epochs = 2
agent = Agent(gamma = 0.5, batch_size=batch_size, n_epochs=n_epochs, step_to_train=step_to_train, device=device)
# agent.load_models("/raid/candi/xiangcen/trained_models/RLModels/actor29.ptm") # 29
agent.load_models("/raid/candi/xiangcen/trained_models/RLModels/actor_reinforce_59.ptm") # 29






_result = []
_index = []

for patient_dir in holdout_list:
    test_result_list = []
    agent.memory.clear_memory()
    env = Env(patient_dir, seg_model)
    obs = env.reset()
    
    for _ in range(20):
        action, features = agent.choose_action(obs.unsqueeze(0).to(device), noise=0.)
        next_obs, reward = env.step(action)
        test_result_list.append(env.calculate_current_accuracy())
        obs = next_obs
        
    max_num = max(test_result_list)  # Finds the maximum value
    max_index = test_result_list.index(max_num) 
    _result.append(max_num)
    _index.append(max_index)
    # print(test_result_list)
    

    
print(torch.tensor(_result).mean(), torch.tensor(_result).std(), torch.tensor(_index).float().mean())

    
    
    
print("navie agent")

_result = []

for patient_dir in holdout_list:
    test_result_list = []
    agent.memory.clear_memory()
    env = Env(patient_dir, seg_model)
    obs = env.reset()
    
    for _ in range(8):
        x = torch.randint(0, 4, size=(1, )).item()
        action = x * 8 + _
        next_obs, reward = env.step(action)
        test_result_list.append(env.calculate_current_accuracy())
        obs = next_obs
        
    max_num = max(test_result_list)  # Finds the maximum value
    _result.append(max_num)

    

    
print(torch.tensor(_result).mean(), torch.tensor(_result).std())

