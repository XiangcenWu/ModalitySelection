from RL.Env import Env
from RL.Agent import Agent
from Training import data_spilt, ReadH5d, create_data_loader
from Training import train_seg_net, test_seg_net
from monai.transforms import *
from monai.networks.nets import DynUNet, SwinUNETR
from monai.losses import DiceFocalLoss
import torch
import random






seg_list, rl_list, holdout_list = data_spilt('/raid/candi/xiangcen/miami-data/miama_h5', 825, 400, 100)




device = 'cuda:1'
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
actor_optimizer = torch.optim.AdamW(agent.actor.parameters(), lr=0.0001)
test_result_list = []




for epoch in range(101): # loop over dataset (patients)
    random.shuffle(rl_list)
    _reward = 0
    _step = 0
    for train_dir in rl_list:
        env = Env(train_dir, seg_model)
        if env.all_zero:
            continue
        obs = env.reset()
        for _ in range(step_per_patient):
            action, features = agent.choose_action(obs.unsqueeze(0).to(device), noise=0.3)
            next_obs, reward = env.step(action)
            agent.remember(obs, action, reward)
            obs = next_obs
            
            _reward += reward
            _step += 1
        agent.learn_reinforce(actor_optimizer)


    if (epoch + 1) % 10 == 0:
        agent.save_models(
            f'/raid/candi/xiangcen/trained_models/RLModels/actor{epoch}.ptm'
        )
        print(f'This is epoch {epoch}, average reward w/noise: {_reward/_step}')



