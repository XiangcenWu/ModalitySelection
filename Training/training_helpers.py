import torch
import random
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def post_process(x: torch.tensor, th: int=0.5):
    x = torch.sigmoid(x)
    return (x > th).to(float)

def dice_coefficient(prediction: torch.tensor, gt: torch.tensor, post=False):
    assert prediction.shape == gt.shape, "Masks must be the same shape"
    if post:
        prediction = post_process(prediction)
    return (2 * torch.sum(prediction * gt)) / (torch.sum(prediction) + torch.sum(gt) + 1e-6)


# def dice_coef_batch(prediction: torch.tensor, gt: torch.tensor):
#     batch_size = prediction.shape[0]
#     assert prediction.shape == gt.shape, "Masks must be the same shape"
#     _tensor = torch.zeros(size=(batch_size, 1))
#     for i in range(batch_size):
#         _tensor[i, 0] = dice_coefficient(prediction[i], gt[i], post=True)
#     return _tensor


def get_t2(batch):
    t2_img = batch['t2']
    return torch.cat([t2_img, torch.zeros_like(t2_img)], dim=1)


def get_dwi(batch):
    dwi_img = batch['dwi']
    return torch.cat([torch.zeros_like(dwi_img), dwi_img], dim=1)


def get_both(batch):
    t2_img = batch['t2']
    dwi_img = batch['dwi']
    return torch.cat([t2_img, dwi_img], dim=1)
    
functions = [get_both, get_dwi, get_t2]
def train_seg_net(
        seg_model, 
        seg_loader,
        seg_optimizer,
        seg_loss_function,
        device="cuda:0"
    ):
    
    seg_model.train()
    seg_model.to(device)

    step = 0.
    loss_a = 0.
    for batch in seg_loader:
        
        label = batch['lesion'].to(device)
        selected_function = random.choice(functions)
        img = selected_function(batch).to(device)
        

        output = seg_model(img)
        loss = seg_loss_function(output, label)
        loss.backward()
        seg_optimizer.step()
        seg_optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch



def test_seg_net(
        seg_model,
        inference_loader,
        device='cuda:0'
    ):
    seg_model.to(device)
    seg_model.eval()  # Set to evaluation mode
    
    

    dice_t2 = []
    dice_dwi = []
    dice_both = []
    
    for batch in inference_loader:
        label = batch['lesion'].to(device)
        t2 = get_t2(batch).to(device)
        dwi = get_dwi(batch).to(device)
        both = get_both(batch).to(device)
        with torch.no_grad():
            
            output = seg_model(t2)
            dice_t2.append(dice_coefficient(output, label, post=True).item())

            output = seg_model(dwi)
            dice_dwi.append(dice_coefficient(output, label, post=True).item())
            
            output = seg_model(both)
            dice_both.append(dice_coefficient(output, label, post=True).item())



    return torch.tensor(dice_t2) , torch.tensor(dice_dwi), torch.tensor(dice_both)


