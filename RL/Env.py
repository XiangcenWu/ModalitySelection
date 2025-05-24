import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Training.data_loading import readh5
from Training.training_helpers import post_process, dice_coefficient
import torch



class Env():

    def __init__(self, patient: str, seg_model):
        patient = readh5(patient)

        self.t2 = patient['t2']
        self.dwi = patient['dwi']
        self.gt = (patient['lesion'] != 0).float()
        self.shape = self.t2.shape
        
        # segmentation that has been post processed
        self.both_seg = self.get_inference_output(seg_model, torch.cat([self.t2, self.dwi]), post=True)
        self.t2_seg = self.get_inference_output(seg_model, torch.cat([self.t2, torch.zeros_like(self.dwi)]), post=True)
        self.dwi_seg = self.get_inference_output(seg_model, torch.cat([torch.zeros_like(self.dwi), self.dwi]), post=True)
        

        self.current_segmentation = torch.zeros(size=self.shape)
        # self.last_action = torch.zeros(self.shape)
        
        
        self.both_dice, self.t2_dice, self.dwi_dice = self.get_all_accuracy()
        
        
        self.mean_dice = self.get_mean_accuracy()
        self.worst_dice = self.get_worse_accuracy()
        self.best_dice = self.get_best_accuracy()
        


        
        
        self.index_32 = [slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16), slice(16, 20), slice(20, 24), slice(24, 28), slice(28, 32)]


        self.current_accuracy = 0.
        self.last_current_accuracy = 0.
        
        
    @property
    def all_zero(self):
        return all(num == 0 for num in (self.mean_dice, self.worst_dice, self.best_dice))

    def get_inference_output(self, seg_model, input_tensor, post=True):
        with torch.no_grad():
            output = seg_model(input_tensor.unsqueeze(0))
            if post:
                output = post_process(output).squeeze(0)
        return output



    def reset(self):
        return torch.cat([
            self.t2,
            self.dwi,
            self.current_segmentation,
        ])



    def step(self, action):
        self.update_current_seg(action)
        self.current_accuracy = self.calculate_current_accuracy()

        reward = self.current_accuracy - self.last_current_accuracy
        self.last_current_accuracy = self.current_accuracy
        

        obs = torch.cat([
            self.t2,
            self.dwi,
            self.current_segmentation,
        ])
        

        return obs, reward


    def update_current_seg(self, action):
        
        if action <= 7:
            d = self.index_32[action]
            self.current_segmentation[:, :, :, d] = self.t2_seg[:, :, :, d]

        elif action > 7 and action <= 15:
            d = self.index_32[action - 8]
            self.current_segmentation[:, :, :, d] = self.dwi_seg[:, :, :, d]

        elif action > 15 and action <= 23:
            d = self.index_32[action - 16]
            self.current_segmentation[:, :, :, d] = self.both_seg[:, :, :, d]

        else:
            d = self.index_32[action - 24]
            self.current_segmentation[:, :, :, d] = 0.



    def calculate_current_accuracy(self): #
        
        return dice_coefficient(self.current_segmentation, self.gt, post=False).item()
    
    
    def get_all_accuracy(self):
        both = dice_coefficient(self.both_seg, self.gt, post=False).item()
        t2 = dice_coefficient(self.t2_seg, self.gt, post=False).item()
        dwi = dice_coefficient(self.dwi_seg, self.gt, post=False).item()
        
        
        return t2, dwi, both
    
    def get_best_accuracy(self):
        return torch.tensor([self.both_dice, self.t2_dice, self.dwi_dice]).max().item()
    
    def get_worse_accuracy(self):
        return torch.tensor([self.both_dice, self.t2_dice, self.dwi_dice]).min().item()
    
    def get_mean_accuracy(self):
        return torch.tensor([self.both_dice, self.t2_dice, self.dwi_dice]).mean().item()
    

        
    def clear_current_segmentation(self):
        self.current_segmentation = torch.zeros(size=self.shape)

