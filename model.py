import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
Deep Q network for driving
network should take single frame as input
'''
class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        # TODO: Create network
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)    
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lin4 = nn.Linear(4096, 256)

        self.q_scores = nn.Linear(256, action_size)




    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        # TODO: Forward pass through the network
        
        # convert to grayscale and reshape
        batch_size = observation.shape[0]
        rgb_weights = [0.2989, 0.5870, 0.1140]
        observation = np.dot(observation[...,:3], rgb_weights)
        observation = observation.reshape(batch_size, 1, 96, 96) / 255
        # print(f"obs shape: {observation.shape}")
        # print(f"obs: {observation}")
        observation = torch.from_numpy(observation).to(self.device, dtype=torch.float)

        # reshape to bring channels last to first
        # observation = observation.permute(0, 3, 1, 2) 

        conv1 = F.relu(self.conv1(observation))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
       	conv_features = torch.flatten(conv3, start_dim=1) 

        intermediate = F.relu(self.lin4(conv_features))

        return self.q_scores(intermediate)




    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels
        Parameters
        ----------
        observation: list
            python list of batch_size many torch.Tensors of size (96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
