import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """

    """ loss function is:
        Expectation[ (reward + gamma * max_a'(Q(s', s'; theta_i)) - Q(s, a; theta) )^2 ]
    """

    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
    # print("act_batch : ", act_batch)
    q_values = policy_net.forward(obs_batch)
    
    # ref: https://discuss.pytorch.org/t/selecting-element-on-dimension-from-list-of-indexes/36319
    q_values = q_values[torch.arange(q_values.size(0)), torch.from_numpy(act_batch)].unsqueeze(1)
    
    # print("q_val: ", q_values)
    #max_target_q = target_net(next_obs_batch).max(1)[0].detach() # detach is also like making a copy
    
    # double q learning 
    target_q =  target_net(next_obs_batch)
    policy_net_best_id = policy_net(next_obs_batch).max(1)[1].detach()
    max_target_q = target_q[torch.arange(target_q.size(0)), policy_net_best_id]
    # print("max target: ", max_target_q) # [N, 1]
    # print("max target shape: ", max_target_q.shape) # [N, 1]
    
    '''
    print("max target: ", max_target_q.shape) [N, 1]
    print("gamma: ",gamma) 
    print("rew: ", rew_batch.shape) (N) numpy
    print("mask: ", torch.from_numpy(done_mask==0).shape) [N]
    print("q_val: ", q_values) [N, 1]
    ''' 
    
    # mask done episodes
    max_target_q *= torch.from_numpy(done_mask==0).to(device)
    q_target = torch.from_numpy(rew_batch).to(device) + gamma*max_target_q

    # calculate loss
    loss = F.smooth_l1_loss(q_values, q_target.unsqueeze(1)) 
    optimizer.zero_grad()
    loss.backward()
    
    # clip the gradients, update model
    torch.nn.utils.clip_grad_value_(policy_net.parameters(),1)
    optimizer.step()
    return loss

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network
    target_net.load_state_dict(policy_net.state_dict())
