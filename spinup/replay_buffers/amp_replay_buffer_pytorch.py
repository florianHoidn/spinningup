import numpy as np
import gym
import torch
import torch.nn as nn
import os
from torch.optim import Adam

#TODO rename AMP to something more general like AdversarialImitation or so.
class AMPReplayBuffer:
    """
    A replay buffer that dynamically adds an imitation bonus to the rewards from the environment.
    The imitation bonus is based on adversarial motion priors (AMP) as described in
    https://xbpeng.github.io/projects/AMP/2021_TOG_AMP.pdf and implemted in UC Berkeley's DeepMimic
    project, https://github.com/xbpeng/DeepMimic.
    """

    # TODO it might be a good idea to implement this as an evironment wrapper instead of a replay buffer wrapper (after all, replay buffers are very specific to a given implementation of an RL algo)
    def __init__(self, inner_replay_buffer, discriminator_model_fn, discriminator_model_kwargs_getter, restore_model_path, save_model_path, imitation_reward_weight, amp_env, logger, cuda_device, agent_buffer_max_size=1e6, expert_buffer_max_size=1000, save_freq_updates=10000):
        self.replay_buffer = inner_replay_buffer # Let's wrap one of spinup's replay buffers.
        self.max_size = self.replay_buffer.max_size

        self.cuda_device = cuda_device
        obs_dim_in = amp_env.get_amp_obs_size()
        self.dict_space = False
        if np.isscalar(obs_dim_in):
            obs_dim = {"obs":[obs_dim_in]}
        elif isinstance(obs_dim_in, gym.spaces.Dict):
            obs_dim = {k:v for k,v in obs_dim_in.spaces.items()}
            self.dict_space = True
        elif isinstance(obs_dim_in, dict):
            obs_dim = {k:v for k,v in obs_dim_in.items()}
            self.dict_space = True
        else:
            obs_dim = {"obs":obs_dim_in}
        self.obs_modalities = obs_dim.keys()
        
        self.agent_buffer_max_size = self.replay_buffer.max_size # The agent buffer must be as large as the main buffer, so that we can compute rewards for each batch from it.
        self.agent_buffer_max_size_disc = agent_buffer_max_size # This is a much smaller buffer that will be used to train the discriminator, i.e., we'll train on the last agent_buffer_max_size transitions only.
        self.agent_buffer_ptr, self.agent_buffer_size = 0, 0
        self.amp_obs_buf_agent = {k:np.zeros((self.replay_buffer.max_size, v) if np.isscalar(v) else (self.replay_buffer.max_size, *v), dtype=np.float32) for k,v in obs_dim.items()}
        
        self.expert_buffer_max_size = expert_buffer_max_size if expert_buffer_max_size is not None else self.replay_buffer.max_size
        self.expert_buffer_ptr, self.expert_buffer_size = 0, 0
        self.amp_obs_buf_expert = {k:np.zeros((self.expert_buffer_max_size, v) if np.isscalar(v) else (self.expert_buffer_max_size, *v), dtype=np.float32) for k,v in obs_dim.items()}

        self.model_path = save_model_path
        self.save_freq_updates = save_freq_updates
        path_head = os.path.split(self.model_path)[0]
        if not os.path.exists(path_head):
            os.makedirs(path_head)
        self.update_counter = 0
        if restore_model_path is not None and os.path.exists(restore_model_path):
            discriminator_model = self.load_model(os.path.join(restore_model_path, os.path.split(self.model_path)[1]))
        else:
            model_kwargs = discriminator_model_kwargs_getter()
            model_kwargs["input_sizes"] = [obs_dim] + model_kwargs["input_sizes"]
            model_kwargs["output_sizes"] = [1]
            model_kwargs["output_activation"] = nn.Identity
            discriminator_model = discriminator_model_fn(**model_kwargs)
        self.amp_discriminator = AMPDiscriminator(discriminator_model=discriminator_model, imitation_reward_weight=imitation_reward_weight, amp_env=amp_env, amp_buffer=self, logger=logger, cuda_device=cuda_device)

    def store(self, obs, act, rew, next_obs, done, state_amp_agent, state_amp_expert, hidden_ep_rew=None):
        if type(self.replay_buffer) is AMPReplayBuffer:
            self.replay_buffer.store(obs, act, rew, next_obs, done, state_amp_agent=state_amp_agent[:-1], state_amp_expert=state_amp_expert[:-1], hidden_ep_rew=hidden_ep_rew)
        else:
            self.replay_buffer.store(obs, act, rew, next_obs, done, hidden_ep_rew=hidden_ep_rew)

        for k in self.obs_modalities:
            self.amp_obs_buf_agent[k][self.agent_buffer_ptr] = state_amp_agent[-1][k] if self.dict_space else state_amp_agent[-1]
            self.amp_obs_buf_expert[k][self.expert_buffer_ptr] = state_amp_expert[-1][k] if self.dict_space else state_amp_expert[-1]

        self.agent_buffer_ptr = (self.agent_buffer_ptr + 1) % self.agent_buffer_max_size
        self.agent_buffer_size = min(self.agent_buffer_size + 1, self.agent_buffer_max_size)

        self.expert_buffer_ptr = (self.expert_buffer_ptr + 1) % self.expert_buffer_max_size
        self.expert_buffer_size = min(self.expert_buffer_size + 1, self.expert_buffer_max_size)

    def sample_batch(self, batch_size, imitation_lambda=1.0):
        batch = self.replay_buffer.sample_batch(batch_size)
        self.cur_idxs = self.replay_buffer.cur_idxs
        
        batch["rew"] = self.amp_discriminator.batch_calc_reward(
            obs_agent_amp_batch={k:self.amp_obs_buf_agent[k][self.replay_buffer.cur_idxs] for k in self.obs_modalities}, 
            rewards_batch=batch["rew"],
            imitation_lambda=imitation_lambda)
        
        return batch

    def sample_amp_batch_agent(self, batch_size):
        # TODO let's try to use only the more recent demonstrations.
        #cur_idxs = np.random.randint(0, self.replay_buffer.size, size=batch_size)
        min_idx = self.agent_buffer_ptr - self.agent_buffer_max_size_disc
        cur_idxs = np.random.randint(min_idx if self.agent_buffer_size == self.agent_buffer_max_size else max(0, min_idx), self.agent_buffer_ptr, size=batch_size)
        return {k:torch.as_tensor(self.amp_obs_buf_agent[k][cur_idxs], dtype=torch.float32, device=self.cuda_device) for k in self.amp_obs_buf_agent}

    def sample_amp_batch_expert(self, batch_size):
        cur_idxs = np.random.randint(0, self.expert_buffer_size, size=batch_size)
        #cur_idxs = np.random.randint(max(0, self.replay_buffer.size - 10000), self.replay_buffer.size, size=batch_size)
        return {k:torch.as_tensor(self.amp_obs_buf_expert[k][cur_idxs], dtype=torch.float32, device=self.cuda_device) for k in self.amp_obs_buf_expert}
    
    def update_prev_batch(self, new_priorities=None):
        self.replay_buffer.update_prev_batch(new_priorities)
        # TODO let's see if we can update the discriminator only if the avg loss surpasses a certain threshold.
        if self.update_counter <= 0 or (self.amp_discriminator.mean_batch_reward >= self.amp_discriminator.disc_loss_thresh and self.update_counter % self.amp_discriminator.amp_update_interval == 0):
        #if self.update_counter <= 0 or self.update_counter % self.amp_discriminator.amp_update_interval == 0:
            self.amp_discriminator.update_disc()
        self.update_counter += 1
        if self.update_counter % self.save_freq_updates == 0:
            self.save_model()

    def discriminator_predict(self, input_obs):
        #return self.amp_discriminator.calc_disc_reward(input_obs) / self.amp_discriminator.max_batch_reward
        return self.amp_discriminator.imitation_reward_weight * self.amp_discriminator.calc_disc_reward(input_obs)

    def save_model(self):
        torch.save(self.amp_discriminator.discriminator_model, self.model_path)

    def load_model(self, restore_model_path):
        return torch.load(restore_model_path)

class AMPDiscriminator:
    """
    An adversarial discriminator that is trained to distinguish between reference transitions and transitions
    produced by an agent. The implementation is based on DeepMimic's AMPAgent but isn't tied to a specific
    RL algorithm like PPO. 
    """
    def __init__(self, discriminator_model, imitation_reward_weight, amp_env, amp_buffer, logger, cuda_device, batchsize=128, steps_per_batch=1):

        # TODO it would certainly be cool to allow for amp observation stacks. Not sure if the expert demonstrations are ordered, though.
        self.amp_env = amp_env
        self.cuda_device = cuda_device
        if self.cuda_device is not None:
            discriminator_model.to(self.cuda_device)
        # TODO I'll get rid of the normalizers for now, because those didn't seem to do very much for me anyway.
        amp_obs_size = amp_env.get_amp_obs_size()
        
        # TODO I should enforce that the output size is 1
        self.discriminator_model = discriminator_model
        self._reward_scale = 1.0

        #self.disc_optimizer = Adam(self.discriminator_model.parameters(), lr=1e-3, weight_decay=5e-5)
        self.disc_optimizer = Adam(self.discriminator_model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        self.imitation_reward_weight = imitation_reward_weight
        self._disc_batchsize = batchsize
        self._disc_steps_per_batch = steps_per_batch
        self.amp_buffer = amp_buffer
        self.amp_expert_inputs_with_grad = None
        #self.max_batch_reward = 1.0
        
        self.mean_batch_reward = 0.0
        self.disc_loss_thresh = 0 #0.25 #0.001
        
        self.amp_update_interval = 1 #128 #100
        #self.rew_scale = 1. / self.disc_loss_thresh

        self.logger = logger

    def compute_loss(self, amp_expert_inputs, amp_agent_inputs, disc_grad_penalty=10.0):
        if disc_grad_penalty > 0:
            if self.amp_expert_inputs_with_grad is None:
                self.amp_expert_inputs_with_grad = {k:amp_expert_inputs[k].clone().detach().requires_grad_(True) for k in amp_expert_inputs}
            else:
                self.assign_data_directly(from_tensor_dict=amp_expert_inputs, to_tensor_dict=self.amp_expert_inputs_with_grad)
            model_outputs_expert = self.discriminator_model(self.amp_expert_inputs_with_grad)
        else:
            model_outputs_expert = self.discriminator_model(amp_expert_inputs)

        model_outputs_agent = self.discriminator_model(amp_agent_inputs)
        disc_loss_expert = 0.5 * torch.sum((model_outputs_expert - 1)**2, dim=-1)
        disc_loss_agent = 0.5 * torch.sum((model_outputs_agent + 1)**2, dim=-1)
        disc_loss_expert = disc_loss_expert.mean()
        disc_loss_agent = disc_loss_agent.mean()

        disc_loss = 0.5 * (disc_loss_agent + disc_loss_expert)
        #disc_loss = 0.9 * disc_loss_agent + 0.1 * disc_loss_expert

        # TODO This, too, seems to be meta info only.
        #acc_expert = torch.gt(model_outputs_expert.detach(), 0).float().mean().cpu().numpy()
        #acc_agent = torch.lt(model_outputs_agent.detach(), 0).float().mean().cpu().numpy()
         
        # TODO let's see if I can get away with not using this extra regularization for the logit layer.

        if disc_grad_penalty > 0:
            disc_loss += disc_grad_penalty * self.disc_grad_penalty_loss(inputs=self.amp_expert_inputs_with_grad, outputs=model_outputs_expert)

        return disc_loss

    def disc_grad_penalty_loss(self, inputs, outputs):
        gradients = [torch.autograd.grad(outputs=outputs, inputs=inputs[k],
                                        grad_outputs=torch.ones_like(outputs), # TODO I don't really know if I need this.
                                        create_graph=True, retain_graph=True,
                                        allow_unused=True)[0] for k in inputs]
        gradient_norm = torch.cat([torch.flatten(torch.sum(grads**2, dim=-1)) for grads in gradients], dim=-1)
        return 0.5 * gradient_norm.mean()

    def assign_data_directly(self, from_tensor_dict, to_tensor_dict):
        with torch.no_grad(): # We need to tell pytorch's autograd to look away for a second.
            for k in from_tensor_dict:
                to_tensor_dict[k].data = from_tensor_dict[k].data #TODO Not sure if this is the most beautiful way of doing it, but at least this should copy.
                to_tensor_dict[k].grad.zero_()

    def update_disc(self):
        obs_agent = self.amp_buffer.sample_amp_batch_agent(self._disc_batchsize)
        obs_expert = self.amp_buffer.sample_amp_batch_expert(self._disc_batchsize)
        self.step_disc(obs_expert=obs_expert, obs_agent=obs_agent)

    def step_disc(self, obs_expert, obs_agent):
        self.discriminator_model.train()
        self.disc_optimizer.zero_grad()

        loss = self.compute_loss(amp_expert_inputs=obs_expert, amp_agent_inputs=obs_agent)
        #loss, acc_expert, acc_agent = self.compute_loss(obs_expert, obs_agent)

        loss.backward()
        self.disc_optimizer.step()
 
        self.logger.store(LossAmp=loss)
        #self.logger.store(AccExpertAMP=acc_expert)
        #self.logger.store(AccAgentAMP=acc_agent)

        self.discriminator_model.eval()

    def batch_calc_reward(self, obs_agent_amp_batch, rewards_batch, imitation_lambda):
        disc_r = self.calc_disc_reward(obs_agent_amp_batch)

        #self.logger.store(AmpRew=np.average(disc_r))
        #self.logger.store(AmpRewBatchMax=np.amax(disc_r))
        #self.logger.store(AmpRewBatchMin=np.amin(disc_r))
        # Let's see what happens if we normalize so that every batch generates a nice gradient.
        
        # TODO I'll move this to the actual loss computation.
        if self.disc_loss_thresh > 0:
            self.mean_batch_reward = torch.mean(disc_r).detach()
        
        #disc_r += 1e-9
        #self.max_batch_reward = torch.max(torch.abs(disc_r)).detach()
        #disc_r = disc_r / self.max_batch_reward

        return imitation_lambda * self.imitation_reward_weight * disc_r + (1.0 - imitation_lambda) * rewards_batch

    def calc_disc_reward(self, amp_obs):
        amp_obs_tensor = {k:torch.as_tensor(amp_obs[k], dtype=torch.float32, device=self.cuda_device) for k in amp_obs}
        r = 1.0 - 0.25 * (1.0 - self.discriminator_model(amp_obs_tensor))**2
        r = torch.clamp(r, min=0.0)
        r = r[:, 0]
        #r *= r > 0.5 # TODO remove
        return r