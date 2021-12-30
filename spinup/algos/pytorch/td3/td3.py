from copy import deepcopy
import itertools
import numpy as np
import torch
import os
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.td3.core as core
import spinup.models.pytorch.multi_modal_net as model # TODO make configurable
#import spinup.models.pytorch.mlp as model
from spinup.utils.logx import EpochLogger
from spinup.utils.plot import DynamicPlotter

from spinup.env_wrappers.env_wrapper_utils import wrap_envs
from spinup.meta_algos.meta_optimizer import MetaOptimizer


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim_in, act_dim, size, cuda_device, split_reward_streams):
        self.dict_space = False
        if np.isscalar(obs_dim_in):
            obs_dim = {"obs":[obs_dim_in]}
        elif isinstance(obs_dim_in, gym.spaces.Dict):
            obs_dim = {k:v.shape for k,v in obs_dim_in.spaces.items()}
            self.dict_space = True
        else:
            obs_dim = {"obs":obs_dim_in.shape}
        self.obs_modalities = obs_dim.keys()
        if self.dict_space:
            self.obs_buf = {k:np.zeros(core.combined_shape(size, v), dtype=np.float32) for k,v in obs_dim.items()}
            self.obs2_buf = {k:np.zeros(core.combined_shape(size, v), dtype=np.float32) for k,v in obs_dim.items()}
        else:
            self.obs_buf = np.zeros(core.combined_shape(size, obs_dim_in.shape), dtype=np.float32)
            self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim_in.shape), dtype=np.float32)
        
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32) if not split_reward_streams else np.zeros((size, 2), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.cur_idxs = []
        self.ptr, self.size, self.max_size = 0, 0, size
        self.cuda_device = cuda_device

    def store(self, obs, act, rew, next_obs, done):
        if self.dict_space:
            for k in self.obs_modalities:
                self.obs_buf[k][self.ptr] = obs[k]
                self.obs2_buf[k][self.ptr] = next_obs[k]
        else:
            self.obs_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        self.cur_idxs = np.random.randint(0, self.size, size=batch_size)
        if self.dict_space:
            batch = dict(obs={k:self.obs_buf[k][self.cur_idxs] for k in self.obs_buf},
                         obs2={k:self.obs2_buf[k][self.cur_idxs] for k in self.obs_buf},
                         act=self.act_buf[self.cur_idxs],
                         rew=self.rew_buf[self.cur_idxs],
                         done=self.done_buf[self.cur_idxs])
            return {k: {sub_k: torch.as_tensor(sub_v, dtype=torch.float32, device=self.cuda_device) for sub_k, sub_v in v.items()} 
                    if k.startswith("obs") else 
                    torch.as_tensor(v, dtype=torch.float32, device=self.cuda_device) for k,v in batch.items()}
        else:
            batch = dict(obs=self.obs_buf[self.cur_idxs],
                     obs2=self.obs2_buf[self.cur_idxs],
                     act=self.act_buf[self.cur_idxs],
                     rew=self.rew_buf[self.cur_idxs],
                     done=self.done_buf[self.cur_idxs])
            return {k: torch.as_tensor(v, dtype=torch.float32, device=self.cuda_device) for k,v in batch.items()}

    def update_prev_batch(self, new_priorities=None):
        pass

def td3(env_fn,
        actor_critic=core.ActorCritic,
        seed=None,
        env_tools_kwargs={},
        ac_kwargs={"model":model.net, "model_kwargs_getter":model.get_default_kwargs},
        split_reward_streams=False,
        imitation_reward_weight=0,
        normalize_env=False,
        stack_size=0,
        greedy_adversarial_priors=False,
        her_k=0,
        her_reward=10.0,
        steps_per_epoch=4000,
        epochs=10000,
        replay_size=int(1e6), 
        gamma=(.8,.8,.99,2000,10,0.01),
        gamma_intrinsic=(.8,.8,.99,2000,10,0.01),
        max_skip=10,
        signal_vector_size=0,
        intrinsic_reward_weight=0,
        episodic_curiosity=0,
        polyak=0.995,
        pi_lr=(1e-6, 5e-4, 1.0, 4000, 10, 1e-5),
        q_lr=(1e-6, 5e-4, 1.0, 4000, 10, 1e-5),
        max_grad_norm=-1.0,
        min_target_q=-float("inf"),
        max_target_q=float("inf"),
        batch_size=1024,
        start_steps=0,
        live_plot=False,
        update_after=1000,
        update_every=1,
        act_noise=(0.,.3,.5,500,25,0.01),
        target_noise=(0.,.2,.2,1000,10,0.01),
        noise_clip=0.5,
        policy_delay=2,
        num_test_episodes=0,
        max_ep_len=2000,
        logger_kwargs=dict(),
        save_freq=10,
        render_steps=True,
        cuda_device="cuda:0",
        restore_model_path=None,
        output_pickle_trajectories=None):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer. 

        gamma (float): Discount factor. (Always between 0 and 1.) Alternative, this hyperparameter can also be a list 
            [lower_bound, initial_value, upper_bound, update_interval, population_size, stddev_meta_search]
            so that it can be tuned automatically by a MetaOptimizer.

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float or list): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)
            Alternative, this hyperparameter can also be a list 
            [lower_bound, initial_value, upper_bound, update_interval, population_size, stddev_meta_search]
            so that it can be tuned automatically by a MetaOptimizer.

        target_noise (float): Stddev for smoothing noise added to target 
            policy. Alternative, this hyperparameter can also be a list 
            [lower_bound, initial_value, upper_bound, update_interval, population_size, stddev_meta_search]
            so that it can be tuned automatically by a MetaOptimizer.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
            
        render_steps (bool): Whether to render the individual trining steps.

        cuda_device (string): Name of the cuda device that will be used. Default "cuda:0".

        restore_model_path (string): Path/to/model that will be restored to continue a training run. 
            None by default, meaning that a newly initialized model will be used.
    """

    logger = EpochLogger(**logger_kwargs) # TODO maybe I need one logger per agent.
    logger.save_config(locals())
    save_path = logger_kwargs["output_dir"]

    ac_kwargs["split_reward_streams"] = split_reward_streams

    if live_plot:
        plot_labels = ["ExtRew", "IntRew"] if split_reward_streams else ["Rew"]
        if imitation_reward_weight > 0:
            plot_labels.append("ImitationRew")
        plotter = DynamicPlotter(plot_labels)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if split_reward_streams:
        meta_optimizer = MetaOptimizer([act_noise, target_noise, gamma, gamma_intrinsic, pi_lr, q_lr])
        act_noise, target_noise, gamma, gamma_intrinsic, pi_lr, q_lr = meta_optimizer.get_hyperparameter_values()
    else:
        meta_optimizer = MetaOptimizer([act_noise, target_noise, gamma, pi_lr, q_lr])
        act_noise, target_noise, gamma, pi_lr, q_lr = meta_optimizer.get_hyperparameter_values()

    env = env_fn()
    if num_test_episodes > 0:
        test_env = env_fn()
    else:
        test_env = None

    env, test_env, intrinsic_reward_generators = wrap_envs(env=env, test_env=test_env,
                                env_tools_kwargs=env_tools_kwargs,
                                output_pickle_trajectories=output_pickle_trajectories,
                                imitation_reward_weight=imitation_reward_weight,
                                normalize=normalize_env,
                                max_skip=max_skip,
                                signal_vector_size=signal_vector_size,
                                stack_size=stack_size,
                                greedy_adversarial_priors=greedy_adversarial_priors,
                                split_reward_streams=split_reward_streams,
                                intrinsic_reward_weight=intrinsic_reward_weight,
                                her_k=her_k, 
                                her_reward=her_reward,
                                cuda_device=cuda_device,
                                model=model,
                                restore_model_path=restore_model_path,
                                render_steps=render_steps,
                                logger=logger)

    if render_steps and intrinsic_reward_weight > 0:
        # TODO for some reason minerl's render crashes for td3, if I don't do it here.
        env.render()

    n_agents = len(env.observation_space)

    obs_dim = [space for space in env.observation_space]
    act_dim = [space.shape[0] for space in env.action_space]
    dict_space = any([isinstance(o, gym.spaces.Dict) for o in obs_dim])

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit_low = [np.array(space.low) for space in env.action_space]
    act_limit_high = [np.array(space.high) for space in env.action_space]
    act_limit_scale = [np.clip(act_limit_high[i] - act_limit_low[i], 0, 100) for i in range(n_agents)] 

    # Create actor-critic module and target networks
    if restore_model_path is None:
        ac = [actor_critic(env.observation_space[i], env.action_space[i], **ac_kwargs) for i in range(n_agents)]
    else:
        # TODO careful, spinup does not use the recommended way of saving models in pytorch by saving just the state_dict (which doesn't depend on the projects current directory structure).
        # Also, it does not restore an entire checkpoint with optimizer variables etc. - so strictly speaking, I can't really use it to resume training.
        ac = torch.load(os.path.join(restore_model_path, "model.pt"))
    ac_targ = deepcopy(ac)

    q_params = []
    replay_buffers = []
    for i in range(n_agents):
        if cuda_device is not None:
            ac[i].to(cuda_device)
            ac_targ[i].to(cuda_device)

        ac[i].eval()
        ac_targ[i].eval()

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ[i].parameters():
            p.requires_grad = False
        
        # List of parameters for both Q-networks (save this for convenience)
        q_params.append(list(itertools.chain(ac[i].q1.parameters(), ac[i].q2.parameters()))) # TODO careful, the chain apparently gets consumed.

        # Experience buffer
        spinup_replay_buffer = ReplayBuffer(obs_dim_in=obs_dim[i], act_dim=act_dim[i], size=replay_size, cuda_device=cuda_device, split_reward_streams=split_reward_streams)
        
        if hasattr(env, "get_amp_obs"):
            from spinup.replay_buffers.amp_replay_buffer_pytorch import AMPReplayBuffer
            replay_buffers.append(AMPReplayBuffer(inner_replay_buffer=spinup_replay_buffer, 
                discriminator_model_fn=model.net, discriminator_model_kwargs_getter=model.get_default_kwargs,
                restore_model_path=restore_model_path, save_model_path=os.path.join(logger.output_dir, "pyt_save", "amp_discriminator_model_" + str(i) + ".pt"),
                imitation_reward_weight=imitation_reward_weight, amp_env=env, logger=logger, cuda_device=cuda_device))
            amp_step_return_required = meta_optimizer is not None # If we want the meta optimizer to maximize amp rewards, we need to compute them for each step.
        else:
            amp_step_return_required = False
            replay_buffers.append(spinup_replay_buffer)
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [ac[i].pi, ac[i].q1, ac[i].q2])
        logger.log("\nNumber of parameters agent " + str(i) + ": \t pi: %d, \t q1: %d, \t q2: %d\n"%var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(agent_nbr, data, overall_max_ep_ret, overall_min_ep_ret):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac[agent_nbr].q1(o,a)
        q2 = ac[agent_nbr].q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ[agent_nbr].pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            #a2 = torch.clamp(a2, act_limit_low[agent_nbr], act_limit_high[agent_nbr])
            a2 = torch.max(torch.min(a2, torch.tensor(act_limit_high[agent_nbr], device=cuda_device)), torch.tensor(act_limit_low[agent_nbr], device=cuda_device))

            # Target Q-values
            q1_pi_targ = ac_targ[agent_nbr].q1(o2, a2)
            q2_pi_targ = ac_targ[agent_nbr].q2(o2, a2)
            q1_pi_targ = torch.clamp(q1_pi_targ, overall_min_ep_ret[agent_nbr], overall_max_ep_ret[agent_nbr])
            q2_pi_targ = torch.clamp(q2_pi_targ, overall_min_ep_ret[agent_nbr], overall_max_ep_ret[agent_nbr])

            if not split_reward_streams:
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + gamma * (1 - d) * q_pi_targ
            else:
                q_pi_targ = torch.min(q1_pi_targ[:,0], q2_pi_targ[:,0])
                backup = r[:,0] + gamma * (1 - d) * q_pi_targ

                # similarly for the intrinsic q value: #TODO Is it a good idea to add the entropy bonus again?
                q_pi_targ_intrinsic = torch.min(q1_pi_targ[:,1], q2_pi_targ[:,1])
                backup_intrinsic = r[:,1] + gamma_intrinsic * (1 - d * episodic_curiosity) * q_pi_targ_intrinsic


        # MSE loss against Bellman backup
        if not split_reward_streams:
            loss_q1 = ((q1 - backup)**2).mean()
            loss_q2 = ((q2 - backup)**2).mean()
            loss_q = loss_q1 + loss_q2

            # Useful info for logging
            q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                          Q2Vals=q2.detach().cpu().numpy())
        else:
            loss_q1_extrinsic = ((q1[:,0] - backup)**2).mean()
            loss_q2_extrinsic = ((q2[:,0] - backup)**2).mean()
            loss_q_extrinsic = loss_q1_extrinsic + loss_q2_extrinsic

            loss_q1_intrinsic = ((q1[:,1] - backup_intrinsic)**2).mean()
            loss_q2_intrinsic = ((q2[:,1] - backup_intrinsic)**2).mean()
            loss_q_intrinsic = loss_q1_intrinsic + loss_q2_intrinsic

            loss_q = loss_q_extrinsic + loss_q_intrinsic

            # Useful info for logging
            q1_np = q1.detach().cpu().numpy()
            q2_np = q1.detach().cpu().numpy()
            q_info = dict(Q1Vals=q1_np[:,0] + q1_np[:,1],
                Q2Vals=q2_np[:,0] + q2_np[:,1], 
                Q1ValsExternal=q1_np[:,0], Q1ValsInternal=q1_np[:,1],
                Q2ValsExternal=q1_np[:,0], Q2ValsInternal=q1_np[:,1])

        return loss_q, q_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(agent_nbr, data, overall_max_ep_ret, overall_min_ep_ret):
        o = data['obs']
        q1_pi = ac[agent_nbr].q1(o, ac[agent_nbr].pi(o))
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = []
    q_optimizer = []
    for i in range(n_agents):
        #pi_optimizer.append(Adam(ac[i].pi.parameters(), lr=pi_lr, weight_decay=0.0001))
        #q_optimizer.append(Adam(q_params[i], lr=q_lr, weight_decay=0.0001))
        pi_optimizer.append(Adam(ac[i].pi.parameters(), lr=pi_lr))
        q_optimizer.append(Adam(q_params[i], lr=q_lr))

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(agent_nbr, data, timer, overall_max_ep_ret, overall_min_ep_ret):
        ac[agent_nbr].train()
        ac_targ[agent_nbr].train()

        # First run one gradient descent step for Q1 and Q2
        q_optimizer[agent_nbr].zero_grad()
        loss_q, loss_info = compute_loss_q(agent_nbr, data, overall_max_ep_ret, overall_min_ep_ret)
        loss_q.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(q_params[agent_nbr], max_grad_norm)
        q_optimizer[agent_nbr].step()

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in q_params[agent_nbr]:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer[agent_nbr].zero_grad()
            loss_pi = compute_loss_pi(agent_nbr, data, overall_max_ep_ret, overall_min_ep_ret)
            loss_pi.backward()
            pi_optimizer[agent_nbr].step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params[agent_nbr]:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac[agent_nbr].parameters(), ac_targ[agent_nbr].parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
            ac[agent_nbr].eval()
            ac_targ[agent_nbr].eval()

    def get_action(o, noise_sigma):      
        actions = [ac[i].act({k_o: torch.as_tensor([v_o], dtype=torch.float32, device=cuda_device) for k_o, v_o in o[i].items()} 
                                    if dict_space else torch.as_tensor([o[i]], dtype=torch.float32, device=cuda_device))[0]
                        for i in range(n_agents)]
        noise_scales = [noise_sigma * act_limit_scale[i] for i in range(n_agents)]
        return np.array([np.clip(actions[i] + noise_scales[i] * np.random.randn(act_dim[i]), act_limit_low[i], act_limit_high[i]) for i in range(n_agents)])

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), [False] * n_agents, np.zeros(n_agents), 0
            while not(any(d) or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += np.array(r)
                ep_len += 1
            logger.store(TestEpRet=np.average(ep_ret), TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, discounted_ep_ret, ep_ret_ext, ep_ret_int, ep_len = env.reset(), np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents), 0

    #param_noise_interval = 100
    overall_max_ep_ret = [max_target_q] * n_agents if max_target_q is not None else [-float("inf")] * n_agents
    overall_min_ep_ret = [min_target_q] * n_agents if min_target_q is not None else [float("inf")] * n_agents
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            #if t % param_noise_interval == 0:
            #    for actor in ac + ac_targ:
            #        for submodule in actor.modules():
            #            if hasattr(submodule, "generate_parameter_noise"):
            #                submodule.generate_parameter_noise(device=cuda_device)
            a = get_action(o, act_noise)
        else:
            a = np.array([space.sample() for space in env.action_space])

        # Step the env
        o2, r, d, _ = env.step(a)

        if not split_reward_streams:
            ep_ret += np.array(r)
            discounted_ep_ret = np.array(r) + gamma * discounted_ep_ret
            overall_max_ep_ret = [discounted_ep_ret[i] if discounted_ep_ret[i] > overall_max_ep_ret[i] else overall_max_ep_ret[i] for i in range(n_agents)]
            overall_min_ep_ret = [discounted_ep_ret[i] if discounted_ep_ret[i] < overall_min_ep_ret[i] else overall_min_ep_ret[i] for i in range(n_agents)]
            if live_plot and (t+1) % steps_per_epoch != 0:
                plotter.add_entry(entry=np.average(r), plot="Rew")
        else:
            r_e = r[0]
            r_i = r[1]
            ep_ret += np.array(r_e) + np.array(r_i)
            discounted_ep_ret = np.array(r) + gamma * discounted_ep_ret
            overall_max_ep_ret = [discounted_ep_ret[i] if discounted_ep_ret[i] > overall_max_ep_ret[i] else overall_max_ep_ret[i] for i in range(n_agents)]
            overall_min_ep_ret = [discounted_ep_ret[i] if discounted_ep_ret[i] < overall_min_ep_ret[i] else overall_min_ep_ret[i] for i in range(n_agents)]
            ep_ret_ext += np.array(r_e)
            ep_ret_int += np.array(r_i)
            if live_plot and (t+1) % steps_per_epoch != 0:
                plotter.add_entry(entry=np.average(r_e), plot="ExtRew")
                plotter.add_entry(entry=np.average(r_i), plot="IntRew")

        live_plot_needs_update = live_plot and (t+1) % steps_per_epoch != 0
        compute_amp_step_return = (amp_step_return_required or live_plot_needs_update) and hasattr(env, "get_amp_obs")
        avg_amp_step_rewards = None
        if compute_amp_step_return:
            prev_amp_discriminator_rewards = []
            for i in range(n_agents):
                if hasattr(replay_buffers[i], "discriminator_predict") and hasattr(env, "get_prev_state_amp_agent"):
                    state_agent = env.get_prev_state_amp_agent(i)
                    if state_agent is not None:
                        prev_amp_discriminator_rewards.append(replay_buffers[i].discriminator_predict(state_agent).detach().cpu().numpy())
            avg_amp_step_rewards = np.average(prev_amp_discriminator_rewards)

        avg_hidden_step_rewards = None
        if hasattr(env, "get_hidden_step_rewards"):
            avg_hidden_step_rewards = np.average(env.get_hidden_step_rewards())

        if live_plot_needs_update:
            if avg_amp_step_rewards is not None:
                plotter.add_entry(entry=avg_amp_step_rewards, plot="ImitationRew")
            if avg_hidden_step_rewards is not None:
                plotter.add_entry(entry=avg_hidden_step_rewards, plot="HiddenStepRew")
            plotter.plot()

        ep_len += 1

        if render_steps:
            env.render()

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        #TODO I think this isn't correct.
        #d = [False] * n_agents if ep_len==max_ep_len else d

        # Store experience to replay buffer
        for i in range(n_agents):
            if hasattr(env, "get_amp_obs"):
                amp_obs = env.get_amp_obs(i)
                if amp_obs is not None:
                    state_amp_agent = amp_obs["state_amp_agent"]
                    state_amp_expert = amp_obs["state_amp_expert"]
                    replay_buffers[i].store(o[i], a[i], r[i], o2[i], d[i], state_amp_agent, state_amp_expert)
            else:
                replay_buffers[i].store(o[i], a[i], r[i], o2[i], d[i])

            if her_k > 0:
                for transition in env.get_her_transitions(i):
                    replay_buffers[i].store(*transition)

        rew_for_meta_optimizer = np.average(r)
        if avg_amp_step_rewards is not None:
            rew_for_meta_optimizer += avg_amp_step_rewards
        if avg_hidden_step_rewards is not None:
            rew_for_meta_optimizer += avg_hidden_step_rewards
        if split_reward_streams:
            act_noise, target_noise, gamma, gamma_intrinsic, pi_lr, q_lr = meta_optimizer.update_hyperparameters(step_counter=t, step_reward=rew_for_meta_optimizer)
        else:
            act_noise, target_noise, gamma, pi_lr, q_lr = meta_optimizer.update_hyperparameters(step_counter=t, step_reward=rew_for_meta_optimizer)
        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if np.any(d) or (ep_len == max_ep_len):
            logger.store(EpRet=np.average(ep_ret), EpRetExternal=np.average(ep_ret_ext), EpRetInternal=np.average(ep_ret_int), EpLen=ep_len)
            o, ep_ret, discounted_ep_ret, ep_ret_ext, ep_ret_int, ep_len = env.reset(), np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents), 0
            if intrinsic_reward_weight > 0 and episodic_curiosity < 1.0:
                # We need to allow bootstrapping of intrinsic values between episodes.
                # TODO check if this works.
                # TODO replicate with sac etc.
                for i in range(n_agents):
                    if hasattr(env, "get_amp_obs"):
                        amp_obs = env.get_amp_obs(i)
                        if amp_obs is not None:
                            state_amp_agent = amp_obs["state_amp_agent"]
                            state_amp_expert = amp_obs["state_amp_expert"]
                            replay_buffers[i].store(o2[i], a[i], (0,0) if split_reward_streams else 0, o[i], False, state_amp_agent, state_amp_expert)
                    else:
                        replay_buffers[i].store(o2[i], a[i], (0,0) if split_reward_streams else 0, o[i], False)

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                for i in range(n_agents):
                    batch = replay_buffers[i].sample_batch(batch_size)
                    for k in range(len(intrinsic_reward_generators)):
                        int_rew_gen = intrinsic_reward_generators[k]
                        obs_batch = batch['obs']
                        if split_reward_streams:
                            batch['rew'][:,k+1] = int_rew_gen.compute_intrinsic_rewards(i, obs_batch)
                        else:
                            batch['rew'] += int_rew_gen.compute_intrinsic_rewards(i, obs_batch)
                    update(agent_nbr=i, data=batch, timer=j, overall_max_ep_ret=overall_max_ep_ret, overall_min_ep_ret=overall_min_ep_ret)
                    replay_buffers[i].update_prev_batch()
            if not np.isscalar(pi_lr):
                for optimizer in pi_optimizer:
                    for params in optimizer.param_groups:
                        params['lr'] = pi_lr
            if not np.isscalar(q_lr):
                for optimizer in q_optimizer:
                    for params in optimizer.param_groups:
                        params['lr'] = q_lr

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpRetExternal', with_min_and_max=True)
            logger.log_tabular('EpRetInternal', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('act_noise', act_noise)
            logger.log_tabular('target_noise', target_noise)
            logger.log_tabular('gamma', gamma)
            logger.log_tabular('pi_lr', pi_lr)
            logger.log_tabular('q_lr', q_lr)
            if split_reward_streams:
                logger.log_tabular('gamma_intrinsic', gamma_intrinsic)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('Q1ValsExternal', with_min_and_max=True)
            logger.log_tabular('Q1ValsInternal', with_min_and_max=True)
            logger.log_tabular('Q2ValsExternal', with_min_and_max=True)
            logger.log_tabular('Q2ValsInternal', with_min_and_max=True)
            logger.log_tabular('LossAmp', average_only=True)
            logger.log_tabular('AccExpertAMP', average_only=True)
            logger.log_tabular('AccAgentAMP', average_only=True)
            logger.log_tabular('AmpRew', average_only=True)
            logger.log_tabular('AmpRewBatchMax', average_only=True)
            logger.log_tabular('AmpRewBatchMin', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=None)
    parser.add_argument('--l', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    td3(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
