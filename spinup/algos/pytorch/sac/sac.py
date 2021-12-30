from copy import deepcopy
import itertools
import numpy as np
import torch
import os
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.sac.core as core
from spinup.env_wrappers.env_wrapper_utils import wrap_envs
from spinup.meta_algos.meta_optimizer import MetaOptimizer
# TODO make configurable
#import spinup.models.pytorch.mlp as model
#import spinup.models.pytorch.resnet as model
import spinup.models.pytorch.multi_modal_net as model

from spinup.utils.logx import EpochLogger
from spinup.utils.plot import DynamicPlotter


# TODO create common replay buffer module
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
        self.return_buf = np.zeros(size, dtype=np.float32)
        self.cur_idxs = []
        self.ptr, self.size, self.max_size = 0, 0, size
        self.cuda_device = cuda_device
        self.episode_buffer = []

    def store(self, obs, act, rew, next_obs, done, hidden_ep_rew=None):

        self.episode_buffer.append([obs, act, rew, next_obs, done, hidden_ep_rew])

        if done:
            # Compute returns for this episode.
            total_ret = 0
            for entries in reversed(self.episode_buffer):
                obs, act, rew, next_obs, done, hidden_ep_ret = entries
                total_ret = (rew if hidden_ep_rew is None else hidden_ep_rew) + total_ret
                #total_ret += entries[2]
                entries.append(total_ret)

            # Dump episode and returns into the actual replay buffer.
            for obs, act, rew, next_obs, done, hidden_ep_ret, ret in self.episode_buffer:
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
                self.return_buf[self.ptr] = ret

                self.ptr = (self.ptr+1) % self.max_size
                self.size = min(self.size+1, self.max_size)
            self.episode_buffer = []

    def sample_batch(self, batch_size=32, imitation_lambda=1.0):
        self.cur_idxs = np.random.randint(0, self.size, size=batch_size)
        if self.dict_space:
            batch = dict(obs={k:self.obs_buf[k][self.cur_idxs] for k in self.obs_buf},
                         obs2={k:self.obs2_buf[k][self.cur_idxs] for k in self.obs_buf},
                         act=self.act_buf[self.cur_idxs],
                         rew=self.rew_buf[self.cur_idxs],
                         done=self.done_buf[self.cur_idxs],
                         ret=self.return_buf[self.cur_idxs])
            return {k: {sub_k: torch.as_tensor(sub_v, dtype=torch.float32, device=self.cuda_device) for sub_k, sub_v in v.items()} 
                    if k.startswith("obs") else 
                    torch.as_tensor(v, dtype=torch.float32, device=self.cuda_device) for k,v in batch.items()}
        else:
            batch = dict(obs=self.obs_buf[self.cur_idxs],
                     obs2=self.obs2_buf[self.cur_idxs],
                     act=self.act_buf[self.cur_idxs],
                     rew=self.rew_buf[self.cur_idxs],
                     done=self.done_buf[self.cur_idxs],
                     ret=self.return_buf[self.cur_idxs])
            return {k: torch.as_tensor(v, dtype=torch.float32, device=self.cuda_device) for k,v in batch.items()}

    def update_prev_batch(self, new_priorities=None):
        pass

def sac(env_fn, seed=None,
        env_tools_kwargs={},
        actor_critic=core.ActorCritic,
        ac_kwargs={"model":model.net, "model_kwargs_getter":model.get_default_kwargs},
        split_reward_streams=False, 
        imitation_reward_weight=0,
        normalize_env=False,
        stack_size=0,
        greedy_adversarial_priors=False,
        imitation_lambda=(0, 1.0, 1.0, 5000, 10, .05, 0), # TODO turn into a dict.
        her_k=0,
        her_reward=10.0,
        steps_per_epoch=10000,
        epochs=10000,
        replay_size=int(1e6),
        gamma=(.8, .8, .99, 5000, 10, .05, .99),
        gamma_intrinsic=.99,#(.8,.8,.99,10000,10,0.01,0.99),
        max_skip=10,
        signal_vector_size = 0,
        decorelate_action_dims = True,
        polyak=.995,#(.99,.995,.995,2000,10,0.01,-1),
        lr=5e-4,#(1e-6, 5e-4, 1e-2, 4000, 10, 1e-5,-1),
        max_grad_norm=-1.0,
        q_grad_penalty=0,#(1e-6, 5e-3, 1e-2, 4000, 10, 1e-5,-1), #0,
        alpha=5,#(0.,.5,1.,10000,10,0.2),
        use_interactive_entropy_source=False,
        min_target_q=-float('inf'),
        max_target_q=float('inf'),
        bootstrapping_loss_thresh=-1,
        intrinsic_reward_weight=0,
        episodic_curiosity=0,
        batch_size=512,#256,
        start_steps=0,
        live_plot=False,
        update_after=1000,
        update_every=1,
        update_to_step_ratio=1,
        num_test_episodes=0,
        max_ep_len=10000, #float("inf"),
        logger_kwargs=dict(),
        save_freq=10,
        render_steps=True,
        cuda_device="cuda:0",
        restore_model_path=None,
        output_pickle_trajectories=None):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators. Set to None to use random seed.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to update_to_step_ratio.

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
    #TODO document new stuff!
    #TODO consider looking into "Soft Actor-Critic Algorithms and Applications", Haarnoja et al. (2019), to fit alpha automatically. 

    logger = EpochLogger(**logger_kwargs) # TODO maybe I need one logger per agent.
    logger.save_config(locals())
    save_path = logger_kwargs["output_dir"]

    ac_kwargs["split_reward_streams"] = split_reward_streams

    if imitation_reward_weight == 0 and greedy_adversarial_priors:
        imitation_reward_weight = 1

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if split_reward_streams:
        meta_optimizer = MetaOptimizer([alpha, gamma, gamma_intrinsic, lr, polyak, q_grad_penalty, imitation_lambda])
        alpha, gamma, gamma_intrinsic, lr, polyak, q_grad_penalty, imitation_lambda = meta_optimizer.get_hyperparameter_values()
    else:
        meta_optimizer = MetaOptimizer([alpha, gamma, lr, polyak, q_grad_penalty, imitation_lambda])
        alpha, gamma, lr, polyak, q_grad_penalty, imitation_lambda = meta_optimizer.get_hyperparameter_values()

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
                                    use_interactive_entropy_source=use_interactive_entropy_source,
                                    logger=logger)
    if use_interactive_entropy_source:
        sample_new_noise = env.get_param_value("sample_new_noise") >= 0
    else:
        sample_new_noise = True

    if live_plot:
        plot_labels = ["ExtRew", "IntRew"] if split_reward_streams else ["Rew"]
        if imitation_reward_weight > 0:
            plot_labels.append("ImitationRew")
        if hasattr(env, "get_hidden_step_rewards"):
            plot_labels.append("HiddenStepRew")
        plotter = DynamicPlotter(plot_labels)

    n_agents = len(env.observation_space)

    obs_dim = [space for space in env.observation_space]
    act_dim = [space.shape[0] for space in env.action_space]
    dict_space = any([isinstance(o, gym.spaces.Dict) for o in obs_dim])

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    #act_limit = [space.high[0] for space in env.action_space]

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
        
        if hasattr(env, "stack_amp_replay_buffers"):
            #from spinup.replay_buffers.amp_replay_buffer_pytorch import AMPReplayBuffer
            #replay_buffers.append(AMPReplayBuffer(inner_replay_buffer=spinup_replay_buffer, 
            #    discriminator_model_fn=model.net, discriminator_model_kwargs_getter=model.get_tiny_kwargs,
            #    restore_model_path=restore_model_path, save_model_path=os.path.join(logger.output_dir, "pyt_save", "amp_discriminator_model_" + str(i) + ".pt"),
            #    imitation_reward_weight=imitation_reward_weight, amp_env=env, logger=logger, cuda_device=cuda_device))
            # TODO restore paths would need to be a list [path_deepest_disc, path_second_deepest, ...]
            replay_buffers.append(env.stack_amp_replay_buffers(spinup_replay_buffer, model, imitation_reward_weight, [restore_model_path] if restore_model_path is not None else None, i, logger, cuda_device))
            amp_step_return_required = meta_optimizer is not None
        else:
            amp_step_return_required = False
            replay_buffers.append(spinup_replay_buffer)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [ac[i].pi, ac[i].q1, ac[i].q2])
        logger.log("\nNumber of parameters agent " + str(i) + ": \t pi: %d, \t q1: %d, \t q2: %d\n"%var_counts)

    # Let's scale the entropy bonus/temperature a bit, because it quickly leads to divergence in larger action spaces. 
    max_entropy_bonus = ac[0].getMaxEntropyBonus(dtype=torch.float32, device=cuda_device)
    logger.log("\nScaling alpha to " + str(alpha/np.abs(max_entropy_bonus)) + " to ensure that, on average, a very deterministic policy (close to an all zero action vector) will receive a penalty of " + str(-alpha) + " at each step (like originally configured).\n")
    alpha /= np.abs(max_entropy_bonus)

    max_ep_ret = [-float('inf')] * n_agents

    # Set up function for computing SAC Q-losses
    def compute_loss_q(agent_nbr, data):
        global q_obs_inputs_with_grad
        global q_act_inputs_with_grad

        # TODO consider trying the q loss variants discussed in "Evolving Reinforcement Learning Algorithms" 
        o, a, r, o2, d, ret = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['ret']

        q1 = ac[agent_nbr].q1(o,a)
        q2 = ac[agent_nbr].q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac[agent_nbr].pi(o2, num_samples=1)#, clamp_logprob=True)

            # Target Q-values
            q1_pi_targ = ac_targ[agent_nbr].q1(o2, a2)
            q2_pi_targ = ac_targ[agent_nbr].q2(o2, a2)

            # Let's clamp the taret q values to avoid overestimation.
            q1_pi_targ = torch.clamp(q1_pi_targ, min_target_q, max_target_q)
            q2_pi_targ = torch.clamp(q2_pi_targ, min_target_q, max_target_q)

            if not split_reward_streams:
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
            else:
                q_pi_targ = torch.min(q1_pi_targ[:,0], q2_pi_targ[:,0])
                backup = r[:,0] + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

                # similarly for the intrinsic q value: #TODO Is it a good idea to add the entropy bonus again?
                q_pi_targ_intrinsic = torch.min(q1_pi_targ[:,1], q2_pi_targ[:,1])
                backup_intrinsic = r[:,1] + gamma_intrinsic * (1 - d * episodic_curiosity) * (q_pi_targ_intrinsic - alpha * logp_a2)
 
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
            q2_np = q2.detach().cpu().numpy()
            q_info = dict(Q1Vals=q1_np[:,0] + q1_np[:,1],
                Q2Vals=q2_np[:,0] + q2_np[:,1], 
                Q1ValsExternal=q1_np[:,0], Q1ValsInternal=q1_np[:,1],
                Q2ValsExternal=q2_np[:,0], Q2ValsInternal=q2_np[:,1])

        if q_grad_penalty != 0:
            with torch.no_grad():
                # Let's make sure that we check the actual best max q estimate.
                a_max, _ = ac_targ[agent_nbr].pi(o2, deterministic=True, with_logprob=False)

            if q_obs_inputs_with_grad is None:
                if type(o) is dict:
                    q_obs_inputs_with_grad = {k:o2[k].clone().detach().requires_grad_(True) for k in o2}
                else:
                    q_obs_inputs_with_grad = o2.clone().detach().requires_grad_(True)
                q_act_inputs_with_grad = a_max.clone().detach().requires_grad_(True)
            else:
                assign_data_directly(from_tensor=o2, to_tensor=q_obs_inputs_with_grad)
                assign_data_directly(from_tensor=a_max, to_tensor=q_act_inputs_with_grad)
            
            # Let's see what the online net thinks of the next state and let's make sure that this estimate is a maximum (or at least an extreme value)
            q1_succ_online_estimate = ac[agent_nbr].q1(q_obs_inputs_with_grad, q_act_inputs_with_grad)
            q2_succ_online_estimate = ac[agent_nbr].q2(q_obs_inputs_with_grad, q_act_inputs_with_grad)

            loss_q += 0.25 * q_grad_penalty * q_grad_penalty_loss(inputs=q_obs_inputs_with_grad, outputs=q1_succ_online_estimate)
            loss_q += 0.25 * q_grad_penalty * q_grad_penalty_loss(inputs=q_act_inputs_with_grad, outputs=q1_succ_online_estimate)
            loss_q += 0.25 * q_grad_penalty * q_grad_penalty_loss(inputs=q_obs_inputs_with_grad, outputs=q2_succ_online_estimate)
            loss_q += 0.25 * q_grad_penalty * q_grad_penalty_loss(inputs=q_act_inputs_with_grad, outputs=q2_succ_online_estimate)

        # As another little auxiliary target, let's keep the q values close to the actual return.
        #loss_q += 0.001 * (((q1 - ret)**2).mean() + ((q2 - ret)**2).mean())
        #loss_q += 0.1 * (torch.abs(q1 - ret).mean() + torch.abs(q2 - ret).mean())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(agent_nbr, data):
        o = data['obs']
        pi, logp_pi = ac[agent_nbr].pi(o, num_samples=1)
        q1_pi = ac[agent_nbr].q1(o, pi)
        q2_pi = ac[agent_nbr].q2(o, pi)

        if split_reward_streams:
            q1_pi = torch.sum(q1_pi, dim=-1)
            q2_pi = torch.sum(q2_pi, dim=-1)
        
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        if signal_vector_size > 0 or decorelate_action_dims:
            # Inspired by VICReg regularization for embedding vectors.
            if hasattr(env, "get_signal_from_to_idx") and not decorelate_action_dims:
                signal_from_to = env.get_signal_from_to_idx(agent_nbr)
                pi_sig = pi[:,signal_from_to[0]:signal_from_to[1]] # TODO this might be a problem when multiple wrappers augment the action space.
            else:
                pi_sig = pi
            pi_norm = pi_sig - pi_sig.mean(dim=0)
            cov_pi = (pi_sig.T @ pi_sig) / batch_size
            off_diag_elems = torch.triu(cov_pi, diagonal=1)
            #decorelation_loss = off_diag_elems.pow(2).sum() / act_dim[agent_nbr]
            #decorelation_loss = off_diag_elems.pow(2).sum().sqrt() / act_dim[agent_nbr]
            decorelation_loss = torch.sqrt(off_diag_elems.pow(2).sum() / act_dim[agent_nbr])
            loss_pi += decorelation_loss

            # Let's also encourage high variance along the diagonal.
            #loss_pi -= torch.diagonal(cov_pi).pow(2).sum() / act_dim[agent_nbr]

        # As another little auxiliary target, let's keep the q values close to the actual return.
        #ret = data['ret']
        #loss_pi += 0.1 * torch.abs(q_pi - ret).mean()

        # If we have a discriminator available, we try to use gradients to optimize it's predictions directly.
        # Note that we can only compute policy gradients directly, if the action is part of the observatoin space.

        if hasattr(replay_buffers[i], "discriminator_predict") and "action" in replay_buffers[i].obs_modalities:
            if type(o) is dict:
                state_agent = {("prev_" + k):v for k,v in o.items()}
                state_agent.update({"action":pi})
            else:
                state_agent = {"prev_obs":o, "action":pi}
            disc_pred = replay_buffers[i].discriminator_predict(state_agent)
            loss_pi -= disc_pred.mean()

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = []
    q_optimizer = []
    for i in range(n_agents):
        pi_optimizer.append(Adam(ac[i].pi.parameters(), lr=lr))#, weight_decay=0.00005))
        q_optimizer.append(Adam(q_params[i], lr=lr))#, weight_decay=0.00005))

    # Set up model saving
    logger.setup_pytorch_saver(ac) #TODO let's see if this works on lists.

    def update(agent_nbr, data):

        ac[agent_nbr].train()
        ac_targ[agent_nbr].train()

        # First run one gradient descent step for Q1 and Q2
        q_optimizer[agent_nbr].zero_grad()
        loss_q, q_info = compute_loss_q(agent_nbr, data)
        loss_q.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(q_params[agent_nbr], max_grad_norm)
        q_optimizer[agent_nbr].step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params[agent_nbr]:
            p.requires_grad = False
        
        # Next run one gradient descent step for pi.
        pi_optimizer[agent_nbr].zero_grad()
        loss_pi, pi_info = compute_loss_pi(agent_nbr, data)
        loss_pi.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ac[agent_nbr].pi.parameters(), max_grad_norm)
        pi_optimizer[agent_nbr].step()

        # Unfreeze Q-networks so you can optimize it at next step.
        for p in q_params[agent_nbr]:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac[agent_nbr].parameters(), ac_targ[agent_nbr].parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                if bootstrapping_loss_thresh < 0 or torch.sqrt(loss_q) < bootstrapping_loss_thresh: # TODO Let's try updating only when we're kinda done bootstrapping the latest changes - otherwise we just decay the target.
                    p_targ.data.add_((1 - polyak) * p.data)
        ac[agent_nbr].eval()
        ac_targ[agent_nbr].eval()

    def get_action(o, deterministic=False, sample_new_noise=True):
        return np.array([ac[i].act({k_o: torch.as_tensor(np.array([v_o]), dtype=torch.float32, device=cuda_device) for k_o, v_o in o[i].items()} 
                                    if dict_space else torch.as_tensor(np.array([o[i]]), dtype=torch.float32, device=cuda_device), 
                                    deterministic=deterministic, 
                                    sample_new_noise=sample_new_noise[i] if isinstance(sample_new_noise, list) else sample_new_noise)[0]
                        for i in range(n_agents)])

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), [False] * n_agents, np.zeros(n_agents), 0
            while not(any(d) or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += np.array(r)
                ep_len += 1
            logger.store(TestEpRet=np.average(ep_ret), TestEpLen=ep_len)
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_ret_ext, ep_ret_int, ep_len = env.reset(), np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents), 0
    hidden_ep_ret = np.zeros(n_agents) if hasattr(env, "get_hidden_step_rewards") else None
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o, sample_new_noise=sample_new_noise)
            if use_interactive_entropy_source:
                sample_new_noise = env.get_param_value("sample_new_noise") >= 0
        else:
            a = np.array([space.sample() for space in env.action_space])

        # Step the env
        o2, r, d, _ = env.step(a)

        if not split_reward_streams:
            ep_ret += np.array(r)
            if live_plot and (t+1) % steps_per_epoch != 0:
                plotter.add_entry(entry=np.average(r), plot="Rew")
        else:
            r_e = r[0]
            r_i = r[1]
            ep_ret += np.array(r_e) + np.array(r_i)
            ep_ret_ext += np.array(r_e)
            ep_ret_int += np.array(r_i)
            if live_plot and (t+1) % steps_per_epoch != 0:
                plotter.add_entry(entry=np.average(r_e), plot="ExtRew")
                plotter.add_entry(entry=np.average(r_i), plot="IntRew")
        
        live_plot_needs_update = live_plot and (t+1) % steps_per_epoch != 0
        compute_amp_step_return = (amp_step_return_required or live_plot_needs_update) and hasattr(env, "get_amp_obs")
        amp_step_rewards = None
        if compute_amp_step_return:
            prev_amp_discriminator_rewards = []
            for i in range(n_agents):
                if hasattr(replay_buffers[i], "discriminator_predict") and hasattr(env, "get_prev_state_amp_agent"):
                    state_agent = env.get_prev_state_amp_agent(i)
                    if state_agent is not None:
                        prev_amp_discriminator_rewards.append(replay_buffers[i].discriminator_predict(state_agent).detach().cpu().numpy())
            amp_step_rewards = prev_amp_discriminator_rewards
        avg_hidden_step_rewards = None
        if hasattr(env, "get_hidden_step_rewards"):
            if env.get_hidden_step_rewards() is not None:
                hidden_ep_ret += env.get_hidden_step_rewards()[:,0] if split_reward_streams else env.get_hidden_step_rewards() 
                avg_hidden_step_rewards = np.average(env.get_hidden_step_rewards())

        if live_plot_needs_update:
            if amp_step_rewards is not None:
                plotter.add_entry(entry=np.average(amp_step_rewards), plot="ImitationRew")
            if avg_hidden_step_rewards is not None:
                plotter.add_entry(entry=avg_hidden_step_rewards, plot="HiddenStepRew")
            plotter.plot()

        # TODO test fixed amp rewards.
        #if amp_step_rewards is not None:
        #   for i in range(n_agents):
        #       r[i] += amp_step_rewards[i]

        ep_len += 1

        if render_steps:
            # TODO crashes on env.close() 
            env.render()

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = [False] * n_agents if ep_len == max_ep_len else d # TODO I'm not sure if this makes sense, but fine...

        # Store experience to replay buffer
        for i in range(n_agents):
            hidden_ep_rew = env.get_hidden_step_rewards()[i,0] if split_reward_streams else env.get_hidden_step_rewards()[i] if hasattr(env, "get_hidden_step_rewards") else None
            if hasattr(env, "get_amp_obs"):
                amp_obs_stack = env.get_amp_obs(i)
                if amp_obs_stack is not None:
                    hidden_ep_rew = amp_step_rewards[i] + (hidden_ep_rew if hidden_ep_rew is not None else 0)
                    state_amp_agent = [amp_obs["state_amp_agent"] for amp_obs in amp_obs_stack]
                    state_amp_expert = [amp_obs["state_amp_expert"] for amp_obs in amp_obs_stack]
                    replay_buffers[i].store(o[i], a[i], r[i], o2[i], d[i], state_amp_agent, state_amp_expert, hidden_ep_rew=hidden_ep_rew)
            else:
                replay_buffers[i].store(o[i], a[i], r[i], o2[i], d[i], hidden_ep_rew=hidden_ep_rew)

            if her_k > 0:
                for transition in env.get_her_transitions(i):
                	replay_buffers[i].store(*transition)

        rew_for_meta_optimizer = np.average(r)
        if amp_step_rewards is not None:
            rew_for_meta_optimizer += np.average(amp_step_rewards)
        if avg_hidden_step_rewards is not None:
            rew_for_meta_optimizer += avg_hidden_step_rewards
        if split_reward_streams:
            alpha, gamma, gamma_intrinsic, lr, polyak, q_grad_penalty, imitation_lambda = meta_optimizer.update_hyperparameters(step_counter=t, step_reward=rew_for_meta_optimizer)
        else:
            alpha, gamma, lr, polyak, q_grad_penalty, imitation_lambda = meta_optimizer.update_hyperparameters(step_counter=t, step_reward=rew_for_meta_optimizer)
        alpha /= np.abs(max_entropy_bonus)
        
        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if np.any(d) or (ep_len == max_ep_len):

            for i in range(n_agents):
                if max_ep_ret[i] < ep_ret[i]:
                    max_ep_ret[i] = ep_ret[i]

            logger.store(EpRet=np.average(ep_ret), EpRetExternal=np.average(ep_ret_ext), EpRetInternal=np.average(ep_ret_int), EpLen=ep_len)
            o, ep_ret, ep_ret_ext, ep_ret_int, ep_len = env.reset(), np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents), 0
            if hidden_ep_ret is not None:
                logger.store(HiddenEpRet=np.average(hidden_ep_ret))
                hidden_ep_ret = np.zeros(n_agents)

            if intrinsic_reward_weight > 0 and episodic_curiosity < 1.0:
                # We need to allow bootstrapping of intrinsic values between episodes.
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
            for j in range(int(np.ceil(update_every * update_to_step_ratio))):
                for i in range(n_agents):
                    batch = replay_buffers[i].sample_batch(batch_size, imitation_lambda=imitation_lambda)
                    for k in range(len(intrinsic_reward_generators)):
                        int_rew_gen = intrinsic_reward_generators[k]
                        obs_batch = batch['obs']
                        if split_reward_streams:
                            batch['rew'][:,k+1] = int_rew_gen.compute_intrinsic_rewards(i, obs_batch)
                        else:
                            batch['rew'] += int_rew_gen.compute_intrinsic_rewards(i, obs_batch)
                    update(agent_nbr=i, data=batch)
                    replay_buffers[i].update_prev_batch()

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
            logger.log_tabular('HiddenEpRet', average_only=True)
            logger.log_tabular('EpRetExternal', with_min_and_max=True)
            logger.log_tabular('EpRetInternal', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('alpha', alpha)
            logger.log_tabular('gamma', gamma)
            logger.log_tabular('imitation_lambda', imitation_lambda)
            logger.log_tabular('lr', lr)
            logger.log_tabular('polyak', polyak)
            logger.log_tabular('q_grad_penalty', q_grad_penalty)
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

# TODO Let's see what a gradient penalty on the q estimate does
q_obs_inputs_with_grad = None
q_act_inputs_with_grad = None

def q_grad_penalty_loss(inputs, outputs):
    if type(inputs) is dict:
        gradients = [torch.autograd.grad(outputs=outputs, inputs=inputs[k],
                                        grad_outputs=torch.ones_like(outputs),
                                        create_graph=True, retain_graph=True,
                                        allow_unused=True)[0] for k in inputs]
        gradient_norm = torch.cat([torch.flatten(torch.sum(grads**2, dim=-1)) for grads in gradients], dim=-1)
        return gradient_norm.mean() / len(inputs)
    else:
        grads = torch.autograd.grad(outputs=outputs, inputs=inputs,
                                        grad_outputs=torch.ones_like(outputs),
                                        create_graph=True, retain_graph=True,
                                        allow_unused=True)[0]
        gradient_norm = torch.flatten(torch.sum(grads**2, dim=-1))
        return gradient_norm.mean()

def assign_data_directly(from_tensor, to_tensor):
    with torch.no_grad(): # We need to tell pytorch's autograd to look away for a second.
        if type(from_tensor) is dict:
            for k in from_tensor:
                to_tensor[k].data = from_tensor[k].data
                to_tensor[k].grad.zero_()
        else:
            to_tensor.data = from_tensor.data
            to_tensor.grad.zero_()
  

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--env_args', type=dict, default=None)
    parser.add_argument('--env_import', type=str, default=None)
    parser.add_argument('--env_tools_kwargs', type=dict, default=None)
    parser.add_argument('--ac_kwargs', type=dict, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--seed', '-s', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    if args.env_import is not None:
        # In case the user wants to import some special package like minerl that registers itself as a gym env, for example.
        __import__(args.env_import)
    
    sac(lambda : gym.make(args.env) if args.env_args is None else gym.make(args.env, **args.env_args), 
        env_tools_kwargs=args.env_tools_kwargs, ac_kwargs=args.ac_kwargs,
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
