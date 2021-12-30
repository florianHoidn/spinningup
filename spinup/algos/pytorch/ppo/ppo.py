import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ppo.core as core
import spinup.models.pytorch.multi_modal_net as model
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from spinup.env_wrappers.env_wrapper_utils import wrap_envs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim_in, act_dim, size, cuda_device, gamma=0.99, lam=0.95):
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
        else:
            self.obs_buf = np.zeros(core.combined_shape(size, obs_dim_in.shape), dtype=np.float32)

        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.cuda_device = cuda_device

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        #assert self.ptr < self.max_size     # buffer has to have room so you can store
        if self.dict_space:
            for k in self.obs_modalities:
                self.obs_buf[k][self.ptr] = obs[k]
        else:
            self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        if self.dict_space:
            data = dict(obs={k:self.obs_buf[k] for k in self.obs_buf},
                        act=self.act_buf, ret=self.ret_buf,
                        adv=self.adv_buf, logp=self.logp_buf)
            return {k: {sub_k: torch.as_tensor(sub_v, dtype=torch.float32, device=self.cuda_device) for sub_k, sub_v in v.items()} 
                    if k.startswith("obs") else 
                    torch.as_tensor(v, dtype=torch.float32, device=self.cuda_device) for k,v in data.items()}
        else:  
            data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                        adv=self.adv_buf, logp=self.logp_buf)
            return {k: torch.as_tensor(v, dtype=torch.float32, device=self.cuda_device) for k,v in data.items()}


def ppo(env_fn, 
        actor_critic=core.ActorCritic, 
        ac_kwargs={"model":model.net, "model_kwargs_getter":model.get_tanh_kwargs}, 
        seed=None, 
        steps_per_epoch=4000,
        epochs=10000, 
        gamma=0.99, 
        clip_ratio=0.2, 
        pi_lr=3e-4,
        vf_lr=1e-3, 
        train_pi_iters=80, 
        train_v_iters=80, 
        lam=0.97, 
        max_ep_len=10000,
        target_kl=-1, #0.05,
        ent_coef=0.2,
        logger_kwargs=dict(), 
        save_freq=10, 
        render_steps=True, 
        cuda_device="cuda:0",
        env_tools_kwargs={},
        split_reward_streams=False, 
        imitation_reward_weight=0,
        normalize_env=False,
        stack_size=0,
        her_k=0,
        her_reward=10.0,
        signal_vector_size = 0,
        output_pickle_trajectories=None,
        max_skip=10,
        greedy_adversarial_priors=False,
        intrinsic_reward_weight=0,
        #restore_model_path="D:/ml_frameworks/spinningup/data/2021-04-24_BipedalWalkerHardcore/pyt_save/model.pt"):
        restore_model_path=None):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
        
        render_steps (bool): Whether to render the individual trining steps.
        
        cuda_device (string): Name of the cuda device that will be used. Default "cuda:0".

        restore_model_path (string): Path/to/model that will be restored to continue a training run. 
            None by default, meaning that a newly initialized model will be used.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    if seed is not None:
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
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

    n_agents = len(env.observation_space) # TODO I can't handle multi agent envs yet.
    
    obs_dim = [space for space in env.observation_space]
    act_dim = [space.shape[0] for space in env.action_space]
    dict_space = any([isinstance(o, gym.spaces.Dict) for o in obs_dim])
    
    # Create actor-critic module
    if restore_model_path is None:
        ac = [actor_critic(env.observation_space[i], env.action_space[i], **ac_kwargs) for i in range(n_agents)]
    else:
        # TODO careful, spinup does not use the recommended way of saving models in pytorch by saving just the state_dict (which doesn't depend on the projects current directory structure).
        # Also, it does not restore an entire checkpoint with optimizer variables etc. - so strictly speaking, I can't really use it to resume training.
        ac = torch.load(restore_model_path)

    for i in range(n_agents):
        if cuda_device is not None:
            ac[i].to(cuda_device)

        ac[i].eval()
    
        # Sync params across processes
        sync_params(ac[i])

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [ac[i].pi, ac[i].v])
        logger.log('\nNumber of parameters agent {}: \t pi: {}, \t v: {}\n'.format(i, *var_counts))

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = [PPOBuffer(obs_dim[i], act_dim[i], local_steps_per_epoch, cuda_device, gamma, lam) for i in range(n_agents)]

    # Set up function for computing PPO policy loss
    def compute_loss_pi(agent_nbr, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy loss
        pi, logp = ac[agent_nbr].pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        # TODO why is ther no entropy coefficient being used? I'll try using one
        loss_pi -= pi.entropy().mean() * ent_coef

        if signal_vector_size > 0:
            # Inspired by VICReg regularization for embedding vectors.
            if hasattr(env, "get_signal_from_to_idx"):
                signal_from_to = env.get_signal_from_to_idx(agent_nbr)
                pi_sig = pi.mean[:,signal_from_to[0]:signal_from_to[1]]
            else:
                pi_sig = pi.mean
            pi_norm = pi_sig - pi_sig.mean(dim=0)
            cov_pi = (pi_sig.T @ pi_sig) / local_steps_per_epoch
            off_diag_elems = torch.triu(cov_pi, diagonal=1)
            decorelation_loss = off_diag_elems.pow(2).sum() / act_dim[agent_nbr]
            loss_pi += decorelation_loss

            # Let's also encourage high variance along the diagonal.
            #loss_pi -= torch.diagonal(cov_pi).pow(2).sum() / act_dim[agent_nbr]

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(agent_nbr, data):
        obs, ret = data['obs'], data['ret']
        return ((ac[agent_nbr].v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = [Adam(ac[i].pi.parameters(), lr=pi_lr) for i in range(n_agents)]
    vf_optimizer = [Adam(ac[i].v.parameters(), lr=vf_lr) for i in range(n_agents)]

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(agent_nbr):
        ac[agent_nbr].train()
        data = buf[agent_nbr].get()

        pi_l_old, pi_info_old = compute_loss_pi(agent_nbr, data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(agent_nbr, data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer[agent_nbr].zero_grad()
            loss_pi, pi_info = compute_loss_pi(agent_nbr, data)
            kl = mpi_avg(pi_info['kl'])
            if target_kl >= 0 and kl > 1.5 * target_kl:
                logger.log(f'Early stopping at step {i} due to reaching max kl (kl: {kl}).')
                break
            loss_pi.backward()
            mpi_avg_grads(ac[agent_nbr].pi)    # average grads across MPI processes
            pi_optimizer[agent_nbr].step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer[agent_nbr].zero_grad()
            loss_v = compute_loss_v(agent_nbr, data)
            loss_v.backward()
            mpi_avg_grads(ac[agent_nbr].v)    # average grads across MPI processes
            vf_optimizer[agent_nbr].step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
        ac[agent_nbr].eval()

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), np.zeros(n_agents), np.zeros(n_agents)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            actions, vs, logps = [], [], []
            for i in range(n_agents):
                a, v, logp = ac[i].step({k_o: torch.as_tensor(v_o, dtype=torch.float32, device=cuda_device) for k_o, v_o in o[i].items()} 
                    if dict_space else torch.as_tensor(o[i], dtype=torch.float32, device=cuda_device))
                actions.append(a)
                vs.append(v)
                logps.append(logp)

            next_o, r, d, _ = env.step(np.array(actions))
            for i in range(n_agents):
                ep_ret[i] += r[i]
                ep_len[i] += 1

            if render_steps:
                env.render()

            # save and log
            for i in range(n_agents):
                buf[i].store(o[i], actions[i], r[i], vs[i], logps[i])
                logger.store(VVals=vs[i])
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = any(d) or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    vs = []
                    for i in range(n_agents):
                        _, v, _ = ac[i].step({k_o: torch.as_tensor(v_o, dtype=torch.float32, device=cuda_device) for k_o, v_o in o[i].items()} 
                                            if dict_space else torch.as_tensor(o[i], dtype=torch.float32, device=cuda_device))
                        vs.append(v)
                    if render_steps:
                        env.render()
                else:
                    vs = np.zeros(n_agents)
                
                for i in range(n_agents):
                    buf[i].finish_path(vs[i])
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=np.average(ep_ret), EpLen=np.average(ep_len))
                o, ep_ret, ep_len = env.reset(), np.zeros(n_agents), np.zeros(n_agents)


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        for i in range(n_agents):
            update(i)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)