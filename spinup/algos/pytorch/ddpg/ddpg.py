from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ddpg.core as core
import spinup.models.pytorch.mlp as model
from spinup.utils.logx import EpochLogger

from spinup.env_wrappers.env_wrapper_utils import wrap_envs

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim_in, act_dim, size, cuda_device):
        if np.isscalar(obs_dim_in):
            obs_dim = [obs_dim_in]
        elif isinstance(obs_dim_in, gym.spaces.Dict):
            obs_dim = np.concatenate([sub_obs.shape for sub_obs in obs_dim_in.spaces.values()], axis=-1)
        else:
            obs_dim = obs_dim_in.shape

        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.cur_idxs = []
        self.ptr, self.size, self.max_size = 0, 0, size
        self.cuda_device = cuda_device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs if not isinstance(obs, dict) else np.concatenate([sub_obs for sub_obs in obs.values()], axis=-1)
        self.obs2_buf[self.ptr] = next_obs if not isinstance(next_obs, dict) else np.concatenate([sub_obs for sub_obs in next_obs.values()], axis=-1)
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        self.cur_idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[self.cur_idxs],
                     obs2=self.obs2_buf[self.cur_idxs],
                     act=self.act_buf[self.cur_idxs],
                     rew=self.rew_buf[self.cur_idxs],
                     done=self.done_buf[self.cur_idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.cuda_device) for k,v in batch.items()}

    def update_prev_batch(self, new_priorities=None):
        # TODO maybe I should remove this, I don't really need it and could get this to work properly yet.
        pass


def ddpg(env_fn, actor_critic=core.ActorCritic, seed=None, ac_kwargs={"model":model.net, "model_kwargs_getter":model.get_default_kwargs}, 
         env_wrapper_kwargs={},
         steps_per_epoch=4000, epochs=100, replay_size=int(5e4), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=1000, #10000, 
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1, render_steps=True, cuda_device="cuda:0",
        #restore_model_path="D:/ml_frameworks/spinningup/data/2021-04-24_BipedalWalkerHardcore/pyt_save/model.pt"):
        restore_model_path=None):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

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

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

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

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = env_fn()
    if num_test_episodes > 0:
        test_env = env_fn()
    else:
        test_env = None

    env, test_env, _ = wrap_envs(env=env, test_env=test_env, **env_wrapper_kwargs)

    #env, test_env = env_fn(), env_fn()
    ## Let's make sure that every incoming env can be treated as a multi agent env.
    #if not type(env.observation_space) is list:
    #    from spinup.env_wrappers.single_agent_env import SingleAgentEnv
    #    env = SingleAgentEnv(env)
    #    if num_test_episodes > 0:
    #        test_env = SingleAgentEnv(test_env)

    n_agents = len(env.observation_space)

    obs_dim = [space for space in env.observation_space]
    act_dim = [space.shape[0] for space in env.action_space]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = [space.high[0] for space in env.action_space]

    # Create actor-critic module and target networks
    if restore_model_path is None:
        ac = [actor_critic(env.observation_space[i], env.action_space[i], **ac_kwargs) for i in range(n_agents)]
    else:
        # TODO careful, spinup does not use the recommended way of saving models in pytorch by saving just the state_dict (which doesn't depend on the projects current directory structure).
        # Also, it does not restore an entire checkpoint with optimizer variables etc. - so strictly speaking, I can't really use it to resume training.
        ac = [torch.load(restore_model_path) for i in range(n_agents)]
    ac_targ = deepcopy(ac)

    if cuda_device is not None:
        for i in range(n_agents):
            ac[i].to(cuda_device)
            ac_targ[i].to(cuda_device)

    replay_buffers = []
    for i in range(n_agents):
        ac[i].eval()
        ac_targ[i].eval()

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ[i].parameters():
            p.requires_grad = False

        # Experience buffer
        spinup_replay_buffer = ReplayBuffer(obs_dim_in=obs_dim[i], act_dim=act_dim[i], size=replay_size, cuda_device=cuda_device)
        replay_buffers.append(spinup_replay_buffer)
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [ac[i].pi, ac[i].q])
        logger.log("\nNumber of parameters agent " + str(i) + ": \t pi: %d, \t q: %d\n"%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(agent_nbr, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac[agent_nbr].q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ[agent_nbr].q(o2, ac_targ[agent_nbr].pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(agent_nbr, data):
        o = data['obs']
        q_pi = ac[agent_nbr].q(o, ac[agent_nbr].pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = []
    q_optimizer = []
    for i in range(n_agents):
        pi_optimizer.append(Adam(ac[i].pi.parameters(), lr=pi_lr))
        q_optimizer.append(Adam(ac[i].q.parameters(), lr=q_lr))

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(agent_nbr, data):
        ac[agent_nbr].train()
        ac_targ[agent_nbr].train()

        # First run one gradient descent step for Q.
        q_optimizer[agent_nbr].zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer[agent_nbr].step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer[agent_nbr].zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer[agent_nbr].step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac[agent_nbr].q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac[agent_nbr].parameters(), ac_targ[agent_nbr].parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        ac[agent_nbr].eval()
        ac_targ[agent_nbr].eval()

    def get_action(o, noise_scale):
        # TODO It would be better if the actor critic let pytorch deal with the multi agent input (in parallel on the gpu).
        a = np.array([ac[i].act({k_o: torch.as_tensor(v_o, dtype=torch.float32, device=cuda_device) for k_o, v_o in o[i].items()} 
                                    if dict_space else torch.as_tensor(o[i], dtype=torch.float32, device=cuda_device), deterministic)
                        for i in range(n_agents)])
        a += np.array([noise_scale * np.random.randn(act_dim[i]) for i in range(n_agents)])
        return np.clip(a, -act_limit, act_limit)

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
    o, ep_ret, ep_ret_ext, ep_ret_int, ep_len = env.reset(), np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents), 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = np.array([space.sample() for space in env.action_space])

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if render_steps:
            env.render()

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        #d = False if ep_len==max_ep_len else d # TODO here too, I'm not sure if the spinup author really wanted to do this.

        # Store experience to replay buffer
        for i in range(n_agents):
            replay_buffers[i].store(o[i], a[i], r[i], o2[i], d[i])

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if np.any(d) or (ep_len == max_ep_len):
            logger.store(EpRet=np.average(ep_ret), EpRetExternal=np.average(ep_ret_ext), EpRetInternal=np.average(ep_ret_int), EpLen=ep_len)
            o, ep_ret, ep_ret_ext, ep_ret_int, ep_len = env.reset(), np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents), 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                for i in range(n_agents):
                    batch = replay_buffer[i].sample_batch(batch_size)
                    update(agent_nbr=i, data=batch)

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
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
