import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.tf1.sac import core
from spinup.algos.tf1.sac.core import get_vars
from spinup.utils.logx import EpochLogger
from spinup.replay_buffers.amp_replay_buffer_tf1 import AMPReplayBuffer


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim_obj, act_dim, size):

        self.dict_space = False
        if np.isscalar(obs_dim_obj):
            obs_dim = [obs_dim_obj]
        elif isinstance(obs_dim_obj, gym.spaces.Dict):
            obs_dim = np.concatenate([sub_obs.shape for sub_obs in obs_dim_obj.spaces.values()], axis=-1)
            self.dict_space = True
        else:
            obs_dim = obs_dim_obj.shape

        self.obs1_buf = np.zeros(np.concatenate([[size], obs_dim]), dtype=np.float32)
        self.obs2_buf = np.zeros(np.concatenate([[size], obs_dim]), dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.cur_idxs = []
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs if not self.dict_space else np.concatenate([sub_obs for sub_obs in obs.values()], axis=-1)
        self.obs2_buf[self.ptr] = next_obs if not self.dict_space else np.concatenate([sub_obs for sub_obs in next_obs.values()], axis=-1)
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        self.cur_idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[self.cur_idxs],
                    obs2=self.obs2_buf[self.cur_idxs],
                    acts=self.acts_buf[self.cur_idxs],
                    rews=self.rews_buf[self.cur_idxs],
                    done=self.done_buf[self.cur_idxs])

    def to_feed_dict(self, tf_ph, obs):
        # TODO for now, the replay buffer simply concatenates the observations. So let's have it generate suitable feed dicts.
        return {tf_ph["state"]:obs}
    
    def update_prev_batch(self, new_priorities=None):
        pass


def sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=10000, epochs=10000, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-4, alpha=0.004, batch_size=256, start_steps=1000, 
        update_after=1000, update_every=1, num_test_episodes=0, max_ep_len=1000,
        dynamic_skip = True,
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.

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
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    #TODO multithreading is a problem with deepmimic at the moment
    #env, test_env = env_fn(), env_fn()
    env = env_fn()

    if dynamic_skip:
        from spinup.env_wrappers.dynamic_skip_env import DynamicSkipEnv
        env = DynamicSkipEnv(env)
        #if num_test_episodes > 0:
        #    test_env = DynamicSkipEnv(test_env)
    test_env = env

    # Let's make sure that every incoming env can be treated as a multi agent env.
    if not type(env.observation_space) is list:
        from spinup.env_wrappers.single_agent_env import SingleAgentEnv
        env = SingleAgentEnv(env)
        if num_test_episodes > 0:
            test_env = SingleAgentEnv(test_env)

    n_agents = len(env.observation_space)

    obs_dim = [space for space in env.observation_space]
    act_dim = [space.shape[0] for space in env.action_space]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = [space.high[0] for space in env.action_space]

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, [None] * n_agents, [None] * n_agents)

    # Main outputs from computation graph
    replay_buffer = []
    step_ops = []
    target_init = []
    mu, pi, q1, q2 = [], [], [], []
    for i in range(n_agents):
        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = env.action_space[i]
        with tf.variable_scope('main' + str(i)):
            mu_, pi_, logp_pi, q1_, q2_ = actor_critic(x_ph[i], a_ph[i], **ac_kwargs)
            mu.append(mu_)
            pi.append(pi_)
            q1.append(q1_)
            q2.append(q2_)

        with tf.variable_scope('main' + str(i), reuse=True):
            # compose q with pi, for pi-learning
            _, _, _, q1_pi, q2_pi = actor_critic(x_ph[i], pi_, **ac_kwargs)

            # get actions and log probs of actions for next states, for Q-learning
            _, pi_next, logp_pi_next, _, _ = actor_critic(x2_ph[i], a_ph[i], **ac_kwargs)

        # Target value network
        with tf.variable_scope('target' + str(i)):
            # target q values, using actions from *current* policy
            _, _, _, q1_targ, q2_targ = actor_critic(x2_ph[i], pi_next, **ac_kwargs)

        # Experience buffer
        #replay_buffer.append(ReplayBuffer(obs_dim_obj=obs_dim[i], act_dim=act_dim[i], size=replay_size))
        spinup_replay_buffer = ReplayBuffer(obs_dim_obj=obs_dim[i], act_dim=act_dim[i], size=replay_size)
        #TODO make AMP optional
        replay_buffer.append(AMPReplayBuffer(inner_replay_buffer=spinup_replay_buffer, env_reward_weight=0.5, amp_env=env, logger=logger))

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['main' + str(i) + '/pi', 'main' + str(i) + '/q1', 'main' + str(i) + '/q2', 'main' + str(i)])
        print('\nNumber of parameters agent ' + str(i) + ': \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi, q2_pi)
        min_q_targ = tf.minimum(q1_targ, q2_targ)


        # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
        q_backup = tf.stop_gradient(r_ph[i] + gamma*(1-d_ph[i])*(min_q_targ - alpha * logp_pi_next))

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_)**2)
        value_loss = q1_loss + q2_loss

        # Policy train op 
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main' + str(i) + '/pi'))

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        value_params = get_vars('main' + str(i) + '/q')
        with tf.control_dependencies([train_pi_op]):
            train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_value_op]):
            target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                      for v_main, v_targ in zip(get_vars('main' + str(i)), get_vars('target' + str(i)))])

        # All ops to call during one training step
        step_ops.append([pi_loss, q1_loss, q2_loss, q1_, q2_, logp_pi, 
                    train_pi_op, train_value_op, target_update])

        # Initializing targets to match main variables
        target_init.append(tf.group([tf.assign(v_targ, v_main)
                                  for v_main, v_targ in zip(get_vars('main' + str(i)), get_vars('target' + str(i)))]))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    dict_space = False
    inputs, outputs = {}, {}
    for i in range(n_agents):
        # multi modality handling
        if isinstance(x_ph[i], dict):
            input_modalities = x_ph[i]
            dict_space = True
        else:
            input_modalities = {'':x_ph[i]}
        inputs.update({'x' + k + str(i):v for k,v in input_modalities.items()})
        inputs.update({'a' + str(i): a_ph[i]})
        outputs.update({'mu' + str(i): mu[i], 'pi' + str(i): pi[i], 'q1' + str(i): q1[i], 'q2' + str(i): q2[i]})

    logger.setup_tf_saver(sess, inputs=inputs, 
                                outputs=outputs)

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        feed_dict = {}
        for i in range(n_agents):
            if dict_space:
                feed_dict.update({x_ph[i][modality]: o[i][modality].reshape(1,-1) for modality in x_ph[i]})
            else:
                feed_dict.update({x_ph[i]:o[i].reshape(1,-1)})
        outputs = sess.run(act_op, feed_dict=feed_dict)
        return np.array([out[0] for out in outputs])

    def test_agent():
        if num_test_episodes == 0:
            return
        test_env.reset()
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, np.zeros(n_agents), 0
            while not(np.any(d) or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += np.array(r)
                ep_len += 1
            logger.store(TestEpRet=np.average(ep_ret), TestEpLen=ep_len)
        test_env.reset()

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), np.zeros(n_agents), 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy.
        if t > start_steps:
            a = get_action(o)
        else:
            a = np.array([space.sample() for space in env.action_space])
        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += np.array(r)
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = [False] if ep_len==max_ep_len else d

        # Store experience to replay buffer
        for i in range(n_agents):
            
            amp_obs = env.get_amp_obs(i)
            state_amp_agent = amp_obs["state_amp_agent"]
            state_amp_expert = amp_obs["state_amp_expert"]

            # TODO remove
            #print("\no[i]: " +  str(o[i]))
            #print("a[i]: " +  str(a[i]))
            #print("o2[i]: " +  str(o2[i]))
            #print("d[i]: " +  str(d[i]))
            #print("state_amp_agent: " +  str(state_amp_agent))
            #print("state_amp_expert: " +  str(state_amp_expert))

            replay_buffer[i].store(o[i], a[i], r[i], o2[i], d[i], state_amp_agent, state_amp_expert)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if np.any(d) or (ep_len == max_ep_len):
            # TODO I think it would be better if I could reset for agents individually. Maybe the env could do this internally.
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                feed_dict = {}
                for i in range(n_agents):
                    batch = replay_buffer[i].sample_batch(batch_size)
                    feed_dict.update(replay_buffer[i].to_feed_dict(x_ph[i], batch['obs1']))
                    feed_dict.update(replay_buffer[i].to_feed_dict(x2_ph[i], batch['obs2']))
                    feed_dict.update({
                                 a_ph[i]: batch['acts'],
                                 r_ph[i]: batch['rews'],
                                 d_ph[i]: batch['done'],
                                })
                outs = sess.run(step_ops, feed_dict)
                
                loss_pi, loss_q1, loss_q2, q1_vals, q2_vals, log_pi = [], [], [], [], [], []
                for i in range(n_agents):
                    loss_pi.append(outs[i][0])
                    loss_q1.append(outs[i][1])
                    loss_q2.append(outs[i][2])
                    q1_vals.append(outs[i][3])
                    q2_vals.append(outs[i][4])
                    log_pi.append(outs[i][5])
                logger.store(LossPi=np.average(loss_pi), LossQ1=np.average(loss_q1), LossQ2=np.average(loss_q2),
                             Q1Vals=np.average(q1_vals), Q2Vals=np.average(q2_vals), LogPi=np.average(log_pi))

        # End of epoch wrap-up
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
            logger.log_tabular('EpLen', average_only=True)
            if num_test_episodes > 0:
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True) 
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            if "LossAmp" in logger.epoch_dict:
                logger.log_tabular('LossAmp', average_only=True)
            if "AccExpertAMP" in logger.epoch_dict:
                logger.log_tabular('AccExpertAMP', average_only=True)
            if "AccAgentAMP" in logger.epoch_dict:
                logger.log_tabular('AccAgentAMP', average_only=True)
            if "AmpRew" in logger.epoch_dict:
                logger.log_tabular('AmpRew', average_only=True)
            if "AmpRewBatchMax" in logger.epoch_dict:
                logger.log_tabular('AmpRewBatchMax', average_only=True)
            if "AmpRewBatchMin" in logger.epoch_dict:
                logger.log_tabular('AmpRewBatchMin', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=None)
    parser.add_argument('--l', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    sac(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict() if args.hid is None or args.l is None else dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
