import numpy as np
import tensorflow as tf

from spinup.replay_buffers.amp_utils.deep_mimic_buffer import DeepMimicStyleBuffer
from spinup.replay_buffers.amp_utils.deep_mimic_normalizer import DeepMimicStyleNormalizer

class AMPReplayBuffer:
    """
    A replay buffer that dynamically adds an imitation bonus to the rewards from the environment.
    The imitation bonus is based on adversarial motion priors (AMP) as described in
    https://xbpeng.github.io/projects/AMP/2021_TOG_AMP.pdf and implemted in UC Berkeley's DeepMimic
    project, https://github.com/xbpeng/DeepMimic.
    """
    def __init__(self, inner_replay_buffer, env_reward_weight, amp_env, logger):
        self.replay_buffer = inner_replay_buffer # Let's wrap one of spinup's replay buffers. 
        self.amp_obs_buf = np.zeros([self.replay_buffer.max_size, amp_env.get_amp_obs_size()], dtype=np.float32)
        self.amp_discriminator = AMPDiscriminator(env_reward_weight, amp_env, logger)
        self.update_counter = 0

    def store(self, obs, act, rew, next_obs, done, state_amp_agent, state_amp_expert):
        self.replay_buffer.store(obs, act, rew, next_obs, done)

        self.amp_obs_buf[self.replay_buffer.ptr] = state_amp_agent
        self.amp_discriminator.add_amp_sample(state_amp_expert, state_amp_agent)

    def sample_batch(self, batch_size):
        batch = self.replay_buffer.sample_batch(batch_size)
        # TODO careful, spinup's replay buffers sometimes use "rew" and somtimes "rews" as key. I need to modularize and unify these buffers. 
        batch["rews"] = self.amp_discriminator.batch_calc_reward(self.amp_obs_buf[self.replay_buffer.cur_idxs], batch["rews"])
        
        # TODO use update_prev_batch instead.
        self.update_counter += 1
        #if self.update_counter % 100 == 0: # TODO don't know if this is a good idea
        #    self.amp_discriminator.update_disc()
        self.amp_discriminator.update_disc()

        #if self.update_counter % 5000 == 0: # TODO don't know if this is a good idea        
        #    self.amp_discriminator.update_normalizers()

        return batch

    #def update_prev_batch(self, new_priorities=None):
    #    self.replay_buffer.update_prev_batch(new_priorities)
    #    self.amp_discriminator.update_disc()
    #    self.amp_discriminator.update_normalizers()

    def to_feed_dict(self, tf_ph, obs):
        return self.replay_buffer.to_feed_dict(tf_ph, obs)

class AMPDiscriminator:
    MAIN_SCOPE = "main"
    DISC_SCOPE = "amp"
    DISC_LOGIT_NAME = "discriminator_logits"
    RESOURCE_SCOPE = "resource"
    SOLVER_SCOPE = "solvers"
    """
    An adversarial discriminator that is trained to distinguish between reference transitions and transitions
    produced by an agent. The implementation is based on DeepMimic's AMPAgent but isn't tied to a specific
    RL algorithm like PPO. 
    """
    def __init__(self, env_reward_weight, amp_env, logger, batchsize=256, steps_per_batch=1, expert_buffer_size=100000, agent_buffer_size=100000):

        self.tf_scope = "discriminator"
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        self.amp_env = amp_env
        amp_obs_size = amp_env.get_amp_obs_size()
        amp_obs_offset = amp_env.get_amp_obs_offset()
        amp_obs_scale = amp_env.get_amp_obs_scale()
        amp_obs_norm_group = amp_env.get_amp_obs_norm_group()

        with self.sess.as_default(), self.graph.as_default(), tf.variable_scope(self.tf_scope):
            with tf.variable_scope(self.RESOURCE_SCOPE):
                self._amp_obs_norm = DeepMimicStyleTFNormalizer(self.sess, "amp_obs_norm", amp_obs_size, amp_obs_norm_group)
                self._amp_obs_norm.set_mean_std(-amp_obs_offset, 1 / amp_obs_scale)
        
        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope(self.tf_scope):
                self.build_net(amp_obs_size)
                
                with tf.variable_scope(self.SOLVER_SCOPE):
                    self.build_losses()
                    self.build_solvers()

                tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)

                #self.sess.run(tf.global_variables_initializer()) # TODO should probably happen elsewhere.
                self.sess.run(tf.variables_initializer(tf_vars)) # TODO maybe that's a bit better.

                tf_vars = [v for v in tf_vars if '/' + self.SOLVER_SCOPE + '/' not in v.name]
                self.saver = tf.train.Saver(tf_vars, max_to_keep=0)

        # TODO in DeepMimic's tf_agent we would now initialize normalizers for s, g, and a. I should probably have those in my DeepMimicEnv - might be a bit annoying without tensorflow.

        self._task_reward_lerp = env_reward_weight
        self._disc_batchsize = batchsize
        self._disc_steps_per_batch = steps_per_batch
        self._disc_expert_buffer_size = expert_buffer_size
        self._disc_agent_buffer_size = agent_buffer_size

        self._disc_expert_buffer = DeepMimicStyleBuffer(self._disc_expert_buffer_size)
        self._disc_agent_buffer = DeepMimicStyleBuffer(self._disc_agent_buffer_size)
        self.logger = logger

    def build_net(self, amp_obs_size, disc_init_output_scale=1, reward_scale=1.0):

        # TODO do I even want to build a custom net here. I think it would be much nicer to stick with spinup's modular approach.
        
        disc_init_output_scale = disc_init_output_scale
        self._reward_scale = reward_scale

        # setup input tensors
        self._amp_obs_expert_ph = tf.placeholder(tf.float32, shape=[None, amp_obs_size], name="amp_obs_expert")
        self._amp_obs_agent_ph = tf.placeholder(tf.float32, shape=[None, amp_obs_size], name="amp_obs_agent")

        self._disc_expert_inputs = [self._amp_obs_norm.normalize_tf(self._amp_obs_expert_ph)] 
        self._disc_agent_inputs = [self._amp_obs_norm.normalize_tf(self._amp_obs_agent_ph)]

        with tf.variable_scope(self.MAIN_SCOPE):
            with tf.variable_scope(self.DISC_SCOPE):
                self._disc_logits_expert_tf = self.build_disc_net(self._disc_expert_inputs, disc_init_output_scale)
                self._disc_logits_agent_tf = self.build_disc_net(self._disc_agent_inputs, disc_init_output_scale, reuse=True)

        # TODO These seem to be computed for meta info only.
        #self._disc_prob_agent_tf = tf.sigmoid(self._disc_logits_agent_tf)
        #self._abs_logit_agent_tf = tf.reduce_mean(tf.abs(self._disc_logits_agent_tf))
        #self._avg_prob_agent_tf = tf.reduce_mean(self._disc_prob_agent_tf)

    def build_disc_net(self, input_tfs, init_output_scale, reuse=False):
        # TODO would be nice if this was configurable like the other spinup nets. Right now it's hard wired to DeepMimic's fc_2layers_1024units net (which seems to be what they always use for the discriminator)
        out_size = 1
        layers = [1024, 512]
        input_tf = tf.concat(axis=-1, values=input_tfs)
        for i, size in enumerate(layers):
            with tf.variable_scope(str(i), reuse=reuse):
                input_tf = tf.layers.dense(inputs=input_tf,
                                        units=size,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation = tf.nn.relu)
        logits_tf = tf.layers.dense(inputs=input_tf, units=out_size, activation=None, reuse=reuse,
                                kernel_initializer=tf.random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale),
                                name=self.DISC_LOGIT_NAME)
        return logits_tf

    #def build_losses(self, disc_weight_decay=0.0005, disc_logit_reg_weight=0.05, disc_grad_penalty=10):
    def build_losses(self, disc_weight_decay=0.0005, disc_logit_reg_weight=0.0005, disc_grad_penalty=0.5):
        
        disc_loss_expert_tf = 0.5 * tf.reduce_sum(tf.square(self._disc_logits_expert_tf - 1), axis=-1)
        disc_loss_agent_tf = 0.5 * tf.reduce_sum(tf.square(self._disc_logits_agent_tf + 1), axis=-1)
        disc_loss_expert_tf = tf.reduce_mean(disc_loss_expert_tf)
        disc_loss_agent_tf = tf.reduce_mean(disc_loss_agent_tf)

        self._disc_loss_tf = 0.5 * (disc_loss_agent_tf + disc_loss_expert_tf)

        # TODO This too seems to be meta info only.
        self._acc_expert_tf = tf.reduce_mean(tf.cast(tf.greater(self._disc_logits_expert_tf, 0), tf.float32))
        self._acc_agent_tf = tf.reduce_mean(tf.cast(tf.less(self._disc_logits_agent_tf, 0), tf.float32))
        
        if disc_weight_decay != 0:
            self._disc_loss_tf += disc_weight_decay * self.weight_decay_loss(self.tf_scope + '/' + self.MAIN_SCOPE + "/" + self.DISC_SCOPE)
                    
        if disc_logit_reg_weight != 0:
            self._disc_loss_tf += disc_logit_reg_weight * self.disc_logit_reg_loss()
        
        if disc_grad_penalty != 0:
            self._grad_penalty_loss_tf = self.disc_grad_penalty_loss(in_tfs=self._disc_expert_inputs, out_tf=self._disc_logits_expert_tf)
            self._disc_loss_tf += disc_grad_penalty * self._grad_penalty_loss_tf
        else:
            self._grad_penalty_loss_tf = tf.constant(0.0, dtype=tf.float32)

    def weight_decay_loss(self, scope):
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        vars_no_bias = [v for v in disc_vars if 'bias' not in v.name]
        loss = tf.add_n([tf.nn.l2_loss(v) for v in vars_no_bias])
        return loss

    #def build_solvers(self, disc_stepsize=0.0001, disc_momentum=0.9):
    def build_solvers(self, disc_stepsize=0.0001):
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.tf_scope + '/' + self.MAIN_SCOPE + "/" + self.DISC_SCOPE)
        #disc_opt = tf.train.MomentumOptimizer(learning_rate=disc_stepsize, momentum=disc_momentum)
        disc_opt = tf.train.AdamOptimizer(learning_rate=disc_stepsize)
        self._disc_grad_tf = tf.gradients(self._disc_loss_tf, disc_vars)
        #self._disc_solver = mpi_solver.MPISolver(self.sess, disc_opt, disc_vars) # TODO get rid of MPI for now and just use the optimizer.
        self.disc_optimizer_op = disc_opt.minimize(self._disc_loss_tf, var_list=disc_vars)

    def disc_logit_reg_loss(self):
        with self.sess.as_default(), self.graph.as_default():
            vars_tf = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.tf_scope + '/' + self.MAIN_SCOPE + "/" + self.DISC_SCOPE)
        logit_vars = [v for v in vars_tf if (self.DISC_LOGIT_NAME in v.name and "bias" not in v.name)]
        loss_tf = tf.add_n([tf.nn.l2_loss(v) for v in logit_vars])
        return loss_tf

    def disc_grad_penalty_loss(self, in_tfs, out_tf):
        grad_tfs = tf.gradients(ys=out_tf, xs=in_tfs)
        grad_tf = tf.concat(grad_tfs, axis=-1)
        norm_tf = tf.reduce_sum(tf.square(grad_tf), axis=-1)
        loss_tf = 0.5 * tf.reduce_mean(norm_tf)
        return loss_tf

    def save_model(self, out_path):
        with self.sess.as_default(), self.graph.as_default():
            try:
                save_path = self.saver.save(self.sess, out_path, write_meta_graph=False, write_state=False)
                print('AMP discriminator model saved to: ' + save_path)
            except:
                print("Failed to save AMP discriminator model to: " + save_path)

    def load_model(self, in_path):
        with self.sess.as_default(), self.graph.as_default():
            self.saver.restore(self.sess, in_path)
            self._amp_obs_norm.load()

    def update_normalizers(self):
        # TODO we need to call this in train
        # TODO also, if we want to update normalizers for s,g, and a, we also need to do this whenever this is called.
        self._amp_obs_norm.update()

    def add_amp_sample(self, new_reference_state, new_agent_state):
        # These AMP observations are contained in the observation from the environment.
        # They contain one new reference/expert state and one state that was produced by the agent.
        self._amp_obs_norm.record(new_reference_state)
        self._amp_obs_norm.record(new_agent_state)
        self._disc_expert_buffer.store([new_reference_state])
        self._disc_agent_buffer.store([new_agent_state])

        # Let's plot the current discriminator reward for the agent state.
        # TODO The buffer would need an agent ID to do this properly.
        self.amp_env.deep_mimic_env.log_val(0, self.calc_disc_reward([new_agent_state])[0])

    def update_disc(self):
        #info = None
        
        # TODO DeepMimic performs lots of training steps for each update. I'll only do one for now.
        #for b in range(num_steps):
        disc_expert_batch = self._disc_expert_buffer.sample(self._disc_batchsize)
        obs_expert = self._disc_expert_buffer.get(disc_expert_batch)
        
        disc_agent_batch = self._disc_agent_buffer.sample(self._disc_batchsize)
        obs_agent = self._disc_agent_buffer.get(disc_agent_batch)

        #cur_info = self.step_disc(obs_expert=obs_expert, obs_agent=obs_agent)
        self.step_disc(obs_expert=obs_expert, obs_agent=obs_agent)

        #if info is None:
        #    info = cur_info
        #else:
        #    for k, v in cur_info.items():
        #        info[k] += v
        #
        #return info

    def step_disc(self, obs_expert, obs_agent):
        feed = {
            self._amp_obs_expert_ph: obs_expert,
            self._amp_obs_agent_ph: obs_agent,
        }

        #run_tfs = [self._disc_grad_tf, self._disc_loss_tf, self._acc_expert_tf, self._acc_agent_tf,
        #           self._avg_prob_agent_tf, self._abs_logit_agent_tf, self._grad_penalty_loss_tf, 
        #           self.disc_optimizer_op] # TODO let's see if this works.
        run_tfs = [self._disc_grad_tf, self._disc_loss_tf, self._acc_expert_tf, self._acc_agent_tf, self._grad_penalty_loss_tf, 
                   self.disc_optimizer_op] # TODO let's see if this works.
        results = self.sess.run(run_tfs, feed)

        self.logger.store(LossAmp=results[1])
        self.logger.store(AccExpertAMP=results[2])
        self.logger.store(AccAgentAMP=results[3])
        
        #grads = results[0]
        #self._disc_solver.update(grads)

        #info = {
        #    "loss": results[1],
        #    #"acc_expert": results[2],
        #    #"acc_agent": results[3],
        #    #"prob_agent": results[4],
        #    #"abs_logit": results[5],
        #    "grad_penalty": results[6],
        #}

        #return info


    def batch_calc_reward(self, obs_agent_amp_batch, rewards_batch):
        disc_r, _ = self.calc_disc_reward(obs_agent_amp_batch)

        self.logger.store(AmpRew=np.average(disc_r))
        self.logger.store(AmpRewBatchMax=np.amax(disc_r))
        self.logger.store(AmpRewBatchMin=np.amin(disc_r))

        return (1.0 - self._task_reward_lerp) * disc_r + self._task_reward_lerp * rewards_batch

    def calc_disc_reward(self, amp_obs):
        feed = {
            self._amp_obs_agent_ph: amp_obs,
        }
        logits = self.sess.run(self._disc_logits_agent_tf, feed_dict=feed)
        r = 1.0 - 0.25 * np.square(1.0 - logits)
        r = np.maximum(r, 0.0)
        r = r[:, 0]
        return r, logits


class DeepMimicStyleTFNormalizer(DeepMimicStyleNormalizer):
    """
    This normalizer is more or less a copy of DeepMimic's TFNormalizer and is meant for internal use by the
    AMP discriminator in an AMPReplayBuffer.
    """
    def __init__(self, sess, scope, size, groups_ids=None, eps=0.02, clip=np.inf):
        self.sess = sess
        self.scope = scope
        super().__init__(size, groups_ids, eps, clip)

        with tf.variable_scope(self.scope):
            self.build_resource_tf()
        return

    # initialze count when loading saved values so that things don't change to quickly during updates
    def load(self):
        self.count = self.count_tf.eval()[0]
        self.mean = self.mean_tf.eval()
        self.std = self.std_tf.eval()
        self.mean_sq = self.calc_mean_sq(self.mean, self.std)
        return

    def update(self):
        super().update()
        self.update_resource_tf()
        return

    def set_mean_std(self, mean, std):
        super().set_mean_std(mean, std)
        self.update_resource_tf()
        return

    def normalize_tf(self, x):
        norm_x = (x - self.mean_tf) / self.std_tf
        norm_x = tf.clip_by_value(norm_x, -self.clip, self.clip)
        return norm_x

    def unnormalize_tf(self, norm_x):
        x = norm_x * self.std_tf + self.mean_tf
        return x
    
    def build_resource_tf(self):
        self.count_tf = tf.get_variable(dtype=tf.int32, name='count', initializer=np.array([self.count], dtype=np.int32), trainable=False)
        self.mean_tf = tf.get_variable(dtype=tf.float32, name='mean', initializer=self.mean.astype(np.float32), trainable=False)
        self.std_tf = tf.get_variable(dtype=tf.float32, name='std', initializer=self.std.astype(np.float32), trainable=False)
        
        self.count_ph = tf.get_variable(dtype=tf.int32, name='count_ph', shape=[1])
        self.mean_ph = tf.get_variable(dtype=tf.float32, name='mean_ph', shape=self.mean.shape)
        self.std_ph = tf.get_variable(dtype=tf.float32, name='std_ph', shape=self.std.shape)
        
        self._update_op = tf.group(
            self.count_tf.assign(self.count_ph),
            self.mean_tf.assign(self.mean_ph),
            self.std_tf.assign(self.std_ph)
        )
        return

    def update_resource_tf(self):
        feed = {
            self.count_ph: np.array([self.count], dtype=np.int32),
            self.mean_ph: self.mean,
            self.std_ph: self.std
        }
        self.sess.run(self._update_op, feed_dict=feed)
        return