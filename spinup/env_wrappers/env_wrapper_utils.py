import gym.spaces
import os

def wrap_envs(env, test_env, env_tools_kwargs={}, output_pickle_trajectories=None, imitation_reward_weight=0, normalize=False, max_skip=10,
            signal_vector_size=0, stack_size=0, greedy_adversarial_priors=False, intrinsic_reward_weight=0, split_reward_streams=False, 
            cuda_device="cuda:0", model=None, restore_model_path=None, render_steps=False, 
            use_interactive_entropy_source=False, logger=None, her_k=0, her_reward=10):
    
    if output_pickle_trajectories is not None:
        from spinup.env_wrappers.trajectory_pickle_env import TrajectoryPickleEnv
        env = TrajectoryPickleEnv(env, output_pickle_trajectories)
        if test_env != None:
            test_env = TrajectoryPickleEnv(test_env, output_pickle_trajectories)
    intrinsic_reward_generators = [] # A list of objects like the RndCuriosityEnv that implement compute_intrinsic_rewards.
    if not type(env.observation_space) is list:
        from spinup.env_wrappers.single_agent_env import SingleAgentEnv
        env = SingleAgentEnv(env)
        if test_env != None:
            test_env = SingleAgentEnv(test_env)
    if any([isinstance(space, gym.spaces.Dict) for space in env.action_space]):
        from spinup.env_wrappers.unpack_action_dict_env import UnpackActionDictEnv
        env = UnpackActionDictEnv(env)
        if test_env != None:
            test_env = UnpackActionDictEnv(test_env)
    if imitation_reward_weight > 0:
        from spinup.replay_buffers.amp_replay_buffer_pytorch import AMPReplayBuffer
        # TODO the AMP env would be a good candidate for an intrinsic_reward_generator - for now, I'll try to stay true to the way it's done in DeepMimic, though.
        if not hasattr(env, "get_amp_obs"):
            from spinup.env_wrappers.amp_env import AmpEnv
            from spinup.game_specific_utils.env_specific_tools import select_demo_provider
            demo_provider = select_demo_provider(env, env_tools_kwargs)
            if demo_provider is not None:
                env = AmpEnv(env, demo_provider)
    if normalize:
        act_mean, act_var = None, None
        if hasattr(env, "get_action_mean") and hasattr(env, "get_action_var"):
            act_mean = env.get_action_mean()
            act_var = env.get_action_var()
        if hasattr(env, "get_amp_obs"):
            from spinup.env_wrappers.amp_normalizing_env import AmpNormalizingEnv
            env = AmpNormalizingEnv(env, act_mean, act_var)
            if test_env != None:
                test_env = AmpNormalizingEnv(test_env, act_mean, act_var)
        else:
            from spinup.env_wrappers.normalizing_env import NormalizingEnv
            env = NormalizingEnv(env, act_mean, act_var)
            if test_env != None:
                test_env = NormalizingEnv(test_env, act_mean, act_var)
    if her_k > 0:
        from spinup.env_wrappers.her_env import GoalGeneratorEnv
        env = GoalGeneratorEnv(env)
        goal_env = env
        if test_env != None:
            test_env = GoalGeneratorEnv(test_env)
            test_goal_env = test_env
    if max_skip > 0:
        from spinup.env_wrappers.dynamic_skip_env import DynamicSkipEnv
        env = DynamicSkipEnv(env, max_repetitions=max_skip, render_steps=render_steps)
        if test_env != None:
            test_env = DynamicSkipEnv(test_env)
    if her_k > 0:
        from spinup.env_wrappers.her_env import HerEnv
        env = HerEnv(env, k=her_k, her_reward=her_reward, goal_generator_env=goal_env, split_reward_streams=split_reward_streams)
        if test_env != None:
            test_env = HerEnv(test_env, k=her_k, her_reward=her_reward, goal_generator_env=test_goal_env, test_mode=True)
    if signal_vector_size > 0:
        from spinup.env_wrappers.signaling_env import SignalingEnv
        env = SignalingEnv(env, signal_vector_size)
        if test_env != None:
            test_env = SignalingEnv(test_env, signal_vector_size)
        #from spinup.env_wrappers.discrete_signaling_env import DiscreteSignalingEnv
        #env = DiscreteSignalingEnv(env, signal_vector_size)
        #if test_env != None:
        #    test_env = DiscreteSignalingEnv(test_env, signal_vector_size)
    if intrinsic_reward_weight > 0:
        from spinup.env_wrappers.rnd_curiosity_env_pytorch import RndCuriosityEnv
        env = RndCuriosityEnv(env, intrinsic_reward_weight=intrinsic_reward_weight, split_reward_streams=split_reward_streams, cuda_device=cuda_device,
                model_fn=model.net, model_kwargs_getter=model.get_default_kwargs, 
                restore_model_path=restore_model_path, save_model_path=os.path.join(logger.output_dir, "pyt_save", "rnd_{}_model_{}.pt"))
        if not env.add_hidden_step_rewards:
            intrinsic_reward_generators.append(env)
    if stack_size > 0:
        from spinup.env_wrappers.obs_stack_env import ObsStackEnv
        env = ObsStackEnv(env, stack_size=stack_size)
        if test_env != None:
            test_env = ObsStackEnv(test_env, stack_size=stack_size, test_mode=True)
    if use_interactive_entropy_source:
        from spinup.env_wrappers.hyper_param_selection_env import HyperParamSelectionEnv
        env = HyperParamSelectionEnv(env, param_name="sample_new_noise", min_param_value=-1, max_param_value=1)
        if test_env != None:
            test_env = HyperParamSelectionEnv(test_env, param_name="sample_new_noise", min_param_value=-1, max_param_value=1)
    if greedy_adversarial_priors:
        from spinup.env_wrappers.greedy_adversarial_priors_env import GreedyAdversarialPriorsEnv
        env = GreedyAdversarialPriorsEnv(env)
        if test_env != None:
            test_env = GreedyAdversarialPriorsEnv(test_env)

        # TODO just for fun (and testing), let's stack two of these
        #env = GreedyAdversarialPriorsEnv(env)
        #if test_env != None:
        #    # TODO this doesn't really make much sense.
        #    test_env = GreedyAdversarialPriorsEnv(test_env)

    return env, test_env, intrinsic_reward_generators
