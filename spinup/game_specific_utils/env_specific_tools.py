

def select_demo_provider(env, env_tools_kwargs):
	"""
	Tries to construct a suitable demonstration data loader for a given env and data path.
	"""
	if "MineRL" in env.spec.id:
		if "data_dir" not in env_tools_kwargs:
			print("Warning: In order to load demonstrations in a minerl environment, please specify a data_dir in the env_tools_kwargs.")
			return None
		from spinup.game_specific_utils.minecraft.minerl_demo_provider import MineRlDemoProvider
		return MineRlDemoProvider(env.spec.id, env_tools_kwargs["data_dir"])
	else:
		print("Warning: env_specific_tools.py doesn't know a suitable demonstration provider for environment " + str(env.spec.id))
		return None # TODO is there a generic way for providing demonstrations? Maybe using the best 5% of trajectories as demonstrations?
