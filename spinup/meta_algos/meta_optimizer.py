import numpy as np

class MetaOptimizer:
	"""
	A hyperparameter optimizer that can be used to find good hyperparameters on the fly. 
	The optimizer explores hyperparameter space and tries to find hyperparameters that maximize 
	average mesa returns.
	Very vaguely inspired by stuff like population based training, https://arxiv.org/abs/1711.09846,
	but without actual populations. Instead we just apply noise to the hyperparameters of a single agent,
	average the returns and optimize on the fly.
	"""
	def __init__(self, hyperparams, check_limit_after=1.5e6):
		"""
		Args:
			hyperparams (list[list or scalar]): A list of hyperparameters to tune. Each hyperparameter must itself
				be specified as a list with entries [lower_bound, initial_value, upper_bound, update_interval, population_size, stddev_meta_search].
				Special case for convinience: If single scalars are passed in, they won't be tuned.
		"""
		self.hyperparams = hyperparams
		self.hyperparam_returns = [0] * len(hyperparams)
		self.prev_param_returns = [[] for _ in range(len(hyperparams))]
		self.prev_param_choices = [[] for _ in range(len(hyperparams))]
		self.current_hyper_param_values = [param if np.isscalar(param) else param[1] for param in hyperparams]
		self.check_limit_after = check_limit_after
		self.final_value_reached = set()
		self.step_counter = 0

	def get_hyperparameter_values(self):
		"""
		Returns (list[float]):
			The current hyperparameter values. This method is used for convenience to get rid of the lists that can be used to specify
			how a hyperparameter should be tuned. 
		"""
		return self.current_hyper_param_values

	def update_hyperparameters(self, step_counter, step_reward):
		"""
		Args:
			step_counter (int): The current step of the underlying mesa optimization process.
		Returns (list[float]):
			A list of updated values for the hyperparameters, odered in the same way in which they where passed to the constructor.
		"""
		idx = 0
		for param in self.hyperparams:
			if np.isscalar(param) or idx in self.final_value_reached:
				idx += 1
				continue
			lower_bound, initial_value, upper_bound, update_interval, population_size, stddev_meta_search, lock_at = param
			self.hyperparam_returns[idx] += step_reward 
			if (step_counter + 1) % update_interval == 0:
				current_return = self.hyperparam_returns[idx]
				prev_returns = self.prev_param_returns[idx]
				prev_returns.append(current_return)
				if len(prev_returns) > population_size:
					prev_returns.pop(0)
				prev_choice = self.prev_param_choices[idx]
				prev_choice.append(self.current_hyper_param_values[idx])
				if len(prev_choice) > population_size:
					prev_choice.pop(0)
				max_choice = prev_choice[np.argmax(prev_returns)]
				if self.step_counter > self.check_limit_after and max_choice == lock_at:
					self.final_value_reached.add(idx)
					self.current_hyper_param_values[idx] = lock_at
				else:
					self.current_hyper_param_values[idx] = np.clip(np.random.normal(loc=max_choice, scale=stddev_meta_search), lower_bound, upper_bound)
				self.hyperparam_returns[idx] = 0
			idx += 1
		self.step_counter += 1
		return self.current_hyper_param_values






