from gym.envs.registration import register

register(
    id='DiscreteContinuousGridworld-v0',
    entry_point='my_gym.envs:DiscreteContinuousGridWorld',
)

register(
    id='DiscreteGaussianGridworld-v0',
    entry_point='my_gym.envs:DiscreteGaussianGridWorld',
)

register(
    id='LinMDP-v0',
    entry_point='my_gym.envs:LinearMDP',
)

