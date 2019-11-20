from gym.envs.registration import register

register(
    id='str_opt-v0',
    entry_point='steering_optimizer.envs:StrOptEnv',
)