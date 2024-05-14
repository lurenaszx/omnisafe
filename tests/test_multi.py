import omnisafe



# env_id = 'RockPaperScissors-v0'
env_id = 'KuhnPoker-v0'
custom_cfgs = {
    'train_cfgs': {
        'total_steps': 2000000,
        'vector_env_nums': 1,
        'parallel': 1,
    },
    'algo_cfgs': {
        'steps_per_epoch': 200,
        'update_iters': 5,
        'policy_delay': 10,
        'obs_normalize': False
    },
    'logger_cfgs': {
        'use_wandb': False,
    },
    'env_cfgs': {
        'num_players': 2
    }
}

agent = omnisafe.MultiAgents('QPG', env_id, custom_cfgs=custom_cfgs)
agent.learn()
agent.evaluate()
