import numpy as np

def random_exploration(env, num_episodes: int, layout_type: int):
    dataset = dict()
    observations = []
    actions = []
    dones = []

    available_actions = np.array([0, 1, 2, 3])
    for _ in range(num_episodes):
        i=0
        env.reset()
        cur_observations = []
        cur_actions = []
        cur_dones = []
        done = False
        while not done:
            i+=1
            cur_observations.append(np.array(env.env.unwrapped.agent_pos, dtype=np.float32))
            action = np.random.choice(available_actions, replace=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            cur_actions.append(np.array(action, dtype=np.float32))
            done = truncated
            cur_dones.append(np.array(done, dtype=np.float32))

        observations.append(np.stack(cur_observations))
        actions.append(np.stack(cur_actions))
        dones.append(np.stack(cur_dones))
    
    dataset['observations'] = np.concatenate(observations)
    dataset['terminals'] = np.concatenate(dones)
    dataset['actions'] = np.concatenate(actions)
    
    ob_mask = (1.0 - dataset['terminals']).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    dataset['next_observations'] = dataset['observations'][next_ob_mask]
    dataset['observations'] = dataset['observations'][ob_mask]
    dataset['actions'] = dataset['actions'][ob_mask].astype(np.int8)
    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)
    dataset['layout_type'] = np.repeat(np.array(layout_type), repeats=(dataset['actions'].shape[0], ))
    return dataset, env

def random_exploration_inference(env, context_len: int, num_episodes=1, layout_type: int=0):
    dataset = dict()
    observations = []
    actions = []
    dones = []

    available_actions = np.array([0, 1, 2, 3])
    for _ in range(num_episodes):
        i=0
        env.reset()
        cur_observations = []
        cur_actions = []
        cur_dones = []
        done = False
        while i < context_len:
            i+=1
            cur_observations.append(np.array(env.env.unwrapped.agent_pos, dtype=np.float32))
            action = np.random.choice(available_actions, replace=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            cur_actions.append(np.array(action, dtype=np.float32))
            done = truncated
            cur_dones.append(np.array(done, dtype=np.float32))

        observations.append(np.stack(cur_observations))
        actions.append(np.stack(cur_actions))
        dones.append(np.stack(cur_dones))
    
    dataset['observations'] = np.concatenate(observations)
    dataset['terminals'] = np.concatenate(dones)
    dataset['actions'] = np.concatenate(actions)
    
    ob_mask = (1.0 - dataset['terminals']).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    dataset['next_observations'] = dataset['observations'][next_ob_mask]
    dataset['observations'] = dataset['observations'][ob_mask]
    dataset['actions'] = dataset['actions'][ob_mask].astype(np.int8)
    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)
    dataset['layout_type'] = np.repeat(np.array(layout_type), repeats=(dataset['actions'].shape[0], ))
    return dataset, env