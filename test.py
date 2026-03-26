import numpy as np

from RL_Model.agent import SACAgent
from RL_Model.replay_buffer import ReplayBuffer

state_dim = 22
action_dim = 1

agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
)

buffer = ReplayBuffer(
    capacity=100,
    state_dim=state_dim,
    action_dim=action_dim,
)

for i in range(10):
    state = np.random.randn(state_dim)
    action = np.random.randn(action_dim)
    reward = float(np.random.randn())
    next_state = np.random.randn(state_dim)
    done = False

    buffer.add(
        state,
        action,
        reward,
        next_state,
        done,
    )

print("Buffer size:", len(buffer))

state = np.random.randn(state_dim)

action = agent.select_action(state)

print("Selected action:", action)

metrics = agent.update(
    replay_buffer=buffer,
    batch_size=5,
)

print("Update metrics:")

for k, v in metrics.items():
    print(k, "=", v)