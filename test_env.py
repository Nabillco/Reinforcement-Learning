from env.lunar_lander import make_env
print("TEST STARTED")

env = make_env()

state, _ = env.reset()

print("State:", state)
print("Action space:", env.action_space)

for _ in range(10):
    action = env.action_space.sample()
    state, reward, done, truncated, _ = env.step(action)
    print(reward)

    if done or truncated:
        state, _ = env.reset()