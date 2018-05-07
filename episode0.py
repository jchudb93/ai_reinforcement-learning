import gym
import numpy as np

def gen_random_policy():
	return (np.random.uniform(-1,1, size=4), np.random.uniform(-1,1))

def policy_to_action(env, policy, obs):
	if np.dot(policy[0], obs) + policy[1] > 0:
		return 1
	else:
		return 0

def run_episode(env, policy, t_max=1000, render = False):
	obs = env.reset()
	total_reward = 0
	for i in range(t_max):
		if render:
			env.render()
		selected_action = policy_to_action(env, policy, obs)
		obs, reward, done, _ = env.step(selected_action)
		total_reward += reward
		if done:
			break

	return total_reward

if __name__ == '__main__':
	env = gym.make('CartPole-v0')

	n_policy = 500

	policy_list = [gen_random_policy() for _ in range(n_policy)]

	score_list = [run_episode(env,p) for p in policy_list]

	print('Best policy score = ' + str(max(score_list)))

	best_policy = policy_list[np.argmax(score_list)]

	print('Running best policy')
	run_episode(env, best_policy, render=True)
