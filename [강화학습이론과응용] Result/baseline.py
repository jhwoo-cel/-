# baseline_eval.py
import numpy as np
from grid_amr_env import AMRSmallEnv


def move_to_action(dx, dy):
    if dx == 0 and dy == -1: return 0  # up
    if dx == 0 and dy == 1:  return 1  # down
    if dx == -1 and dy == 0: return 2  # left
    if dx == 1 and dy == 0:  return 3  # right
    return 4  # no-op


def greedy_step(current, target, walls, BPs):
    cx, cy = current
    tx, ty = target

    candidates = [
        (cx, cy - 1),  # up
        (cx, cy + 1),  # down
        (cx - 1, cy),  # left
        (cx + 1, cy),  # right
    ]

    valid = []
    for nx, ny in candidates:
        if nx < 0 or ny < 0 or nx >= 10 or ny >= 10:
            continue
        if (nx, ny) in walls:
            continue

    
        if (nx, ny) in BPs and not (nx == tx and ny == ty):
            continue

        valid.append((nx, ny))

    if not valid:
        return current

    def manhattan(p):
        return abs(p[0] - tx) + abs(p[1] - ty)

    best = min(valid, key=manhattan)
    return np.array(best)


def baseline_policy(obs, env: AMRSmallEnv):
    A1 = obs[0:2].astype(int)
    A2 = obs[2:4].astype(int)
    GP = obs[4:6].astype(int)

    BP1 = obs[6:8].astype(int)
    BP2 = obs[8:10].astype(int)
    BP3 = obs[10:12].astype(int)

    BPs = []
    for bp in [BP1, BP2, BP3]:
        if not np.array_equal(bp, [-1, -1]):
            BPs.append((bp[0], bp[1]))

    walls = env.walls

    if env.phase == 1:
        required_BP = tuple(env.required_BP)

        if not env.required_BP_removed:
            if np.array_equal(A1, env.required_BP):
                return 5  # lift
            next_pos = greedy_step(A1, env.required_BP, walls, BPs)
        else:
            if np.array_equal(A1, env.A1_goal):
                return 4  # no-op
            next_pos = greedy_step(A1, env.A1_goal, walls, BPs)

        dx, dy = next_pos - A1
        return move_to_action(dx, dy)


    if env.phase == 2:
        if not env.GP_removed:

            if np.array_equal(A2, GP):
                return 5  # lift
            next_pos = greedy_step(A2, GP, walls, BPs)
        else:

            if np.array_equal(A2, env.A2_goal):
                return 4  # no-op
            next_pos = greedy_step(A2, env.A2_goal, walls, BPs)

        dx, dy = next_pos - A2
        return move_to_action(dx, dy)

    if env.phase == 3:
        required_BP = tuple(env.required_BP)

        if not env.A1_has_BP:
            if np.array_equal(A1, env.A1_goal):
                return 5  # lift
            next_pos = greedy_step(A1, env.A1_goal, walls, BPs)
        else:
            if np.array_equal(A1, required_BP):
                return 6  # put
            next_pos = greedy_step(A1, env.required_BP, walls, BPs)

        dx, dy = next_pos - A1
        return move_to_action(dx, dy)
    
    return 4  


def evaluate_baseline(num_episodes=20, render=False):
    env = AMRSmallEnv()
    rewards = []
    lengths = []
    successes = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        ep_len = 0

        while not done:
            action = baseline_policy(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            ep_len += 1
            done = terminated or truncated

            if render:
                env.render()

        rewards.append(ep_rew)
        lengths.append(ep_len)
        if info.get("is_success", False):
            successes += 1

        print(f"[EP {ep+1:02d}] reward={ep_rew:.1f}, length={ep_len}, success={info.get('is_success', False)}")

    mean_rew = np.mean(rewards)
    mean_len = np.mean(lengths)
    success_rate = successes / num_episodes

    print("\n========== BASELINE SUMMARY ==========")
    print(f" Episodes           : {num_episodes}")
    print(f" Mean Reward        : {mean_rew:.2f}")
    print(f" Mean Episode Length: {mean_len:.2f}")
    print(f" Success Rate       : {success_rate:.2f}")
    print("======================================")

    return mean_rew, mean_len, success_rate


if __name__ == "__main__":
    evaluate_baseline(num_episodes=20, render=False)
