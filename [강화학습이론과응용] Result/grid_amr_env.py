# grid_amr_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class AMRSmallEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=(10, 10), render_delay=0.2):
        super().__init__()
        self.grid_w, self.grid_h = grid_size
        self.render_delay = render_delay

        # Pallet, GP positions
        self.required_BP = np.array([4, 2])  # blocking pallet
        self.BP_positions = [
            self.required_BP.copy(),
            np.array([5, 2]),
            np.array([5, 1])
        ]
        self.GP_init = np.array([4, 1])       # goal pallet

        # AMRs
        self.A1_init = np.array([3, 5])
        self.A2_init = np.array([6, 5])
        self.A1_goal = self.A1_init.copy()
        self.A2_goal = self.A2_init.copy()

        # Walls
        self.walls = set()
        for x in range(0, 10):
            self.walls.add((x, 0))
        self.walls.add((3, 1))
        self.walls.add((3, 2))
        self.walls.add((6, 1))
        self.walls.add((6, 2))

        # Observation = A1(2) + A2(2) + GP(2) + BP1~3(6) = 12
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.grid_w, self.grid_h),
            shape=(12,),
            dtype=np.float32
        )

        # Actions = 0~3 move, 4 no-op, 5 lift, 6 put
        self.action_space = spaces.Discrete(7)

        self.fig, self.ax = None, None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.phase = 1

        self.required_BP_removed = False
        self.GP_removed = False
        self.A1_has_BP = False

        self.A1 = self.A1_init.copy()
        self.A2 = self.A2_init.copy()
        self.GP = self.GP_init.copy()
        self.BPs = [bp.copy() for bp in self.BP_positions]

        return self._get_state(), {}

    def _get_state(self):
        flat_BPs = [coord for bp in self.BPs for coord in bp]
        return np.array([
            *self.A1,
            *self.A2,
            *self.GP,
            *flat_BPs
        ], dtype=np.float32)

    def _move(self, pos, dx, dy):
        nx, ny = pos[0] + dx, pos[1] + dy
        if (nx, ny) in self.walls:
            return pos
        if nx < 0 or nx >= self.grid_w or ny < 0 or ny >= self.grid_h:
            return pos
        return np.array([nx, ny])


    def step(self, action):
        self.steps += 1
        reward = -0.1
        terminated = False
        truncated = False
        info = {"is_success": False}

        # PHASE 1 — AMR1 moves required BP
        if self.phase == 1:
            old_pos = self.A1.copy()

            # movement 
            if action == 0: self.A1 = self._move(self.A1, 0, -1)
            elif action == 1: self.A1 = self._move(self.A1, 0, 1)
            elif action == 2: self.A1 = self._move(self.A1, -1, 0)
            elif action == 3: self.A1 = self._move(self.A1, 1, 0)

            # lift 
            elif action == 5:
                for i, bp in enumerate(self.BPs):
                    if np.array_equal(self.A1, bp):
                        if np.array_equal(bp, self.required_BP):
                            reward += 30
                            self.required_BP_removed = True
                            self.A1_has_BP = True
                            self.BPs[i] = np.array([-1, -1])
                        break

            # shaping
            if not self.required_BP_removed:
                old_d = np.linalg.norm(old_pos - self.required_BP)
                new_d = np.linalg.norm(self.A1 - self.required_BP)
                reward += 0.1 * (old_d - new_d)
            else:
                old_d = np.linalg.norm(old_pos - self.A1_goal)
                new_d = np.linalg.norm(self.A1 - self.A1_goal)
                reward += 0.1 * (old_d - new_d)

                if np.array_equal(self.A1, self.A1_goal):
                    reward += 20
                    self.phase = 2

        # PHASE 2 — AMR2 moves GP and returns

        elif self.phase == 2:
            old_pos = self.A2.copy()

            # movement
            if action == 0: self.A2 = self._move(self.A2, 0, -1)
            elif action == 1: self.A2 = self._move(self.A2, 0, 1)
            elif action == 2: self.A2 = self._move(self.A2, -1, 0)
            elif action == 3: self.A2 = self._move(self.A2, 1, 0)

            # lift GP
            elif action == 5:
                if np.array_equal(self.A2, self.GP):
                    reward += 30
                    self.GP_removed = True
                    self.GP = np.array([-1, -1])

            # shaping
            if not self.GP_removed:
                old_d = np.linalg.norm(old_pos - self.GP)
                new_d = np.linalg.norm(self.A2 - self.GP)
                reward += 0.1 * (old_d - new_d)
            else:
                old_d = np.linalg.norm(old_pos - self.A2_goal)
                new_d = np.linalg.norm(self.A2 - self.A2_goal)
                reward += 0.1 * (old_d - new_d)

                if np.array_equal(self.A2, self.A2_goal):
                    reward += 30
                    self.phase = 3

        # PHASE 3 — AMR1 restores required BP then return home

        elif self.phase == 3:
            old_pos = self.A1.copy()

            # movement
            if action == 0: self.A1 = self._move(self.A1, 0, -1)
            elif action == 1: self.A1 = self._move(self.A1, 0, 1)
            elif action == 2: self.A1 = self._move(self.A1, -1, 0)
            elif action == 3: self.A1 = self._move(self.A1, 1, 0)

            # lift again at A1_goal
            elif action == 5:
                if not self.A1_has_BP and np.array_equal(self.A1, self.A1_goal):
                    self.A1_has_BP = True
                    reward += 20

            # put back to required position
            elif action == 6:
                if self.A1_has_BP and np.array_equal(self.A1, self.required_BP):
                    for i, bp in enumerate(self.BPs):
                        if np.array_equal(bp, [-1, -1]):
                            self.BPs[i] = self.required_BP.copy()
                            break
                    reward += 30
                    self.A1_has_BP = False

            # shaping
            if self.A1_has_BP:
                old_d = np.linalg.norm(old_pos - self.required_BP)
                new_d = np.linalg.norm(self.A1 - self.required_BP)
                reward += 0.1 * (old_d - new_d)
            else:
                old_d = np.linalg.norm(old_pos - self.A1_goal)
                new_d = np.linalg.norm(self.A1 - self.A1_goal)
                reward += 0.1 * (old_d - new_d)

                if np.array_equal(self.A1, self.A1_goal):
                    reward += 50
                    terminated = True
                    info["is_success"] = True

        # Collision / Timeout

        if np.array_equal(self.A1, self.A2):
            reward -= 5
            terminated = True

        if self.steps >= 400:
            truncated = True

        # Episode Summary for Monitor 

        if terminated:
            info["episode"] = {
                "r": reward,
                "l": self.steps,
                "is_success": info.get("is_success", False)  
            }
    

        return self._get_state(), reward, terminated, truncated, info


    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6,6))

        self.ax.clear()

        # grid
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                self.ax.add_patch(plt.Rectangle((x-0.5,y-0.5),1,1,fill=False,edgecolor="lightgray"))

        # Legend handles
        legend_elements = []

        # walls
        for (x,y) in self.walls:
            rect = plt.Rectangle((x-0.5,y-0.5),1,1,color="black")
            self.ax.add_patch(rect)
        legend_elements.append(plt.Rectangle((0,0),1,1,color="black", label="Wall"))

        # AMR1
        amr1_plot, = self.ax.plot(self.A1[0], self.A1[1], "o",
                                color="lightblue", markersize=12, markeredgecolor="blue")
        legend_elements.append(amr1_plot)
        amr1_plot.set_label("AMR1")

        # AMR2
        amr2_plot, = self.ax.plot(self.A2[0], self.A2[1], "ro", markersize=12)
        legend_elements.append(amr2_plot)
        amr2_plot.set_label("AMR2")

        # BP (Blocking Pallets)
        for bp in self.BPs:
            if not np.array_equal(bp, [-1, -1]):
                bp_plot, = self.ax.plot(bp[0], bp[1], "s", color="gray", markersize=12)
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='gray',
                                        markersize=10, linestyle='None', label='BP'))

        # GP (Goal Pallet)
        if not np.array_equal(self.GP, [-1, -1]):
            gp_plot, = self.ax.plot(self.GP[0], self.GP[1], "ys", markersize=12)
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='yellow',
                                        markersize=10, linestyle='None', label='GP'))

        # Legend
        self.ax.legend(handles=legend_elements, loc="upper right")

        # axis settings
        self.ax.set_xlim(-0.5, 9.5)
        self.ax.set_ylim(-0.5, 9.5)
        self.ax.invert_yaxis()

        plt.pause(self.render_delay)
