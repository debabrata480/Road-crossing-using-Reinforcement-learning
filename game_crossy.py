# === game_crossy.py ===
from ursina import *
import random
import numpy as np

class CrossyEnv:
    def __init__(self, reward_progress_scale: float = 0.0, step_penalty: float = -0.01, collision_penalty: float = -10.0, oob_penalty: float = -50.0, success_reward: float = 20.0):
        # initialize Ursina app
        self.app = Ursina()


        window.fps_counter.enabled = False
        window.exit_button.visible = False

        sky_tex = load_texture('assets/skybox.png')
        grass_tex = load_texture('assets/grass_block.png')
        stone_tex = load_texture('assets/stone_block.png')
        car_tex = load_texture('assets/blue.png')
        boy_tex = load_texture('assets/uv_low.png')
        
        # world size
        self.grid_width = 10
        self.grid_height = 20   # used for spacing

        # --- Camera: fixed isometric 45° view ---
        camera.orthographic = False
        camera.fov = 45   # zoom
        camera.position = (20, 20, -30)   # offset so grid is visible
        camera.rotation_x = 45
        camera.rotation_y = -30
        camera.look_at(Vec3(0, 0, 0))

        # --- Cars / Obstacles ---
        self.cars = []
        self.num_lanes = 5
        # place lanes spaced out starting above the start row
        self.lane_z = [i * 2 - self.grid_height // 2 for i in range(2, 2 + self.num_lanes)]
        last_lane_z = max(self.lane_z)

        for z in self.lane_z:
            car = Entity(
                model='assets/blue.obj',
                texture=car_tex,
                position=(random.randint(-self.grid_width // 2, self.grid_width // 2), 0, z),
                scale=0.5,
                
            )
            car.speed = random.choice([0.05, 0.07, 0.1]) * random.choice([-1, 1])

            # --- set rotation based on direction ---
            if car.speed > 0:   # moving right
                car.rotation_y = 90
            else:               # moving left
                car.rotation_y =-90

            self.cars.append(car)


        # --- Ground tiles (flattened cubes) ---
        # start row -> finish row (just after last lane)
        self.finish_line = last_lane_z + 1
        for y in range(-self.grid_height // 2, self.finish_line + 1):
            for x in range(-self.grid_width // 2, self.grid_width // 2 + 1):
                if y == -self.grid_height // 2:
                    tile_color = grass_tex   # start safe zone
                elif y == self.finish_line:
                    tile_color = grass_tex   # finish safe zone
                else:
                    tile_color = stone_tex
                Entity(
                    model='assets/block',
                    texture = tile_color,
                    position=(x, -1, y),
                    scale=(0.5,0.5,0.5)
                )

        # --- Player ---
        self.player = Entity(
            model='assets/mulder',
            texture = boy_tex,
            position=(0, 1, -self.grid_height // 2),
            scale=0.75,
            rotation_y=90
        )
        # --- Sky ---
        self.sky = Entity(
            parent=scene,
            model='sphere',
            texture=sky_tex,
            scale=100,
            rotation_x=180,
            double_sided=True   # render inside the sphere
        )




        # --- Actions (mapped to x,z movement) ---
        self.action_map = {
            0: Vec2(-1, 0),   # left
            1: Vec2(1, 0),    # right
            2: Vec2(0, 1),    # up
            3: Vec2(0, -1),   # down
            4: Vec2(0, 0)     # stay
        }
        self.n_actions = len(self.action_map)

        # --- Observation shape ---
        self.observation_shape = (2 + 2 * len(self.cars),)

        self.done = False
        self.score = 0
        # --- Hyperparameters for reward shaping and penalties ---
        self.reward_progress_scale = float(reward_progress_scale)
        self.step_penalty = float(step_penalty)
        self.collision_penalty = float(collision_penalty)
        self.oob_penalty = float(oob_penalty)
        self.success_reward = float(success_reward)
        # track last z for progress-based reward shaping
        self._last_player_z = None

    def reset(self):
        self.player.position = (0, 1, -self.grid_height // 2)
        for car in self.cars:
            car.x = random.randint(-self.grid_width // 2, self.grid_width // 2)
        self.done = False
        self.score = 0
        self._last_player_z = self.player.z
        return self._get_obs()

    def step(self, action):
        if self.done:
            return self.reset(), 0.0, True, {}

        move = self.action_map[action]
        new_position = self.player.position + Vec3(move.x, 0, move.y)

        # Boundary checks with heavy penalty for going out of bounds
        x_min, x_max = -self.grid_width // 2, self.grid_width // 2
        z_min = -self.grid_height // 2
        z_max = self.finish_line
        
        if new_position.x < x_min or new_position.x > x_max or new_position.z < z_min or new_position.z > z_max:
            # Player attempted to go out of bounds - apply heavy penalty and don't move
            self.done = True
            return self._get_obs(), float(self.oob_penalty), True, {"termination": "oob"}
        
        # Move player
        self.player.position = new_position

        # base step penalty
        reward = float(self.step_penalty)

        for car in self.cars:
            car.x += car.speed
            if car.x > self.grid_width // 2:
                car.x = -self.grid_width // 2
            if car.x < -self.grid_width // 2:
                car.x = self.grid_width // 2

            # collision check - tighter tolerance
            distance = ((self.player.x - car.x)**2 + (self.player.z - car.z)**2)**0.5
            if distance < 0.8:  # Collision if within 0.8 units
                self.done = True
                return self._get_obs(), float(self.collision_penalty), True, {"termination": "collision"}

        # reached finish line
        if self.player.z >= self.finish_line:
            self.done = True
            return self._get_obs(), float(self.success_reward), True, {"termination": "success"}
        # reward shaping: add positive reward for forward progress along +z
        if self.reward_progress_scale != 0.0 and self._last_player_z is not None:
            delta_z = float(self.player.z - self._last_player_z)
            if delta_z > 0:
                reward += self.reward_progress_scale * delta_z
        self._last_player_z = self.player.z

        return self._get_obs(), reward, self.done, {"termination": None}

    def _get_obs(self):
        obs = [self.player.x, self.player.z]
        for car in self.cars:
            obs.append(car.x)
            obs.append(car.z)
        return np.array(obs, dtype=np.float32)

    def render(self):
        self.app.step()


# --- Debug run ---
if __name__ == "__main__":
    env = CrossyEnv()


    obs = env.reset()
    print("Initial obs:", obs)
    for _ in range(200):
        a = random.randint(0, env.n_actions - 1)
        obs, r, done, _ = env.step(a)
        env.render()
        print(obs, r, done)
        if done:
            obs = env.reset()
