import pygame
import numpy as np
import sys
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

# Pygame setup
WIDTH, HEIGHT = 800, 800
BG_COLOR = (30, 30, 30)
ROBOT_COLOR = (200, 255, 255)
OBSTACLE_COLOR = (200, 50, 50)
FONT_COLOR = (255, 255, 255)

SIM_DT = 1 / 60.0

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "Robots Detecting Each Other and Obstacles with Proximity & RAB"
)
font = pygame.font.SysFont(None, 20)

# Surrounding walls
ARENA_BOUNDS = {"left": 0, "right": WIDTH, "top": 0, "bottom": HEIGHT}

# Parameters
NUM_ROBOTS = 16
ROBOT_RADIUS = 10

NUM_PROX_SENSORS = 6
NUM_RAB_SENSORS = 12

PROX_SENSOR_RANGE = 100  # pixels
RAB_RANGE = 150  # pixels

MAX_SPEED = 150
MAX_TURN = 16  # radians/sec - a real robot like the e-puck or TurtleBot typically turns at 90–180 deg/sec (≈ 1.5–3.1 rad/sec)

# sensor noise and dropout
RAB_NOISE_BEARING = 0  # std dev of directional noise to bearing in RAB:  0.1 rad. = ~5.7 degree in the bearing
RAB_DROPOUT = 0  # chance to drop a signal
LIGHT_NOISE_STD = 0  # noise in perceived light
ORIENTATION_NOISE_STD = 0  # noise in IMU readings of the robot’s own orientation

# noise in the motion model (simulates actuation/motor errors)
MOTION_NOISE_STD = 0  # Try 0.5   # Positional noise in dx/dy (pixels)
HEADING_NOISE_STD = 0  # Try 0.01 # Rotational noise in heading (radians)


def rotate_vector(vec, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


class LightSource:
    def __init__(self, pos, intensity=1.0, core_radius=0, decay_radius=200):
        self.pos = np.array(pos, dtype=float)
        self.intensity = intensity
        self.core_radius = core_radius
        self.decay_radius = decay_radius

    def get_intensity_at(self, dist):
        if dist > self.decay_radius:
            return 0.0
        if dist < self.core_radius:
            return self.intensity
        # Inverse-square decay beyond the center radius (you can change to linear or exponential)
        return self.intensity * max(
            0.0,
            1.0
            - ((dist - self.core_radius) / (self.decay_radius - self.core_radius)) ** 2,
        )


# Utility function to sum light intensity from all sources
def _get_light_intensity(pos):
    raw_intensity = sum(
        source.get_intensity_at(np.linalg.norm(pos - source.pos))
        for source in LIGHT_SOURCES
    )
    noise_factor = np.random.normal(1.0, LIGHT_NOISE_STD)
    return np.clip(raw_intensity * noise_factor, 0.0, 1.0)


class Obstacle:
    def __init__(self, pos, radius):
        self.pos = np.array(pos, dtype=float)
        self.radius = radius
        self.type = "obstacle"  # used in sensing


OBSTACLES = [
    # Obstacle(pos=(200, 150), radius=20),
    # Obstacle(pos=(600, 120), radius=30),
]

LIGHT_SOURCES = [
    # LightSource(pos=(100, 100), intensity=1.0, core_radius=50, decay_radius=300),
    # LightSource(pos=(700, 500), intensity=0.9, core_radius=10, decay_radius=100)
]


class Robot:
    def __init__(self, id, pos, heading):
        self.id = id
        self._pos = np.array(pos, dtype=float)
        self._heading = heading
        self._radius = ROBOT_RADIUS
        self._linear_velocity = MAX_SPEED * 0.5
        self._angular_velocity = 0
        self._last_angle = 0

        #### signal broadcast via RAB (a very short message)
        self.broadcast_signal = False

        #### Sensor readings
        # proximity sensors
        self.prox_angles = np.pi / NUM_PROX_SENSORS + np.linspace(
            0, 2 * np.pi, NUM_PROX_SENSORS, endpoint=False
        )
        self.prox_readings = [
            {"distance": PROX_SENSOR_RANGE, "type": None}
            for _ in range(NUM_PROX_SENSORS)
        ]
        # RAB sensor
        self.rab_angles = np.pi / NUM_RAB_SENSORS + np.linspace(
            0, 2 * np.pi, NUM_RAB_SENSORS, endpoint=False
        )
        self.rab_signals = []
        # light sensor
        self.light_intensity = 0.0
        # IMU (Inertial Measurement Unit) sensor providing robot's orientation
        self.orientation = 0.0

    def move(self, dt):
        # Update heading
        self._heading += self._angular_velocity * dt
        self._heading += np.random.normal(0, HEADING_NOISE_STD)
        self._heading %= 2 * np.pi  # keep in [0, 2π)

        # Update position
        dx = self._linear_velocity * np.cos(self._heading) * dt
        dy = self._linear_velocity * np.sin(self._heading) * dt
        dx += np.random.normal(0, MOTION_NOISE_STD)
        dy += np.random.normal(0, MOTION_NOISE_STD)
        self._pos += np.array([dx, dy])

        # Arena bounds clipping
        self._pos[0] = np.clip(self._pos[0], self._radius, WIDTH - self._radius)
        self._pos[1] = np.clip(self._pos[1], self._radius, HEIGHT - self._radius)

    def compute_distance_to_wall(self, direction, bounds, max_range):
        x, y = self._pos
        dx, dy = direction
        distances = []
        if dx != 0:
            if dx > 0:
                t = (bounds["right"] - x) / dx
            else:
                t = (bounds["left"] - x) / dx
            if t > 0:
                distances.append(t)
        if dy != 0:
            if dy > 0:
                t = (bounds["bottom"] - y) / dy
            else:
                t = (bounds["top"] - y) / dy
            if t > 0:
                distances.append(t)
        wall_distance = min(distances) if distances else max_range
        wall_distance = max(wall_distance, 0)
        return min(wall_distance, max_range)

    def read_sensors(self, robots, obstacles, arena_bounds):

        # Empty the sensors
        self.prox_readings = [
            {"distance": PROX_SENSOR_RANGE, "type": None}
            for _ in range(NUM_PROX_SENSORS)
        ]
        self.rab_signals = []

        # Light sensing
        self.light_intensity = _get_light_intensity(self._pos)

        # Detect other robots
        for other in robots:
            if other.id == self.id:
                continue

            rel_vec = other._pos - self._pos
            distance = max(0, np.linalg.norm(rel_vec) - other._radius)
            if distance > max(PROX_SENSOR_RANGE, RAB_RANGE):
                continue

            bearing = (np.arctan2(rel_vec[1], rel_vec[0]) - self._heading) % (2 * np.pi)

            # Communication (RAB)
            if distance <= RAB_RANGE:
                # signal dropout
                dropout_probability = RAB_DROPOUT  # chance to drop a signal

                if np.random.rand() > dropout_probability:
                    # adding noise (directional error) to bearing
                    bearing = (bearing + np.random.normal(0, RAB_NOISE_BEARING)) % (
                        2 * np.pi
                    )

                    rab_idx = int((bearing / (2 * np.pi)) * NUM_RAB_SENSORS)

                    self.rab_signals.append(
                        {
                            "message": {"heading": other.orientation},
                            "distance": distance,
                            "bearing": self.rab_angles[rab_idx],  # local
                            "sensor_idx": rab_idx,
                            "intensity": 1 / ((distance / RAB_RANGE) ** 2 + 1),
                        }
                    )

            # Also treat robot as obstacle (for IR)
            if distance <= PROX_SENSOR_RANGE:
                prox_idx = int((bearing / (2 * np.pi)) * NUM_PROX_SENSORS)
                if distance < self.prox_readings[prox_idx]["distance"]:
                    self.prox_readings[prox_idx] = {
                        "distance": distance,
                        "type": "robot",
                    }

        # Detect obstacles
        for obs in obstacles:
            rel_vec = obs.pos - self._pos
            distance = max(0, np.linalg.norm(rel_vec) - obs.radius)
            if distance <= PROX_SENSOR_RANGE:
                bearing = (np.arctan2(rel_vec[1], rel_vec[0]) - self._heading) % (
                    2 * np.pi
                )
                prox_idx = int((bearing / (2 * np.pi)) * NUM_PROX_SENSORS)
                if distance < self.prox_readings[prox_idx]["distance"]:
                    self.prox_readings[prox_idx] = {
                        "distance": distance,
                        "type": "obstacle",
                    }

        # Wall sensing (raycast style)
        for i, angle in enumerate(self.prox_angles):
            global_angle = (self._heading + angle) % (2 * np.pi)
            direction = np.array([np.cos(global_angle), np.sin(global_angle)])
            wall_dist = self.compute_distance_to_wall(
                direction, arena_bounds, PROX_SENSOR_RANGE
            )
            if wall_dist < self.prox_readings[i]["distance"]:
                self.prox_readings[i] = {"distance": wall_dist, "type": "wall"}

        # Read IMU for own orientation
        self.orientation = (
            self._heading + np.random.normal(0, ORIENTATION_NOISE_STD)
        ) % (2 * np.pi)

    def _set_velocity(self, linear, angular):
        # Internal use only. Use set_rotation_and_speed instead
        assert 0 <= linear <= MAX_SPEED, "Linear velocity out of bounds"
        assert -MAX_TURN <= angular <= MAX_TURN, "Angular velocity out of bounds"
        self._linear_velocity = linear
        self._angular_velocity = angular

    def compute_angle_diff(self, target_angle):
        # Returns shortest signed angle between current heading and target
        return (target_angle - self._heading + np.pi) % (2 * np.pi) - np.pi

    def get_relative_heading(self, neighbor_heading):
        # Convert a neighbor's global heading into this robot's local frame (radians).
        #     Positive = CCW, Negative = CW.
        return (neighbor_heading - self.orientation + np.pi) % (2 * np.pi) - np.pi

    def set_rotation_and_speed(self, delta_bearing, target_speed, kp=0.5):
        """
        Sets angular and linear velocity using a proportional controller
        to achieve the given relative turn (rotation) with the given target speed.
        Robot-frame API: delta_bearing is relative to current heading (rad)
        + = turn left (CCW), - = turn right (CW).
        """
        target_heading = (self._heading + delta_bearing) % (2 * np.pi)
        angle_diff = self.compute_angle_diff(target_heading)
        angular_velocity = np.clip(kp * angle_diff, -MAX_TURN, MAX_TURN)
        target_speed = np.clip(target_speed, 0, MAX_SPEED)
        # Slow down when turning sharply
        linear_velocity = (
            target_speed * (1 - min(abs(angle_diff) / np.pi, 1)) * 0.9 + 0.1
        )
        self._set_velocity(linear_velocity, angular_velocity)

    def robot_controller(self, swarm_mode):

        ## --- DISPERSION --- ##
        if swarm_mode == 1:

            repel_vec = np.array([0.0, 0.0])
            too_close = False
            # Loop though proximity sensors
            for i, reading in enumerate(self.prox_readings):
                # If a sensor sees any obstacle within 60% of PROX_SENSOR_RANGE trigger too_close
                if (
                    reading["type"] in ["wall", "robot", "obstacle"]
                    and reading["distance"] < PROX_SENSOR_RANGE * 0.6
                ):
                    too_close = True
                    # Build a new heading pointing away from the obstacle
                    angle = self._heading + self.prox_angles[i]
                    repel_vec -= np.array([np.cos(angle), np.sin(angle)])

            # too_close = immediate threat ; broadcast_signal = avoidance mode from 'old' threat
            if too_close or self.broadcast_signal:
                # Convert the repulsion vector into a global angle.
                if np.linalg.norm(repel_vec) > 1e-5:
                    target_angle = np.arctan2(repel_vec[1], repel_vec[0])
                    delta_bearing = self.compute_angle_diff(target_angle)
                    self.set_rotation_and_speed(delta_bearing, MAX_SPEED * 0.5)
                # Robot stays in avoidance mode
                self.broadcast_signal = True
                # stop broadcast_signal only when ALL sensors report safe distance
                if all(
                    r["distance"] > PROX_SENSOR_RANGE * 0.8 for r in self.prox_readings
                ):
                    self.broadcast_signal = False
            else:
                self.set_rotation_and_speed(0, MAX_SPEED * 0.5)

        ## --- Flocking (Boids) --- ##

        if swarm_mode == 2:
            # --- parameters ---
            VISIBLE_RANGE = RAB_RANGE  # use RAB for neighbor detection
            PROTECTED = PROX_SENSOR_RANGE * 0.6  # very close -> strong separation
            OBSTACLE_ALERT = PROX_SENSOR_RANGE * 0.7

            # weights (tune these)
            w_align = 1.0
            w_cohesion = 0.6
            w_separation = 2.2
            w_repel = 4.0

            # accumulators
            alignment = np.array([0.0, 0.0])
            cohesion = np.array([0.0, 0.0])
            separation = np.array([0.0, 0.0])
            neighbor_count = 0

            repel_vec = np.array([0.0, 0.0])
            too_close = False

            # 1) obstacle / wall avoidance from proximity sensors
            for i, reading in enumerate(self.prox_readings):
                if (
                    reading["type"] in ("wall", "obstacle", "robot")
                    and reading["distance"] < OBSTACLE_ALERT
                ):
                    too_close = True
                    ang = self._heading + self.prox_angles[i]
                    repel_vec -= np.array([np.cos(ang), np.sin(ang)]) * (
                        1.0 - reading["distance"] / PROX_SENSOR_RANGE
                    )

            # 2) neighbours via RAB signals
            for sig in self.rab_signals:
                d = sig["distance"]
                if d > VISIBLE_RANGE:
                    continue
                # global angle to neighbor
                ang = (sig["bearing"] + self._heading) % (2 * np.pi)
                neighbor_pos = self._pos + np.array([np.cos(ang), np.sin(ang)]) * d

                # alignment: use neighbor heading (message stores global heading)
                nh = sig["message"].get("heading", 0.0)
                alignment += np.array([np.cos(nh), np.sin(nh)])
                cohesion += neighbor_pos
                neighbor_count += 1

                # separation stronger when very close
                offset = self._pos - neighbor_pos
                if d < PROTECTED:
                    # weight by proximity (closer => stronger)
                    separation += (offset / (d + 1e-6)) * ((PROTECTED - d) / PROTECTED)

            # normalize / finalize boid components
            if neighbor_count > 0:
                alignment = alignment / neighbor_count
                center = cohesion / neighbor_count
                cohesion = center - self._pos  # vector toward flock center

            # Normalize component lengths (prevents domination by any one)
            def normed(v):
                n = np.linalg.norm(v)
                return v / n if n > 1e-8 else v

            alignment = normed(alignment) * w_align
            cohesion = normed(cohesion) * w_cohesion
            separation = normed(separation) * w_separation
            repel = normed(repel_vec) * w_repel

            # If immediate danger, give repulsion priority and slow down
            if (too_close and np.linalg.norm(repel_vec) > 1e-6) or (
                np.linalg.norm(separation) > 0.0 and np.linalg.norm(separation) > 0.1
            ):
                # high priority avoidance
                flock_vec = repel + separation * 1.5
                target_speed = MAX_SPEED * 0.25
            else:
                # combine peacefully
                flock_vec = alignment + cohesion + separation + repel
                # increase forward speed with stronger alignment & cohesion (so group moves)
                speed_factor = 0.5 + 0.5 * min(1.0, neighbor_count / 6.0)
                target_speed = MAX_SPEED * (0.2 + 0.5 * speed_factor)

            # if there's something to do, compute heading & set velocities
            if np.linalg.norm(flock_vec) > 1e-6:
                target_angle = np.arctan2(flock_vec[1], flock_vec[0])
                delta_bearing = self.compute_angle_diff(target_angle)
                # use a bit more aggressive rotational gain so robots align quickly but still clipped by MAX_TURN
                self.set_rotation_and_speed(delta_bearing, target_speed, kp=1.2)
            else:
                self.set_rotation_and_speed(0, target_speed, kp=0.8)

        ## --- STOP ROBOTS --- ##
        if swarm_mode == 3:
            self.set_rotation_and_speed(0, MAX_SPEED * 0.0)
            self.broadcast_signal = False

    def draw(self, screen, active):
        if active:
            # --- IR proximity sensors ---
            for i, reading in enumerate(self.prox_readings):
                dist = reading["distance"]
                obj_type = reading["type"]

                angle = self._heading + self.prox_angles[i]
                sensor_dir = np.array([np.cos(angle), np.sin(angle)])
                end_pos = self._pos + sensor_dir * dist

                # Color code by detected object type
                if obj_type == "robot":
                    color = (0, 150, 255)  # Blue
                elif obj_type == "obstacle":
                    color = (255, 165, 0)  # Orange
                elif obj_type == "wall":
                    color = (255, 255, 100)  # Yellow
                else:
                    color = (20, 80, 20)  # Green (no hit)

                pygame.draw.line(screen, color, self._pos, end_pos, 2)
                pygame.draw.circle(screen, color, end_pos.astype(int), 3)

            # --- RAB signals ---
            for sig in self.rab_signals:
                sig_angle = self._heading + self.rab_angles[sig["sensor_idx"]]
                sensor_dir = np.array([np.cos(sig_angle), np.sin(sig_angle)])

                start = self._pos + sensor_dir * (self._radius + 3)
                end = self._pos + sensor_dir * (self._radius + 3 + sig["distance"])

                intensity_color = 55 + int(200 * (sig["intensity"] * 2 - 1))
                color = (intensity_color, 50, intensity_color)

                pygame.draw.line(screen, color, start, end, 2)

        # --- Robot body ---
        pygame.draw.circle(screen, ROBOT_COLOR, self._pos.astype(int), self._radius)

        # --- Heading indicator ---
        heading_vec = rotate_vector(np.array([self._radius + 2, 0]), self._heading)
        pygame.draw.line(screen, ROBOT_COLOR, self._pos, self._pos + heading_vec, 3)


def draw_obstacles(screen):
    for obs in OBSTACLES:
        pygame.draw.circle(screen, (120, 120, 120), obs.pos.astype(int), obs.radius)


def draw_light_sources(screen):
    for light in LIGHT_SOURCES:
        # Draw fading light circle
        for r in range(light.decay_radius, light.core_radius, -10):
            alpha = int(255 * light.get_intensity_at(r))
            surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                surface,
                (light.intensity * 255, light.intensity * 235, 0, alpha),
                (r, r),
                r,
            )
            screen.blit(surface, (light.pos[0] - r, light.pos[1] - r))

        # Draw core of the light
        pygame.draw.circle(
            screen,
            (light.intensity * 255, light.intensity * 255, 0),
            light.pos.astype(int),
            max(5, light.core_radius),
        )


def logging_init():  # initialize your log file
    pass


def log_metrics(frame_count, total_time, metrics):  # write to your log file
    pass


def logging_close():  # close your log file
    pass


def compute_metrics():  # pass as many arguments as you need and compute relevant metrics to be logged for performance analysis
    return []


def compute_convex_hull_area(robots):
    """Compute convex hull area of all robot positions."""
    points = np.array([r._pos for r in robots])

    # trivial cases (1 or 2 robots)
    if len(points) < 3:
        return 0.0

    hull = ConvexHull(points)
    return hull.volume  # for 2D hulls, "volume" is the polygon area


def compute_avg_nearest_neighbor(robots):
    dists = []
    for i, r in enumerate(robots):
        nearest = float("inf")
        for j, other in enumerate(robots):
            if i == j:
                continue
            dist = np.linalg.norm(r._pos - other._pos)
            if dist < nearest:
                nearest = dist
        if nearest < float("inf"):
            dists.append(nearest)
    return np.mean(dists) if dists else 0.0


def compute_num_flocks(robots, eps=None, min_samples=1):
    """
    Count the number of flocks using DBSCAN clustering.

    Args:
        robots: List of robot objects
        eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples: Minimum number of samples in a neighborhood for a point to be considered as a core point

    Returns:
        int: Number of flocks (clusters)
    """
    if len(robots) < min_samples:
        return 0

    if eps is None:
        eps = RAB_RANGE * 0.8

    # Extract robot positions
    positions = np.array([robot._pos for robot in robots])

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)

    # Count number of clusters (excluding noise points labeled as -1)
    num_flocks = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

    return num_flocks


def init_robots(seed=42):
    np.random.seed(seed)
    robots = []
    for i in range(NUM_ROBOTS):
        pos = np.random.uniform(
            [ROBOT_RADIUS, ROBOT_RADIUS], [WIDTH - ROBOT_RADIUS, HEIGHT - ROBOT_RADIUS]
        )
        heading = np.random.uniform(0, 2 * np.pi)
        robots.append(Robot(i, pos, heading))
    return robots


def snapshot_dispersion_metrics(robots, total_time, last_metric_time, interval=5.0):
    """Print dispersion metrics at intervals (nearest neighbor + hull)."""
    if total_time - last_metric_time >= interval:
        avg_nn = compute_avg_nearest_neighbor(robots)
        hull_area = compute_convex_hull_area(robots)

        print(f"[{total_time:5.1f}s] avg_nn={avg_nn:.2f}, hull_area={hull_area:.2f}")

        return total_time
    return last_metric_time


def snapshot_flocking_metrics(robots, total_time, last_metric_time, interval=5.0):
    """Print flocking metrics at intervals (alignment + cohesion + collisions + flock count)."""
    if total_time - last_metric_time >= interval:
        headings = np.array([r._heading for r in robots])
        heading_vecs = np.column_stack((np.cos(headings), np.sin(headings)))
        mean_vec = np.mean(heading_vecs, axis=0)
        alignment = np.linalg.norm(mean_vec)
        hull_area = compute_convex_hull_area(robots)
        avg_nn = compute_avg_nearest_neighbor(robots)
        num_flocks = compute_num_flocks(robots)

        print(
            f"[{total_time:5.1f}s] alignment={alignment:.2f}, "
            f"cohesion={avg_nn:.2f}, hull={hull_area:.2f}, "
            f"flocks={num_flocks}"
        )
        return total_time, num_flocks

    # return 0 if no metric snapshot is taken bc. of interval
    return last_metric_time, 0


def run_headless_once(runtime=120.0, swarm_mode=1, seed=None, interval=5.0):
    """Run one simulation in headless mode and return final metrics."""

    dt = SIM_DT
    robots = []

    if seed is not None:
        np.random.seed(seed)

    for i in range(NUM_ROBOTS):
        pos = np.random.uniform(
            [ROBOT_RADIUS, ROBOT_RADIUS],
            [WIDTH - ROBOT_RADIUS, HEIGHT - ROBOT_RADIUS],
        )
        heading = np.random.uniform(0, 2 * np.pi)
        robots.append(Robot(i, pos, heading))

    total_time = 0.0
    last_metric_time = 0.0
    list_num_flocks = []

    while total_time < runtime:
        total_time += dt

        for r in robots:
            r.read_sensors(robots, OBSTACLES, ARENA_BOUNDS)
        for r in robots:
            r.robot_controller(swarm_mode)
        for r in robots:
            r.move(dt)

        # --- metrics snapshot ---
        if swarm_mode == 1:
            last_metric_time = snapshot_dispersion_metrics(
                robots, total_time, last_metric_time, interval=interval
            )
        elif swarm_mode == 2:
            last_metric_time, last_num_flock = snapshot_flocking_metrics(
                robots,
                total_time,
                last_metric_time,
                interval=interval,
            )
            if last_num_flock:
                list_num_flocks.append(last_num_flock)

    # ===== FINAL METRICS =====
    if swarm_mode == 1:
        avg_nn = compute_avg_nearest_neighbor(robots)
        hull_area = compute_convex_hull_area(robots)
        return avg_nn, hull_area

    elif swarm_mode == 2:
        headings = np.array([r._heading for r in robots])
        heading_vecs = np.column_stack((np.cos(headings), np.sin(headings)))
        mean_vec = np.mean(heading_vecs, axis=0)
        alignment = np.linalg.norm(mean_vec)
        avg_nn = compute_avg_nearest_neighbor(robots)
        hull_area = compute_convex_hull_area(robots)
        return alignment, avg_nn, hull_area, list_num_flocks


def test(num_runs=5, runtime=120.0, swarm_mode=1, interval=5.0):
    """Run multiple headless experiments and print summary statistics."""

    if swarm_mode == 1:
        nn_values, hull_values = [], []
        for i in range(num_runs):
            avg_nn, hull_area = run_headless_once(
                runtime, swarm_mode, seed=i, interval=interval
            )
            nn_values.append(avg_nn)
            hull_values.append(hull_area)

        nn_mean, nn_min, nn_max = (
            np.mean(nn_values),
            np.min(nn_values),
            np.max(nn_values),
        )
        hull_mean, hull_min, hull_max = (
            np.mean(hull_values),
            np.min(hull_values),
            np.max(hull_values),
        )
        print("\n=== TEST SUMMARY over", num_runs, "runs ===")
        print(
            f"Nearest Neighbor Distance: avg={nn_mean:.2f}, "
            f"min={nn_min:.2f}, max={nn_max:.2f}"
        )
        print(
            f"Convex Hull Area: avg={hull_mean:.2f}, "
            f"min={hull_min:.2f}, max={hull_max:.2f}"
        )
        print("=======================================")
        return nn_mean, nn_min, nn_max, hull_mean, hull_min, hull_max

    if swarm_mode == 2:
        cluster_counts = {}
        alignments, cohesions, hull_values, list_num_flock_lists = [], [], [], []
        for i in range(num_runs):
            alignment, avg_nn, avg_hull_area, num_flocks = run_headless_once(
                runtime, swarm_mode, seed=i, interval=interval
            )
            alignments.append(alignment)
            cohesions.append(avg_nn)
            hull_values.append(avg_hull_area)
            list_num_flock_lists.append(num_flocks)

            # TOD: FIX, ONLY WORKS ON 1 ITERATION CURRENTLY
            for _, c in enumerate(num_flocks):
                cluster_counts[c] = cluster_counts.get(c, 0) + 1
            # get(key, orDefaultVal)
            percent_as_1_cluster = (cluster_counts.get(1, 0) / len(num_flocks)) * 100
            percent_as_2_cluster = (cluster_counts.get(2, 0) / len(num_flocks)) * 100
            percent_as_20_cluster = (cluster_counts.get(20, 0) / len(num_flocks)) * 100

        align_mean, align_min, align_max = (
            np.mean(alignments),
            np.min(alignments),
            np.max(alignments),
        )
        coh_mean, coh_min, coh_max = (
            np.mean(cohesions),
            np.min(cohesions),
            np.max(cohesions),
        )
        hull_mean, hull_min, hull_max = (
            np.mean(hull_values),
            np.min(hull_values),
            np.max(hull_values),
        )
        flock_mean, flock_min, flock_max = (
            np.mean(list_num_flock_lists),
            np.min(list_num_flock_lists),
            np.max(list_num_flock_lists),
        )

        print("\n=== FLOCKING TEST SUMMARY over", num_runs, "runs ===")
        print(
            f"Heading Alignment (0-1): avg={align_mean:.2f}, min={align_min:.2f}, max={align_max:.2f}"
        )
        print(
            f"Neighbor Distance (Cohesion): avg={coh_mean:.2f}, min={coh_min:.2f}, max={coh_max:.2f}"
        )
        print(
            f"Convex Hull Area: avg={hull_mean:.2f}, min={hull_min:.2f}, max={hull_max:.2f}"
        )
        print(
            f"Number of Flocks: avg={flock_mean:.2f}, min={flock_min:.2f}, max={flock_max:.2f}"
        )
        print(
            f"Flock percentage as 1: {percent_as_1_cluster:.2f}% count: {cluster_counts.get(1)}"
        )
        print(
            f"Flock percentage as 2: {percent_as_2_cluster:.2f}% count: {cluster_counts.get(2)}"
        )
        print("=======================================")

        cluster_counts = {}  # RESET

        return (
            align_mean,
            align_min,
            align_max,
            coh_mean,
            coh_min,
            coh_max,
            hull_mean,
            hull_min,
            hull_max,
            flock_mean,
            flock_min,
            flock_max,
        )


def main():
    clock = pygame.time.Clock()
    dt = SIM_DT
    robots = []

    np.random.seed(42)
    robots = init_robots(seed=42)

    logging_init()

    frame_count = 0
    total_time = 0.0
    running = True
    paused = False
    visualize = True
    last_metric_time = 0.0  # used for nearest neighbour clock
    swarm_mode = 1
    show_visual_lines = True
    metric_interval = 1.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_SPACE:
                    visualize = not visualize
                    print(
                        "Visualization",
                        "enabled" if visualize else "disabled",
                        "at",
                        total_time,
                    )
                elif event.key == pygame.K_1:
                    swarm_mode = 1
                elif event.key == pygame.K_2:
                    swarm_mode = 2
                elif event.key == pygame.K_3:
                    swarm_mode = 3
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_v:
                    show_visual_lines = not show_visual_lines
                elif event.key == pygame.K_r:
                    print("Restarting simulation (same seed)")
                    robots = init_robots(seed=42)
                    frame_count = 0
                    total_time = 0
                    last_metric_time = 0
                elif event.key == pygame.K_t:
                    print("Restarting simulation (random seed)")
                    robots = init_robots(seed=None)
                    frame_count = 0
                    total_time = 0
                    last_metric_time = 0

        if not paused:
            total_time += dt  # accumulate time

            for robot in robots:
                robot.read_sensors(robots, OBSTACLES, ARENA_BOUNDS)

            for robot in robots:
                robot.robot_controller(swarm_mode)

            for robot in robots:
                robot.move(dt)

            metrics = compute_metrics()
            log_metrics(frame_count, total_time, metrics)

            frame_count += 1

        # Every 'interval' seconds: pause, compute metric, resume
        if swarm_mode == 1:
            last_metric_time = snapshot_dispersion_metrics(
                robots, total_time, last_metric_time, interval=metric_interval
            )
        elif swarm_mode == 2:
            last_metric_time = snapshot_flocking_metrics(
                robots, total_time, last_metric_time, interval=metric_interval
            )

        if visualize:
            clock.tick(60 if not paused else 10)
            screen.fill(BG_COLOR)
            draw_light_sources(screen)
            draw_obstacles(screen)
            for robot in robots:
                robot.draw(screen, show_visual_lines)
            if paused:
                txt = font.render("PAUSED", True, (255, 100, 100))
                screen.blit(txt, (10, 10))
            pygame.display.flip()
            pygame.display.set_caption("Robot Sim — VISUAL MODE")
        else:
            pygame.display.set_caption(
                "Robot Sim — PAUSED in HEADLESS" if paused else "Robot Sim — HEADLESS"
            )

    pygame.quit()
    logging_close()


def print_dispersion_summary(nn_stats, hull_stats):
    nn_mean, nn_min, nn_max = nn_stats
    hull_mean, hull_min, hull_max = hull_stats

    print("Dispersion Test Summary:")
    print(
        f"Nearest Neighbor Distance: avg={nn_mean:.2f}, min={nn_min:.2f}, max={nn_max:.2f}"
    )
    print(
        f"Convex Hull Area: avg={hull_mean:.2f}, min={hull_min:.2f}, max={hull_max:.2f}"
    )
    print("=======================================")


def print_flocking_summary(align_stats, coh_stats, hull_stats, flock_stats):
    align_mean, align_min, align_max = align_stats
    coh_mean, coh_min, coh_max = coh_stats
    hull_mean, hull_min, hull_max = hull_stats
    flock_mean, flock_min, flock_max = flock_stats

    print("Flocking Test Summary:")
    print(
        f"Heading Alignment: avg={align_mean:.2f}, min={align_min:.2f}, max={align_max:.2f}"
    )
    print(
        f"Neighbor Distance (Cohesion): avg={coh_mean:.2f}, min={coh_min:.2f}, max={coh_max:.2f}"
    )
    print(
        f"Convex Hull Area: avg={hull_mean:.2f}, min={hull_min:.2f}, max={hull_max:.2f}"
    )
    print(
        f"Number of Flocks: avg={flock_mean:.2f}, min={flock_min:.2f}, max={flock_max:.2f}"
    )
    print("=======================================")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # print("Swarm mode: Dispersion\n")
        # (nn_mean, nn_min, nn_max, hull_mean, hull_min, hull_max) = test(
        #     num_runs=1, swarm_mode=1, interval=1.0
        # )

        print("\nSwarm mode: Flocking\n")
        (
            align_mean,
            align_min,
            align_max,
            coh_mean,
            coh_min,
            coh_max,
            hull_mean,
            hull_min,
            hull_max,
            flock_mean,
            flock_min,
            flock_max,
        ) = test(num_runs=1, swarm_mode=2, interval=1.0)

        print("\n============== Summaries ==============")

        # print_dispersion_summary(
        #     (nn_mean, nn_min, nn_max), (hull_mean, hull_min, hull_max)
        # )

        print_flocking_summary(
            (align_mean, align_min, align_max),
            (coh_mean, coh_min, coh_max),
            (hull_mean, hull_min, hull_max),
            (flock_mean, flock_min, flock_max),
        )
    else:
        main()  # run interactive
