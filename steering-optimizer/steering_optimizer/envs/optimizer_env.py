"""
Basic custom 2D steering geometry optimizer based on OpenAI Gym Examples
Implemented by Tamas Doka
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import circle_test as ct
from matplotlib import pyplot as plt
from pathlib import Path
# For interpolation
from scipy.interpolate import interp1d


# Changing the 4 variables: raising or lowering
N_DISCRETE_ACTIONS = 9
WHEELBASE = 1900
TRACK_WIDTH = 1200
KINGPIN = 150
TURNING_RADIUS = 4000
EPISODE_LENGTH = 200
MAX_ERROR = 10000


class StrOptEnv(gym.Env):
    """
    Description:
        The goal is to find the optimal steering geometry for a given wheelbase, track width and kingpin distance and
        a minimal turning radius by altering the parameters of the steering rack and the control arm, in a simplified
        2D top view representation.
    Source:
        This environment uses the analytical solution for steering angles of a front steered ideally turning four-wheel
        vehicle.
    Observation:
        Type: Box(5)
        Num	Observation                                     Min         Max
        0	Left Control Arm endpoint x-coordinate         -TW/2          0
        1	Left Control Arm endpoint y-coordinate         -TW/2        TW/2
        2	Steering rack left endpoint x-coordinate       -TW/2          0
        3	Steering rack left endpoint y-coordinate       -TW/2        TW/2

    Actions:
        Type: Discrete(9)
        Num	Action
        0	Increasing Left Control Arm endpoint x-coordinate
        1	Increasing Left Control Arm endpoint y-coordinate
        2	Increasing Steering rack left endpoint x-coordinate
        3	Increasing Steering rack left endpoint y-coordinate
        4	Decreasing Left Control Arm endpoint x-coordinate
        5	Decreasing Left Control Arm endpoint y-coordinate
        6	Decreasing Steering rack left endpoint x-coordinate
        7	Decreasing Steering rack left endpoint y-coordinate
        8   Do nothing

        Note: The amount of increasing or decreasing the coordinates is fixed in a constant
    Reward:
        Reward is the multiplicative inverse of the steering angle error from infinite to minimal turning radius.
    Starting State:
        ####All observations are assigned a uniform random value in [-0.05..0.05]
        Left Control arm x and y coordinates calculated from the Ackerman-angle
        and a random length between [0.15 .. 0.2]*TW, where TW is the track width of the vehicle.

        Steering rack x coordinate are a random value between -[0.15 .. 0.3]* TW
        Steering rack y coordinate are a random value between -[0.15 .. 0.3]* TW
    Episode Termination:
        ####Pole Angle is more than 12 degrees
        ####Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        ####Solved Requirements
        ####Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 24
    }

    def __init__(self):
        # Wheelbase length
        self.WB = WHEELBASE
        # Track width length
        self.TW = TRACK_WIDTH
        # Distance between the kingpin point and the center of the wheel
        self.KP = KINGPIN
        # Minimal turning radius
        self.tr_min = TURNING_RADIUS

        # Outer (right) wheel angle at minimum turning radius
        self.border_ang = np.arcsin(self.WB / (self.tr_min - self.KP))
        # Center of the front left wheel (x,y)
        self.WLX = -self.TW / 2
        self.WLY = 0.0
        # Front left kingpin point (x,y)
        self.KPLX = self.WLX + self.KP
        self.KPLY = self.WLY


        # Amount of increase or decrease
        self.amount = 1
        # Threshold for observations
        x_threshold = 2 * self.amount

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            0 + x_threshold,
            self.TW/2,
            0 + x_threshold,
            self.TW/2])

        low = np.array([
            -self.TW/2,
            -self.TW/2,
            -self.TW/2,
            -self.TW/2])

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None

        self.state = None
        self.reward = None
        self.error = None
        self.max_r = None

        self.total_reward = None

        self.steps_beyond_done = None
        self.steps_since_reset = None

        self.switch = np.zeros(4, )

        self.check_error = None
        self.check_r = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.steps_since_reset += 1

        state = self.state

        #### Action ###
        dx, dy, ax, ay = self._take_action(action, state)

        # initial (left) control arm length and angle
        arm_length = np.sqrt((ax - self.KPLX) ** 2 + (ay - self.KPLY) ** 2)
        beta_deg = np.arctan2(ay, ax - self.KPLX)/np.pi*180
        beta = beta_deg / 180 * np.pi
        # initial (right - outer) control arm angle
        beta_c_deg = -180 - beta_deg
        beta_c = beta_c_deg / 180 * np.pi

        # Tierod length
        tierod_length = np.sqrt(np.power((ax - dx), 2) + np.power((ay - dy), 2))

        # Calculating the maximum of rack travel to positive direction (right), which turns the wheels to left (positive
        # turn angles)
        max_ta_distance = tierod_length + arm_length
        max_ta_distance_x = np.sqrt(np.power(max_ta_distance, 2) - np.power(dy, 2))
        max_ta_distance_x_coord = max_ta_distance_x + self.KPLX
        rack_travel_max = max_ta_distance_x_coord - dx

        # Maximum rack travel until the counter arms becomes collinear
        min_ta_distance = np.abs(tierod_length - arm_length)
        # When the rack distance from the front axle is big enough, collinear configuration is not possible
        if np.abs(dy) > min_ta_distance:
            rack_travel = np.abs(rack_travel_max)

        else:
            # print('tierod length:', tierod_length)
            # print('arm_length:', arm_length)
            # print('min ta distance:', min_ta_distance)
            # print('min ta distance squared:', np.power(min_ta_distance, 2))
            # print('dy squared:', np.power(dy, 2))
            #
            # print('what goes to sqrt:', np.power(min_ta_distance, 2) - np.power(dy, 2))

            min_ta_distance_x = np.sqrt(np.power(min_ta_distance, 2) - np.power(dy, 2))
            min_ta_distance_x_coord = min_ta_distance_x + self.KPLX
            rack_travel_min = min_ta_distance_x_coord - dx

            # Locating the unstable configuration and choosing what we could reach sooner as rack travel in each direction
            if np.abs(rack_travel_max) > np.abs(rack_travel_min):
                rack_travel = np.abs(rack_travel_min)
            else:
                rack_travel = np.abs(rack_travel_max)

        self.rack_travel = rack_travel

        # Initial angle relations: alpha is the angle of the TA line
        init_alpha_deg = np.arctan((-self.KPLY + dy) / (-self.KPLX + dx)) / np.pi * 180
        ang_diff_deg = beta_deg - init_alpha_deg
        ang_diff_rad = ang_diff_deg / 180 * np.pi
        ang_diff_c_deg = beta_c_deg - init_alpha_deg
        ang_diff_c_rad = ang_diff_c_deg / 180 * np.pi

        # Rack travel from initial position
        x = 0
        x_array = np.array([])
        l_array = np.array([])
        r_array = np.array([])
        k_array = np.array([])

        # Data point number of rack travel - turning angle curve
        max_loop = 50
        loop_count = max_loop

        # We only check the turning angle error above minimal turning radius, so we throw away the unnecessary values
        integral_check = 0
        # Calculating the error curve in a loop
        while loop_count > 0:
            # Rack left endpoint position
            x_eval = dx + x
            x_array = np.append(x_array, x_eval - dx)

            # The actual TA line angle
            alpha = np.arctan2((-self.KPLY + dy), (-self.KPLX + x_eval))

            # Circles around the kingpin (A) and the left tierod endpoint (T)
            c1 = [0, 0, arm_length]
            c2 = [x_eval - self.KPLX, dy - self.KPLY, tierod_length]
            # Section points of the circles gives the mathematical solution for configuration
            c_sec = ct.Geometry().circle_intersection(c1, c2)
            betas = [np.arctan2(c_sec[1], c_sec[0]), np.arctan2(c_sec[3], c_sec[2])]

            # Turning angle of the left wheel
            if ang_diff_rad < 0 and (betas[0] - alpha) < 0:
                arm_a = betas[0] - beta
            else:
                arm_a = betas[1] - beta

            # Rack other side position
            x_eval_c = -dx + x

            alpha_c = np.arctan2((-self.KPLY + dy), (self.KPLX + x_eval_c))
            c1 = [0, 0, arm_length]
            c2 = [x_eval_c + self.KPLX, dy - self.KPLY, tierod_length]
            c_sec = ct.Geometry().circle_intersection(c1, c2)
            betas = [np.arctan2(c_sec[1], c_sec[0]), np.arctan2(c_sec[3], c_sec[2])]

            # TODO here could be the problem. Check how could arm_ca become negative.

            # Turning angle of the right wheel
            if ang_diff_c_rad > 0 and betas[0] - alpha_c >= 0:
                arm_ca = round(betas[0] - beta_c, 5)
            else:
                arm_ca = round(betas[1] - beta_c, 5)

            # if arm_ca < 0 and round(arm_ca, 5) == 0:
            #     #print('Warning: counter beta turned negative! Value: %f' % (arm_ca/np.pi*180))
            #     arm_ca = 0
            #     #print('Warning: counter beta corrected to zero! Value: %f' % (arm_ca / np.pi * 180))
            # elif arm_ca < 0:
            #     print('Warning: undetected error, counter beta-> outer turning angle is negative! Value: ', arm_ca, arm_ca*180/np.pi, 'deg')
            #     print('x_eval_c: ', x_eval_c)
            #     print('x: ', x)
            #     print('alpha_c[deg]: ', alpha_c*180/np.pi)
            #     print('ang_diff_c[deg]: ', ang_diff_c_deg)
            #     print('betas[deg]: ', np.asarray(betas) / np.pi * 180)
            #     print('beta_c[deg]: ', beta_c_deg)
            #     print('betas[deg] - beta_c: ', (np.asarray(betas) / np.pi * 180) - beta_c_deg)

            # If the outer (right) wheel angle is bigger than the border angle, the minimal turning radius is smaller
            # than the desired, which is not bad, but could cause large errors

            # Over the border angle
            if arm_ca > self.border_ang:
                integral_check += 1

            # Storing the angle values corresponding to the rack positions
            l_array = np.append(l_array, arm_a)
            r_array = np.append(r_array, arm_ca)

            x = x + (rack_travel / max_loop)

            loop_count = loop_count - 1

        # For every given position
        for i in range(0, len(x_array), 1):
            # Calculating the distance between the rear right kingpin point and the center of the turning circle
            # Because the outer (right) wheel determines the turning radius
            # When the wheels are facing front, the turning radius is infinite, so the tangent of 0 turning degree
            # becomes 0
            if round(r_array[i], 5) == 0:
                ideal_left_angle = 0
            else:
                dist = self.WB / np.tan(r_array[i])
                # Calculating the ideal inner (front left wheel) angle to the given turning radius
                ideal_left_angle = np.arctan2(self.WB, (dist - self.TW))

            k_array = np.append(k_array, ideal_left_angle)

        max_turning_angle = round(max(r_array), 5)

        if max_turning_angle == 0:
            tr_eval = np.inf
        else:
            tr_eval = np.sqrt((self.WB / np.tan(max_turning_angle)) ** 2 + self.WB ** 2) - self.KP

        # Calculating the error of the steering configuration
        # This array contains the difference between the ideal and the real value of the left turning angle
        # For every rack position
        error_array = np.power((k_array - l_array), 2)

        if integral_check > 0:
            # Integrating the error only above the minimal turning radius
            # First element index outside desired
            b_index = len(error_array) - integral_check

            # Values to keep
            error_array_mod = error_array[0:b_index - 1]
            r_array_mod = r_array[0:b_index - 1]

            # interpolation through border angle
            border_y = np.array([error_array[(b_index - 1)], error_array[b_index]])
            border_x = np.array([r_array[(b_index - 1)], r_array[b_index]])
            # Interpolating function
            f_inter = interp1d(border_x, border_y)

            # Calculating error at border angle
            border_error = f_inter(self.border_ang)

            error_array_mod = np.append(error_array_mod, border_error)
            r_array_mod = np.append(r_array_mod, self.border_ang)
        else:
            error_array_mod = error_array
            r_array_mod = r_array

        self.check_error = error_array_mod
        self.check_r = r_array_mod

        done = None

        # self.save_plot(error_array_mod, r_array_mod)

        unique = np.unique(r_array_mod)

        if len(unique) != len(r_array_mod):
            done = True
            error = MAX_ERROR
            print('Here is the problem!')
        else:
            error = np.trapz(error_array_mod * 100, r_array_mod * 100)

            #self.check_error = error
            self.state = (dx, dy, ax, ay)

            # Error is between 0 and 100000
            if abs(error) > MAX_ERROR:
                print('Top error reached', error)
                error = MAX_ERROR
                print('state: %f.3, %f.3, %f.3, %f.3' % dx, dy, ax, ay)

            # self.save_plot(error_array_mod, r_array_mod)

            if error < 0:
                print('Error is not valid!:', error)
                print('Invalid state:', self.state)

            # Integrating the total error
            # error_orig = np.trapz(error_array, r_array)

            # TODO write a function for printing curve plots to file
            # self.save_plot(error_array, r_array)

        # Getting the reward after action
        # Scenario 1: Max turning angle is under border angle.
        # This is the first step after initialization
        if self.max_r is None:
            reward = 0.0
        elif max(r_array) <= self.border_ang:
            # The max angle is bigger than the previous
            if max(r_array) > self.max_r:
                reward = 0.01
            else:
                reward = -0.01
        # Scenario 2: Max turning angle is above border angle
        else:
            # After the applied action the error got smaller
            if error < self.error:
                reward = 1.0
            # Being above max error, or consecutive no action (8)
            elif error == self.error:
                reward = -0.5
            # Error is getting bigger -> wrong action to take
            elif self.max_r <= self.border_ang:
                reward = 0.01
            else:
                reward = -1.0

        self.state = (dx, dy, ax, ay)
        self.max_r = max(r_array)
        self.error = error

        # Stepping out of boundaries
        done = dx > 0 or dx < -self.TW / 2 \
               or ax > 0 or ax < -self.TW / 2 \
               or dy > self.TW / 2 or dy < -self.TW / 2 \
               or ay > self.TW / 2 or ay < -self.TW / 2 \
               or self.steps_since_reset > EPISODE_LENGTH \
               or error < 0 or error == 0
        done = bool(done)

        if not done:
            self.reward = reward
            # If the turning radius is above desired the reward function scales down
            #if max_turning_angle < self.border_ang:
                # However we must give a reward for going towards border angle
                #reward = reward * 0.05 - reward * 0.02 * ((self.border_ang - max(r_array)) / self.border_ang)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True."
                    " You should always call 'reset()' once you receive 'done = True' "
                    "-- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
            print('total reward after episode: ', self.total_reward)

        self.total_reward += reward
        #print('total reward', self.total_reward)

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.ackerman_state()
        self.steps_beyond_done = None
        self.error = None
        self.max_r = None
        self.steps_since_reset = 0
        self.total_reward = 0

        return self.state

    def _take_action(self, action, state):

        amount = self.amount
        mod = np.array(state)

        for i in range(0, 3, 1):
            if action == i:
                self.switch[i] = -1

        for i in range(4, 7, 1):
            if action == i:
                self.switch[i-4] = 1

        if action == 8:
            self.switch = np.zeros((4,))

        for i in range(4):
            mod[i] = self.switch[i] * amount + state[i]
        return mod

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.TW + 200
        scale = screen_width / world_width

        #
        # carty = 100  # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * (2 * self.length)
        # cartwidth = 50.0
        # cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.geometry = rendering.Line((self.KPLX,
                                         self.state[2],
                                         self.state[0],
                                        -self.state[0],
                                        -self.state[2],
                                        -self.KPLX),
                                        (self.KPLY,
                                         self.state[3],
                                         self.state[1],
                                         self.state[1],
                                         self.state[3],
                                         self.KPLY))
            self.geometry.set_color(0, 0, 0)
            # self.geometrytrans = rendering.Transform()
            # self.geometry.add_attr(self.geometrytrans)
            self.viewer.add_geom(self.geometry)


            # l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            # axleoffset = cartheight / 4.0
            # cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            # self.carttrans = rendering.Transform()
            # cart.add_attr(self.carttrans)
            # self.viewer.add_geom(cart)
            #
            # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            # pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            # pole.set_color(.8, .6, .4)
            # self.poletrans = rendering.Transform(translation=(0, axleoffset))
            # pole.add_attr(self.poletrans)
            # pole.add_attr(self.carttrans)
            # self.viewer.add_geom(pole)
            # self.axle = rendering.make_circle(polewidth / 2)
            # self.axle.add_attr(self.poletrans)
            # self.axle.add_attr(self.carttrans)
            # self.axle.set_color(.5, .5, .8)
            # self.viewer.add_geom(self.axle)
            #
            # self._pole_geom = pole

        if self.state is None: return None

        # # Edit the pole polygon vertex
        # pole = self._pole_geom
        # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        # pole.v = [(l, b), (l, t), (r, t), (r, b)]
        #
        # x = self.state
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def ackerman_state(self):
        length = 100 + np.random.uniform(-1.0, 1.0) * 20
        ackerman_angle = np.arctan2(-2 * self.WB, self.TW)

        ax0 = self.KPLX + length * np.cos(ackerman_angle)
        ay0 = self.KPLY + length * np.sin(ackerman_angle)
        rack_x = -100 + np.random.uniform(-1.0, 1.0) * 20
        rack_y = -200 + np.random.uniform(-1.0, 1.0) * 20

        state = np.array([rack_x, rack_y, ax0, ay0])
        return state

    def save_plot(self, error_array, r_array):
        # Save error plot
        plt.plot(r_array/np.pi*180, error_array/np.pi*180)
        plt.axvline(x=self.border_ang/np.pi*180)
        plt.axis([20, (self.border_ang/np.pi*180)+1, 0, 0.2])

        filename = str(self.steps_since_reset) + '.png'
        pic = Path("pic/")
        pic_save_path = pic / filename
        plt.savefig(pic_save_path, bbox_inches='tight')


