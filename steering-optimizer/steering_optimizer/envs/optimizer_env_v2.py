"""
Basic custom 2D steering geometry optimizer based on OpenAI Gym Examples
Implemented by Tamas Doka
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.optimize import minimize

import circle_v2 as ct
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
EPISODE_LENGTH = 300
MAX_ERROR = 10000
ERROR_THRESHOLD = 0.01
MAX_LOOP = 50


def angle_chop(angle):
    chop = (angle*180/np.pi) % (np.sign(angle)*180)*np.pi/180
    real = -np.sign(angle)*np.pi + chop
    return real


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
        self.amount = 5
        # Threshold for observations
        x_threshold = 2 * self.amount

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            0 + x_threshold,
            self.TW / 2,
            0 + x_threshold,
            0 + x_threshold])

        low = np.array([
            -self.TW / 2,
            -self.TW / 2,
            -self.TW / 2,
            -self.TW / 2])

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

        done = None

        state = self.state

        #### Action ###
        dx, dy, ax, ay = self._take_action(action, state)

        # Stepping out of boundaries
        done = dx > 0 or dx < -self.TW / 2 \
               or ax > 0 or ax < -self.TW / 2 \
               or dy > self.TW / 2 or dy < -self.TW / 2 \
               or ay > self.TW / 2 or ay < -self.TW / 2

        done2 = abs(dx) > abs(ax)

        done = bool(done) or bool(done2)

        if done:
            return self.invalid_configuration()


        # Initial TA distance
        init_dist = np.sqrt((dx - self.KPLX) ** 2 + (dy - self.KPLY) ** 2)

        # initial (left) control arm length and angle
        arm_length = np.sqrt((ax - self.KPLX) ** 2 + (ay - self.KPLY) ** 2)

        # Tierod length
        tierod_length = np.sqrt((ax - dx) ** 2 + (ay - dy) ** 2)

        if arm_length + tierod_length < init_dist:
            print("WRONG initial calc! ARM, TIE, DIST", arm_length, tierod_length, init_dist)

        # TODO explain the geometrical background
        # Invalid configuration.
        # if init_dist < arm_length:
        #     print('Invalid configuration: arm is longer than initial distance')
        #     self.error = MAX_ERROR
        #     done = True
        #     reward = -5.0
        #
        #     #WITHOUT RESETTING!!!!
        #     #self.state = self.ackerman_state(True)
        #
        #     return np.array(self.state), reward, done, {}

        beta = np.arctan2(ay, ax - self.KPLX)
        beta_deg = beta * 180 / np.pi
        # initial (right - outer) control arm angle
        beta_c = np.sign(beta)*np.pi - beta
        beta_c_deg = beta_c * 180 / np.pi

        # TODO Should not cause problem
        if (round(beta_c, 5) == 0 or round(beta_c, 5) == 180 or round(beta_c, 5) == -180) and dy == 0:
            return self.invalid_configuration()

        # If beta is null we drop the solution because of possible inconsistency
        if (round(beta_c, 5) == 0 or round(beta_c, 5) == 180 or round(beta_c, 5) == -180):
            return self.invalid_configuration()

        #print('initial beta and beta_c', beta_deg, beta_c_deg, 'since reset', self.steps_since_reset, '%f, %f, %f, %f' % (dx, dy, ax, ay))

        # Calculating the maximum of rack travel to positive direction (right), which turns the wheels to left (positive
        # turn angles)
        max_ta_distance = tierod_length + arm_length
        max_ta_distance_x = np.sqrt(max_ta_distance ** 2 - (dy - self.KPLY) ** 2)
        max_ta_distance_x_coord = max_ta_distance_x + self.KPLX
        rack_travel_max = max_ta_distance_x_coord - dx

        maxtravel_check = True

        # Maximum rack travel until the counter arms becomes collinear
        min_ta_distance = np.abs(tierod_length - arm_length)
        # When the rack distance from the front axle is big enough, collinear configuration is not possible
        if min_ta_distance < np.abs(dy):
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

            # Locating the unstable configuration and choosing what we could reach sooner as rack travel in each
            # direction
            if np.abs(rack_travel_max) > np.abs(rack_travel_min):
                rack_travel = np.abs(rack_travel_min)
                maxtravel_check = False
            else:
                rack_travel = np.abs(rack_travel_max)

        self.rack_travel = rack_travel * 0.95

        # Initial angle relations: alpha is the angle of the TA line
        init_alpha_deg = np.arctan2((-self.KPLY + dy), (-self.KPLX + dx)) / np.pi * 180

        # if init_alpha_deg > 0:
        #     print('wrong_alpha', init_alpha_deg)
        #     print('dx, dy, ax, ay', dx, dy, ax, ay)

        ang_diff_deg = beta_deg - init_alpha_deg
        ang_diff_rad = ang_diff_deg / 180 * np.pi
        # ang_diff_c_deg = beta_c_deg - init_alpha_deg
        ang_diff_c_deg = beta_c_deg - (np.sign(beta_deg)*180 - init_alpha_deg)
        ang_diff_c_rad = ang_diff_c_deg / 180 * np.pi

        if np.sign(init_alpha_deg) == np.sign(beta_deg) and abs(init_alpha_deg) >= abs(beta_deg):
            return self.invalid_configuration()

        # setting direction of rack moving
        direction = 1

        if np.sign(beta) <= 0:

            direction = 1
        else:
            direction = -1

        if round(ang_diff_deg + ang_diff_c_deg, 5) != 0:
            print('Messed up initial condition!', ang_diff_deg, ang_diff_c_deg)

        # Near collinear initial arm!
        if abs(ang_diff_deg) < 10:
            return self.invalid_configuration()
        # else:
        #     print("Everything is in order, initial diff angle: ", ang_diff_deg)
        #     print("State: ", state)


        # Rack travel from initial position
        x = 0
        x_array = np.array([])
        l_array = np.array([])
        r_array = np.array([])
        k_array = np.array([])

        diag_array = np.array([])

        # Data point number of rack travel - turning angle curve
        loop_count = MAX_LOOP

        mark = None

        # We only check the turning angle error above minimal turning radius, so we throw away the unnecessary values
        integral_check = 0
        # Calculating the error curve in a loop
        while loop_count > 0:
            # Rack left endpoint position
            x_eval = dx + x
            x_array = np.append(x_array, x_eval - dx)

            # The actual TA line angle
            alpha = np.arctan2((-self.KPLY + dy), (-self.KPLX + x_eval))
            dist = np.sqrt((x_eval - self.KPLX) ** 2 + (dy - self.KPLY) ** 2)

            if dist > (arm_length + tierod_length):
                print(loop_count, "Ez valamiért itt szarul van.")


            if alpha > np.pi:
                print('alpha error in loop calculation!')
                #alpha -= 2 * np.pi

            # Circles around the kingpin (A) and the left tierod endpoint (T)
            c1 = [0, 0, arm_length]
            c2 = [x_eval - self.KPLX, dy - self.KPLY, tierod_length]


            if dist > (arm_length + tierod_length):
                print("#1 - separate dist, arm, tie -> NORMAL SIDE", dist, arm_length, tierod_length, "loop_count: ", loop_count, "maxtravelcheck:", maxtravel_check, "rack_travel", self.rack_travel)
                print(init_dist)
                print("state:", state)
                return self.invalid_configuration()

            if dist < abs(arm_length - tierod_length):
                print("#2 - containing dist, arm, tie -> NORMAL SIDE", dist, arm_length, tierod_length, "loop_count: ", loop_count, "maxtravelcheck:", maxtravel_check, "rack_travel", self.rack_travel)
                print(init_dist)
                print("state:", state)
                return self.invalid_configuration()

            if round(dist, 1) == 0 and round(arm_length, 3) == round(tierod_length, 3):
                print("#3 - coincident dist, arm, tie -> NORMAL SIDE", dist, arm_length, tierod_length, "loop_count: ", loop_count, "maxtravelcheck:", maxtravel_check, "rack_travel", self.rack_travel)
                print(init_dist)
                print("state:", state)
                return self.invalid_configuration()

            # Section points of the circles gives the mathematical solution for configuration
            c_sec = ct.Geometry().circle_intersection(c1, c2)

            if c_sec is None:
                print("circle_v2 error! dist, arm, tie -> NORMAL SIDE", init_dist, arm_length, tierod_length, "loop_count: ", loop_count, "maxtravelcheck:", maxtravel_check, "rack_travel", self.rack_travel)
                print(init_dist)
                print("state:", state)
                return self.invalid_configuration()

            betas = [np.arctan2(c_sec[1], c_sec[0]), np.arctan2(c_sec[3], c_sec[2])]

            # TODO all angles should be between -180 and 180

            diff = betas[0] - alpha

            # Turning angle of the left wheel
            if np.sign(ang_diff_rad) == np.sign(diff):
                arm_a = betas[0] - beta
            else:
                arm_a = betas[1] - beta

            if arm_a < 0 and round(arm_a, 5) == 0:
                arm_a = 0

            # Rack other side position
            x_eval_c = -dx + x

            alpha_c = np.arctan2((self.KPLY + dy), (self.KPLX + x_eval_c))

            if alpha_c > np.pi:
                alpha_c -= 2 * np.pi

            c1 = [0, 0, arm_length]
            c2 = [x_eval_c + self.KPLX, dy + self.KPLY, tierod_length]
            dist = np.sqrt((x_eval - self.KPLX) ** 2 + (dy - self.KPLY) ** 2)
            if dist > (arm_length + tierod_length):
                print("#1 - separate dist, arm, tie -> COUNTER SIDE", dist, arm_length, tierod_length, "loop_count: ", loop_count, "maxtravelcheck:", maxtravel_check, "rack_travel", self.rack_travel)
                print(init_dist)
                print("state:", state)
                return self.invalid_configuration()

            if dist < abs(arm_length - tierod_length):
                print("#2 - containing dist, arm, tie -> COUNTER SIDE", dist, arm_length, tierod_length, "loop_count: ", loop_count, "maxtravelcheck:", maxtravel_check, "rack_travel", self.rack_travel)
                print(init_dist)
                print("state:", state)
                return self.invalid_configuration()

            if round(dist, 1) == 0 and round(arm_length, 3) == round(tierod_length, 3):
                print("#3 - coincident dist, arm, tie -> COUNTER SIDE", dist, arm_length, tierod_length, "loop_count: ", loop_count, "maxtravelcheck:", maxtravel_check, "rack_travel", self.rack_travel)
                print(init_dist)
                print("state:", state)
                return self.invalid_configuration()

            # Section points of the circles gives the mathematical solution for configuration
            c_sec = ct.Geometry().circle_intersection(c1, c2)

            if c_sec is None:
                print("circle_v2 error! dist, arm, tie -> COUNTER SIDE", dist, arm_length, tierod_length, "loop_count: ", loop_count, "maxtravelcheck:", maxtravel_check, "rack_travel", self.rack_travel)
                print(init_dist)
                print("state:", state)
                return self.invalid_configuration()

            betas = [np.arctan2(c_sec[1], c_sec[0]), np.arctan2(c_sec[3], c_sec[2])]

            # TODO : IT IS A FORCED SOLUTION
            diff_c = angle_chop(round(betas[0] - alpha_c, 5))

            if x == 0:
                if round(betas[0] - beta_c, 10) == 0.000000:
                    mark = False
                    arm_ca = round(betas[0] - beta_c, 10)
                else:
                    mark = True
                    arm_ca = round(betas[1] - beta_c, 10)
            else:
                if mark:
                    arm_ca = betas[1] - beta_c
                else:
                    arm_ca = betas[0] - beta_c

            # TODO : FORCED SOLUTION END
            # Turning angle of the right wheel
            # if ((np.sign(ang_diff_c_rad) == np.sign(diff_c)) and (round(betas[1], 5) != round(beta_c, 5))) or (round(betas[0], 5) == round(beta_c, 5)):
            #     arm_ca = betas[0] - beta_c
            #     diag_array = np.append(diag_array, 0)
            # else:
            #     arm_ca = betas[1] - beta_c
            #     diag_array = np.append(diag_array, 1)

            if x == 0 and round(arm_ca, 5) != 0:
                print('Bad front facing because of betas!')
                print('Mark', mark)
                print('initial beta and beta_c', beta_deg, beta_c_deg)
                print('steps since reset', self.steps_since_reset)
                print(diff_c)
                print('signs of operands', np.sign(ang_diff_c_rad), np.sign(diff_c))
                print('Possible ARMS_ca', (betas[0] - beta_c)*180/np.pi, (betas[1] - beta_c)*180/np.pi)
                print('ARM_ca', arm_ca * 180 / np.pi)
                print('angdiff_c, 0:', ang_diff_c_deg, angle_chop(betas[0] - alpha_c) * 180 / np.pi)
                print('angdiff_c, 1:', ang_diff_c_deg, angle_chop(betas[1] - alpha_c) * 180 / np.pi)
                print('initalpha, alpha_c, betadeg, beta_c', init_alpha_deg, alpha_c*180/np.pi, beta_deg, beta_c_deg)
                print('diff_calc', - (np.sign(beta_deg)*180 - init_alpha_deg))

            if arm_ca < 0 and round(arm_ca, 5) == 0:
                arm_ca = 0

            if abs(arm_ca) > np.pi:
                #print('Possible ARMS_ca', (betas[0] - beta_c)*180/np.pi, (betas[1] - beta_c)*180/np.pi)
                #print('Possible ARMS_ca chop', angle_chop(betas[0] - beta_c) * 180 / np.pi, angle_chop(betas[1] - beta_c) * 180 / np.pi)
                arm_ca = angle_chop(arm_ca)

            # If the outer (right) wheel angle is bigger than the border angle, the minimal turning radius is smaller
            # than the desired, which is not bad, but could cause large errors

            # Over the border angle
            if arm_ca > self.border_ang:

                if x == 0:
                    print('Front facing is not zero')
                    print('ARM, border', arm_ca*180/np.pi, self.border_ang/np.pi*180, loop_count)
                    print('initial beta and beta_c', beta_deg, beta_c_deg)
                    print('alpha', init_alpha_deg, alpha*180/np.pi)
                    print('alpha_c', -180 - init_alpha_deg, alpha_c * 180 / np.pi)
                    print('betas:', betas[0] * 180 / np.pi, betas[1] * 180 / np.pi)
                    print('angdiff_c, 0:', ang_diff_c_deg, (betas[0] - alpha_c) * 180 / np.pi)
                    print('angdiff_c, 1:', ang_diff_c_deg, (betas[1] - alpha_c) * 180 / np.pi)
                    print('Front facing is not zero')

                integral_check += 1
                #print('arm_ca', arm_ca*180/np.pi, integral_check)

            # Storing the angle values corresponding to the rack positions
            l_array = np.append(l_array, arm_a)
            r_array = np.append(r_array, arm_ca)

            #diag_array = np.append(diag_array, diff_c)

            x = x + (rack_travel / MAX_LOOP)*direction

            loop_count = loop_count - 1

        # Checking if r_array is monotonic, if not the solution is bad

        def isMonotonic(A):

            return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or all(A[i] >= A[i + 1] for i in range(len(A) - 1)))

        # TODO: Sometimes not monotonic needs further investigation
        if isMonotonic(r_array) is False:
            #print('Not monotonic! -> Bad configuration! loop:', loop_count)
            #print('angdiff_c, 0:', ang_diff_c_deg, angle_chop(betas[0] - alpha_c) * 180 / np.pi)
            #print('angdiff_c, 1:', ang_diff_c_deg, angle_chop(betas[1] - alpha_c) * 180 / np.pi)
            #print('initial beta and beta_c', beta_deg, beta_c_deg)
            #print(r_array*180/np.pi)
            #print(diag_array)
            done = True
            reward = 0.0
            return np.array(self.state), reward, done, {}

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
            # TODO Not works properly!!! if border_angle is not valid, the error isn't valid
            if self.border_ang <= border_x[0] or self.border_ang >= border_x[1]:

                self.error = MAX_ERROR
                done = True
                print('Done: Border angle error')
                reward = -5.0

                print('border_ang error! It is not between borders', border_x)
                print('border_ang value: ', self.border_ang)
                print('border_ang value: ', self.border_ang*180/np.pi)
                print('b_index ', b_index)
                print('Integral check ', integral_check)
                print(r_array[b_index - 2])
                print(r_array[b_index - 1])
                print(r_array[b_index])
                print(r_array[b_index + 1])
                #print(r_array_mod)


                # WITHOUT RESETTING ENV!!!
                #self.state = self.ackerman_state(True)

                return np.array(self.state), reward, done, {}



                #border_error = (error_array[(b_index - 1)] + error_array[b_index]) / 2
            else:
                border_error = f_inter(self.border_ang)

            error_array_mod = np.append(error_array_mod, border_error)
            r_array_mod = np.append(r_array_mod, self.border_ang)
        else:
            error_array_mod = error_array
            r_array_mod = r_array

        self.check_error = error_array_mod
        self.check_r = r_array_mod



        # self.save_plot(error_array_mod, r_array_mod)

        unique = np.unique(r_array_mod)

        if len(unique) != len(r_array_mod):
            done = True

            #print('Done: Array length error')
            #print('Step since reset: ', self.steps_since_reset)
            error = MAX_ERROR
            #print('Here is the problem! -> not every angle value is unique')
            #print('r_array_mod', r_array_mod)
            #input('Press enter!')
            return self.invalid_configuration()
        else:
            error = np.trapz(error_array_mod * 180/np.pi, r_array_mod * 180/np.pi)

            self.error = error

            # self.check_error = error
            self.state = (dx, dy, ax, ay)

            # Error is between 0 and 100000
            #if abs(error) > MAX_ERROR:
            #    print('Top error reached', error)
            #    error = MAX_ERROR

            # Not normal configuration
            if error < 0:
                #print('Error is not valid!:', error)
                #print(self.check_r)
                #print(self.check_error)
                #print('Invalid state:', self.state)

                #self.save_plot(error_array_mod, r_array_mod, False)
                done = True
                reward = -1
                # print('Done: not coherent turning direction')
                return np.array(self.state), reward, done, {}


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
            elif error == self.error:
                reward = -0.0005
            else:
                reward = -0.01
        # Scenario 2: Max turning angle is above border angle
        else:
            # After the applied action the error got smaller
            if error < self.error:
                reward = 1.0
            # Being above max error, or consecutive no action (8)
            elif error == self.error:
                reward = -0.05
            # Error is getting bigger -> wrong action to take
            elif self.max_r <= self.border_ang:
                reward = 0.01
            else:
                reward = -0.5

        self.state = (dx, dy, ax, ay)
        self.max_r = max(r_array)
        self.error = error

        if error < ERROR_THRESHOLD and self.max_r > self.border_ang:
            done = True
            #print('Done: Reached error threshold!')
            reward = 100

        if done is None:

            done = self.steps_since_reset > EPISODE_LENGTH
            done = bool(done)
            if done:
                print('Done: Episode ended')
            else:
                done = error == 0
                done = bool(done)

        # TODO error < 0 is not handled

        if not done:
            self.reward = reward
            # If the turning radius is above desired the reward function scales down
            # if max_turning_angle < self.border_ang:
            # However we must give a reward for going towards border angle
            # reward = reward * 0.05 - reward * 0.02 * ((self.border_ang - max(r_array)) / self.border_ang)
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
            #print('total reward after episode: ', self.total_reward)

        self.total_reward += reward
        # print('total reward', self.total_reward)

        self.state = self.state

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

        if action == 0:
            return state
        elif action == 1:
            mod[0] = state[0] - amount
        elif action == 2:
            mod[0] = state[0] + amount
        elif action == 3:
            mod[1] = state[1] - amount
        elif action == 4:
            mod[1] = state[1] + amount
        elif action == 5:
            mod[2] = state[2] - amount
        elif action == 6:
            mod[2] = state[2] + amount
        elif action == 7:
            mod[3] = state[3] - amount
        elif action == 8:
            mod[3] = state[3] + amount
        return mod

        ### SWITCH METHOD ###

        # for i in range(0, 3, 1):
        #     if action == i:
        #         self.switch[i] = -1
        #
        # for i in range(4, 7, 1):
        #     if action == i:
        #         self.switch[i-4] = 1
        #
        # if action == 8:
        #     self.switch = np.zeros((4,))
        #
        # for i in range(4):
        #     mod[i] = self.switch[i] * amount + state[i]
        # return mod

    def render(self, mode='human'):
        screen_width = self.TW
        screen_height = self.TW

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

            #self.KPLX += self.TW/2
            #self.KPLY += self.TW / 2
            #self.state += self.TW / 2

            # self.geometry = rendering.Line((self.KPLX,
            #                                 self.state[2],
            #                                 self.state[0],
            #                                 -self.state[0],
            #                                 -self.state[2],
            #                                 -self.KPLX),
            #                                (self.KPLY,
            #                                 self.state[3],
            #                                 self.state[1],
            #                                 self.state[1],
            #                                 self.state[3],
            #                                 self.KPLY))
            self.line_0 = rendering.Line((self.KPLX, self.KPLY), (self.state[2], self.state[3]))
            self.line_1 = rendering.Line((self.state[2], self.state[3]), (self.state[0], self.state[1]))
            self.line_2 = rendering.Line((self.state[0], self.state[1]), (-self.state[0] + self.TW, self.state[1]))
            self.line_3 = rendering.Line((-self.state[0] + self.TW, self.state[1]), (-self.state[2] + self.TW, self.state[3]))
            self.line_4 = rendering.Line((-self.state[2] + self.TW, self.state[3]), (-self.KPLX + self.TW, self.KPLY))
            self.line_5 = rendering.Line((-self.KPLX, self.KPLY))

            self.line_0.set_color(0, 0, 0)
            self.line_1.set_color(0, 0, 0)
            self.line_2.set_color(0, 0, 0)
            self.line_3.set_color(0, 0, 0)
            self.line_4.set_color(0, 0, 0)
            self.line_5.set_color(0, 0, 0)
            # self.geometrytrans = rendering.Transform()
            # self.geometry.add_attr(self.geometrytrans)

            self.viewer.add_geom(self.line_0)
            self.viewer.add_geom(self.line_1)
            self.viewer.add_geom(self.line_2)
            self.viewer.add_geom(self.line_3)
            self.viewer.add_geom(self.line_4)
            #self.viewer.add_geom(self.line_5)

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

        #self.KPLX -= self.TW / 2
        #self.KPLY -= self.TW / 2
        #self.state -= self.TW / 2

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def ackerman_state(self, random=True):
        if random:
            # length = 100 + np.random.uniform(-1.0, 1.0) * 20
            # rack_x = -100 + np.random.uniform(-1.0, 1.0) * 20
            # rack_y = -200 + np.random.uniform(-1.0, 1.0) * 20
            # rand = np.random.uniform(-2.0, 2.0)
            # #rand = 1

            length = np.random.uniform(0.2, 1.0) * 120
            rack_x = np.random.uniform(-1.0, 0.2) * 120
            rack_y = np.random.uniform(-1.0, 1.0) * 120
            rand = np.random.uniform(-2.0, 2.0)
            # rand = 1

        else:
            length = 100
            rack_x = -100
            rack_y = -200
            rand = 1

        ackerman_angle = np.arctan2(-2 * self.WB, self.TW)*rand
        ax0 = self.KPLX + length * np.cos(ackerman_angle)
        ay0 = self.KPLY + length * np.sin(ackerman_angle)
        state = np.array([rack_x, rack_y, ax0, ay0])

        return state

    def save_plot(self, error_array, r_array, sized=True):
        # Save error plot
        plt.plot(r_array / np.pi * 180, error_array / np.pi * 180)
        plt.axvline(x=self.border_ang / np.pi * 180)
        if sized:
            plt.axis([20, (self.border_ang / np.pi * 180) + 1, 0, 0.2])

        #filename = str(self.steps_since_reset) + '.png'
        filename = str(np.random.randint(1000)) + '.png'
        pic = Path("pic/")
        pic_save_path = pic / filename
        plt.savefig(pic_save_path, bbox_inches='tight')

    def check_version(self):
        # Print version
        print('StrOpt version: dev')

    def objective(self, x):
        self.reset()
        self.state = x
        # No action during step
        self.step(0)

        # print(self.border_ang, self.max_r)

        if self.border_ang > self.max_r:
            self.error += (self.border_ang - self.max_r)*max(self.check_error)*(180/np.pi)**2

        return self.error

    def geometry_optimize(self):
        # Lower and upper bounds
        b1 = (-self.TW / 2, 0)
        b2 = (-self.TW / 2, self.TW / 2)
        b3 = (-self.TW / 2, 0)
        bnds = (b1, b2, b1, b3)
        # Initial position
        initial_state = self.ackerman_state(False)

        solution = minimize(self.objective, initial_state, method='SLSQP', bounds=bnds)

        return solution

    def invalid_configuration(self):
        done = True
        reward = -1
        self.error = None
        return np.array(self.state), reward, done, {}