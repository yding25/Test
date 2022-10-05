#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ??
# 加载需要的包和环境
from __future__ import print_function
import argparse
import glob
import logging
import os
import sys
import numpy as np
import time
import math
import random
import statistics
import getpass

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from carla import ColorConverter as cc

try:
    sys.path.append('/home/yan/CARLA_0.9.10.1/PythonAPI/carla')
except IndexError:
    pass

from agents.navigation.basic_agent import BasicAgent  # 创造自己的运动模型，可以设置不同的速度参数

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('PythonAPI')[0])
except IndexError:
    pass

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# -----------------------------------------
# debug for where is the car and how car moves
# -----------------------------------------
carla_version = 'CARLA_0.9.10.1'
address1 = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/Test/'


# -----------------------------------------
# functions
# -----------------------------------------
def create_car_ego(world, car_ego_make, car_ego_color, car_ego_name, source_state):
    blueprint_library = world.get_blueprint_library()  # get blueprint library
    bp_ego = blueprint_library.filter(car_ego_make)[0]
    bp_ego.set_attribute('color', car_ego_color)
    bp_ego.set_attribute('role_name', car_ego_name)
    state_ego = carla.Transform(carla.Location(x=source_state[0], y=source_state[1], z=1.2),
                                carla.Rotation(pitch=0, yaw=source_state[2], roll=0))
    car_ego = world.spawn_actor(bp_ego, state_ego)
    return car_ego


def direct_twoPoints(x, y):
    origin_x = x[0]
    origin_y = x[1]
    destination_x = y[0]
    destination_y = y[1]
    deltaX = destination_x - origin_x
    deltaY = destination_y - origin_y
    degrees_temp = math.atan2(deltaX, deltaY) / math.pi * 180
    if degrees_temp < 0:
        degrees_final = 360 + degrees_temp
    else:
        degrees_final = degrees_temp
    return degrees_final


def game_loop(args):
    actor_list = []

    try:
        client = carla.Client(args.host, args.port)  # get client
        client.set_timeout(4.0)

        world = client.get_world()
        curr_map = world.get_map()
        print(curr_map.name)
        if curr_map.name != 'Town05':
            world = client.load_world('Town05')

        synchronous_master = False
        # -----------------------------------------
        # spawn current location and destination of our ego car
        # -----------------------------------------
        wayPoints = np.load(address1 + 'interaction/wayPoints.npy')
        mergelane = np.load(address1 + 'interaction/mergelane.npy')
        numLane = np.load(address1 + 'interaction/numLane.npy')

        for [left_lane, right_lane] in mergelane:
            # debug
            target_lane = 37
            if int(right_lane) == target_lane:

                for i in range(0, len(wayPoints)):
                    if wayPoints[i][5] == right_lane:
                        lane_length = numLane[int(right_lane) - 1]
                        front_waypoint_right = wayPoints[i + round(int(lane_length)/3)]
                        back_waypoint_right = wayPoints[i + lane_length - 1]
                        break

                for i in range(0, len(wayPoints)):
                    if wayPoints[i][5] == left_lane:
                        lane_length = numLane[int(left_lane) - 1]
                        front_waypoint_left = wayPoints[i + round(int(lane_length)/3)]
                        back_waypoint_left = wayPoints[i + lane_length - 1]
                        break

                # option 1
                source_state1 = [back_waypoint_right[2], back_waypoint_right[3], back_waypoint_right[4]]
                dest_state1 = [front_waypoint_left[2], front_waypoint_left[3], front_waypoint_left[4]]
                x_option1 = [source_state1[1], source_state1[0]]
                y_option1 = [dest_state1[1], dest_state1[0]]
                direction1 = direct_twoPoints(x_option1, y_option1)

                # option 2
                source_state2 = [front_waypoint_right[2], front_waypoint_right[3], front_waypoint_right[4]]
                dest_state2 = [back_waypoint_left[2], back_waypoint_left[3], back_waypoint_left[4]]
                x_option2 = [source_state2[1], source_state2[0]]
                y_option2 = [dest_state2[1], dest_state2[0]]
                direction2 = direct_twoPoints(x_option2, y_option2)

                if abs(direction1 - source_state1[2]) < abs(direction2 - source_state1[2]):
                    source_state = source_state1
                    dest_state = dest_state1
                else:
                    source_state = source_state2
                    dest_state = dest_state2

        # 0, 1, 37, 27.372021, -128.617462, 91.532082
        # 1, 0, 38, 30.229090, -104.532463, 91.532082
        # 2, 0, 42, 29.058886, -60.780251, 91.532082
        # 3, 0, 105, 41.152187, 1.743095, 0.250259
        # 4, 0, 139, 123.816780, 1.448214, 0.255043
        # 5, 0, 47, 41.222652, 141.436386, -177.771576
        #
        source_state = [27.900251, 12.570639, 90.224159]
        # dest_state = [29.058886, -60.780251, 91.532082]

        print('source_state = ', source_state)
        print('dest_state = ', dest_state)
        print('trip distance:', math.sqrt((source_state[0] - dest_state[0]) ** 2 + (source_state[1] - dest_state[1]) ** 2))
        # -----------------------------------------
        # create our ego car：make; color; name; initial state
        # -----------------------------------------
        x = [source_state[0], source_state[1]]
        y = [dest_state[0], dest_state[1]]
        direction = direct_twoPoints(x, y)
        print(direction)

        car_ego_make = 'vehicle.audi.tt'
        car_ego_color = '0,0,255'
        car_ego_name = 'car_ego'
        car_ego = create_car_ego(world, car_ego_make, car_ego_color, car_ego_name, source_state)
        actor_list.append(car_ego)
        print('ego car is created and its id:', car_ego.id)

        # -----------------------------------------
        # implement agents: basic_agent.py
        # -----------------------------------------
        agent_ego = BasicAgent(car_ego)
        agent_ego.set_destination((dest_state[0], dest_state[1], 1.2))

        ego_loc = car_ego.get_location()  # current location of our ego car
        mini_dis = 1  # a minimal distance to check if our ego car achieves the destination
        temp_dist = math.sqrt((ego_loc.x - dest_state[0]) ** 2 + (ego_loc.y - dest_state[1]) ** 2)
        while temp_dist > mini_dis:
            temp_dist = math.sqrt((ego_loc.x - dest_state[0]) ** 2 + (ego_loc.y - dest_state[1]) ** 2)
            print(temp_dist)
            print('location x and y: %f, %f' % (ego_loc.x, ego_loc.y))
            print('target location x and y: %f, %f' % (dest_state[0], dest_state[1]))
            if not world.wait_for_tick():  # as soon as the server is ready continue!
                continue
            control = agent_ego.run_step()
            car_ego.apply_control(control)
            control.manual_gear_shift = False
            ego_loc = car_ego.get_location()

    finally:
        for actor in actor_list:  # delete our ego car
            actor.destroy()
        print("ALL cleaned up!")


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================
def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('Cancelled by user. Bye!')


if __name__ == '__main__':
    main()
