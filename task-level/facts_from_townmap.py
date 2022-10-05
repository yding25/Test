#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is extrac facts from a map
# -----------------------------------------

from __future__ import print_function
import argparse
import glob
import logging
import os
import sys
import getpass
import weakref
import cv2
from queue import Queue, Empty
import numpy as np
import time
import math
import random
import torch
from collections import deque
try:
    # -----------------------------------------
    # find carla module
    # -----------------------------------------
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, sys.version_info.minor, 'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    # -----------------------------------------
    # find carla module (local)
    # -----------------------------------------
    sys.path.append('/home/yan/CARLA_0.9.10.1/PythonAPI/carla')
    # -----------------------------------------
    # add PythonAPI for release mode
    # -----------------------------------------
    sys.path.append(glob.glob('PythonAPI')[0])
except IndexError:
    pass
import carla
from carla import ColorConverter
from agents.navigation.basic_agent import BasicAgent
from models.model_supervised import Model_Segmentation_Traffic_Light_Supervised


# -----------------------------------------
# import customized functions
# -----------------------------------------
def get_waypoints_world(world, distance):
    map = world.get_map()
    waypoints = map.generate_waypoints(distance)
    print('number of waypoints:{}'.format(len(waypoints)))
    print('a sample waypoint:{}'.format(waypoints[0]))
    return waypoints


def sort_waypoints(waypoints_list, road_id_list, lane_id_list):
    # -----------------------------------------
    # sort waypoints: road_id (from min to max), lane_id(from min to max)
    # -----------------------------------------
    waypoints_list_sorted = []
    road_id_min = min(road_id_list)
    road_id_max = max(road_id_list)
    lane_id_min = min(lane_id_list)
    lane_id_max = max(lane_id_list)
    id_in_taskplanner = 1
    for road_id in range(road_id_min, road_id_max + 1):
        for lane_id in range(lane_id_min, lane_id_max + 1):
            signal_found = 0
            for element in waypoints_list:
                if element[0] == road_id and element[1] == lane_id:
                    signal_found = 1
                    element.append(id_in_taskplanner)
                    waypoints_list_sorted.append(element)
            if signal_found == 1:
                id_in_taskplanner = id_in_taskplanner + 1
    return waypoints_list_sorted


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
        # -----------------------------------------
        # connect server and load Town map
        # -----------------------------------------
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)
        world = client.get_world()
        origin_settings = world.get_settings()

        # -----------------------------------------
        # retrieve all waypoints from the world
        # -----------------------------------------
        waypoints = get_waypoints_world(world, 1.0)

        waypoints_list = []
        road_id_list = []  # used to sort
        lane_id_list = []  # used to sort
        for waypoint in waypoints:
            # print('------------analysis------------')
            # print('waypoint.road_id', waypoint.road_id)
            # print('waypoint.lane_id', waypoint.lane_id)
            print('waypoint.location(x,y,z)', waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z)
            # print('waypoint.rotation(yaw)', waypoint.transform.rotation.yaw)
            # print('---------------------------------')
            row = [waypoint.road_id, waypoint.lane_id, waypoint.transform.location.x, waypoint.transform.location.y,
                   waypoint.transform.rotation.yaw]
            waypoints_list.append(row)
            road_id_list.append(waypoint.road_id)
            lane_id_list.append(waypoint.lane_id)  # this value can be positive or negative which represents the
            # direction of the current lane
    except KeyboardInterrupt:
        # -----------------------------------------
        # reset the world
        # -----------------------------------------
        world.apply_settings(origin_settings)

        print('Cancelled by user. Bye!')
    
    waypoints_list_sorted = sort_waypoints(waypoints_list, road_id_list, lane_id_list)
    carla_version = 'CARLA_0.9.10.1'
    address1 = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/TMPUD/task-level/'
    fileout = open(address1 + 'waypoints_list_sorted.txt', 'w')  # store waypoints
    for row in waypoints_list_sorted:
        fileout.write('%d,%d,%0.6f,%0.6f,%0.6f,%d\n' % (row[0], row[1], row[2], row[3], row[4], row[5]))
    fileout.close()
    print('All is done!')


if __name__ == '__main__':
    main()
