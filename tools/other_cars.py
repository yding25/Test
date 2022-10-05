#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to get other cars information
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
from utils import compute_direction, generate_navigation


def game_loop(args):
    # -----------------------------------------
    # connect server and load Town map
    # -----------------------------------------
    client = carla.Client(args.host, args.port)
    client.set_timeout(4.0)
    world = client.get_world()
    origin_settings = world.get_settings()
    synchronous_master = False
    curr_map = world.get_map()
    print('current map: {}'.format((curr_map.name)))
    if curr_map.name != 'Town05':
        world = client.load_world('Town05')

    # get library of other cars
    blueprints_vehicle = world.get_blueprint_library().filter("vehicle.*")
    blueprints_vehicle = [x for x in blueprints_vehicle if int(x.get_attribute('number_of_wheels')) == 4]
    blueprints_vehicle = sorted(blueprints_vehicle, key=lambda bp: bp.id)
    print('number of other cars: {}'.format(len(blueprints_vehicle)))

    other_cars = []
    for _ in range(1000):
        blueprint = random.choice(blueprints_vehicle)
        if blueprint.id not in other_cars: 
            other_cars.append(blueprint.id)
            print('{}'.format(blueprint.id))
    
# volkswagen t2
# tesla model3
# jeep wrangler_rubicon
# carlamotors carlacola
# mini cooperst
# nissan micra
# bmw isetta
# dodge_charger police
# lincoln mkz2017
# citroen c3
# bmw grandtourer
# seat leon
# chevrolet impala
# audi tt
# nissan patrol
# audi a2
# audi etron
# mustang mustang
# mercedes-benz coupe
# tesla cybertruck
# toyota prius



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
    argparser.add_argument(
        '-d', '--number-of-dangerous-vehicles',
        metavar='N',
        default=0,
        type=int,
        help='number of dangerous vehicles (default: 3)')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        default=True,
        help='Synchronous mode execution')
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
