#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to collect data! (low GPU only)
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
    sys.path.append('/home/kitrob/CARLA_0.9.10.1/PythonAPI/carla')
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

class EgoCarManager:
    def __init__(self, world):
        self.world = world
    
    def create_car(self, car_ego_make, car_ego_color, car_ego_name, init_state):
        blueprint_library = self.world.get_blueprint_library()  # get blueprint library
        bp_ego = blueprint_library.filter(car_ego_make)[0]
        bp_ego.set_attribute('color', car_ego_color)
        bp_ego.set_attribute('role_name', car_ego_name)
        state_ego = carla.Transform(carla.Location(x=init_state[0], y=init_state[1], z=1.2),
                                    carla.Rotation(pitch=0, yaw=init_state[2], roll=0))
        car_ego = self.world.spawn_actor(bp_ego, state_ego)
        car_ego.set_simulate_physics(True)
        return car_ego


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick(1.0)
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class CameraManager:
    def __init__(self, world):
        self.world = world
        self.actors = []

    def set_up_sensors(self, vehicle):
        blueprint_library = self.world.get_blueprint_library()
        bp_camera = blueprint_library.find('sensor.camera.rgb')
        camera_rgb = self.world.spawn_actor(
            bp_camera,
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        self.actors.append(camera_rgb)

        bp_camera_288 = blueprint_library.find('sensor.camera.rgb')
        bp_camera_288.set_attribute('image_size_x', "288")
        bp_camera_288.set_attribute('image_size_y', '288')

        camera_front = self.world.spawn_actor(
            bp_camera_288,
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0)),
            attach_to=vehicle)
        self.actors.append(camera_front)

        camera_back = self.world.spawn_actor(
            bp_camera_288,
            carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(pitch=0, yaw=180)),
            attach_to=vehicle)
        self.actors.append(camera_back)

        camera_left = self.world.spawn_actor(
            bp_camera_288,
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0, yaw=90)),
            attach_to=vehicle)
        self.actors.append(camera_left)

        camera_right = self.world.spawn_actor(
            bp_camera_288,
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0, yaw=270)),
            attach_to=vehicle)
        self.actors.append(camera_right)
        return camera_rgb, camera_front, camera_back, camera_left, camera_right


class CollisionManager:
    flag = 0  # no collision

    def __init__(self, world):
        self.collision_sensor = None
        self.world = world
        self.blueprints = self.world.get_blueprint_library()

    def set_up_sensors(self, vehicle):
        self.collision_sensor = self.world.spawn_actor(
            self.blueprints.find("sensor.other.collision"),
            carla.Transform(),
            attach_to=vehicle,
        )
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: CollisionManager._on_collision(weak_self, event))
        return self.collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        CollisionManager.flag = 1


class SupervisedEncoder:
    def __init__(self):
        self.steps_image = [-10, -2, -1, 0]
        # self.model_supervised = Model_Segmentation_Traffic_Light_Supervised(len(self.steps_image),
        #                                                                     len(self.steps_image), 1024, 6, 4,
        #                                                                     crop_sky=False)
        # self.device = torch.device("cuda")
        # path_to_encoder_dict = path1 + 'interaction/model_epoch_34.pth'
        # self.model_supervised.load_state_dict(torch.load(path_to_encoder_dict, map_location=self.device))
        # self.model_supervised.to(device=self.device)
        # self.encoder = self.model_supervised.encoder
        # self.last_conv_downsample = self.model_supervised.last_conv_downsample
        self._rgb_queue = None
        self.window = (max([abs(number) for number in self.steps_image]) + 1)
        self.RGB_image_buffer = deque([], maxlen=self.window)
        for _ in range(self.window):
            self.RGB_image_buffer.append(np.zeros((3, 288, 288)))
        self.render = True

    def carla_img_to_np(self, carla_img):
        carla_img.convert(ColorConverter.Raw)
        img = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (carla_img.height, carla_img.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]  # carla uses bgr so need to convert it to rgb
        return img

    def run_step(self, image, name, snapshot, save_flag=False):
        rgb = self.carla_img_to_np(image).copy()
        rgb = np.array(rgb)

        if self.render:
            bgr = rgb[:, :, ::-1]
        rgb = np.rollaxis(rgb, 2, 0)
        self.RGB_image_buffer.append(rgb)

        np_array_RGB_input = np.concatenate(
            [
                self.RGB_image_buffer[indice_image + self.window - 1]
                for indice_image in self.steps_image
            ]
        )

        # -----------------------------------------
        #  save temp vectored image
        # -----------------------------------------
        if save_flag:
            np.save(path1 + 'data_for_model/raw_data/encoded_png/' + str(init_time) + '_' + name + '_' + str(snapshot), np_array_RGB_input)

        if save_flag:
            for indice_image in self.steps_image:
                new_image = self.RGB_image_buffer[indice_image + self.window - 1].copy()
                new_image = np.rollaxis(new_image, 0, 3)
                new_bgr = new_image[:, :, ::-1]
                cv2.imwrite(
                    path1 + 'data_for_model/raw_data/raw_png/' + str(init_time) + '_' + snapshot + "_" + str(
                        abs(indice_image)) + "_" + name + ".png",
                    new_bgr)

        # torch_tensor_input = (
        #     torch.from_numpy(np_array_RGB_input)
        #         .to(dtype=torch.float32, device=self.device)
        #         .div_(255)
        #         .unsqueeze(0)
        # )

        # with torch.no_grad():
        #     current_encoding = self.encoder(torch_tensor_input)
        #     current_encoding = self.last_conv_downsample(current_encoding)

        # current_encoding_np = current_encoding.cpu().numpy().flatten()
        # return current_encoding_np

# -----------------------------------------
# this part is to generate a set of folders to store collected data
# - data_for_model
#     - raw_data
#         - raw_frame
#         - raw_png
#         - recording
# -----------------------------------------
global path1, signal_stop, signal_collision, signal_merge, signal_error, init_time
carla_version = 'CARLA_0.9.10.1'
path1 = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/Test/'
init_time = int(time.time())
if not os.path.exists(path1 + 'data_for_model/'):
    os.mkdir(path1 + 'data_for_model/')
if not os.path.exists(path1 + 'data_for_model/raw_data/'):
    os.mkdir(path1 + 'data_for_model/raw_data/')
if not os.path.exists(path1 + 'data_for_model/raw_data/raw_frame/'):
    os.mkdir(path1 + 'data_for_model/raw_data/raw_frame/')
if not os.path.exists(path1 + 'data_for_model/raw_data/encoded_png/'):
    os.mkdir(path1 + 'data_for_model/raw_data/encoded_png/')
if not os.path.exists(path1 + 'data_for_model/raw_data/raw_png/'):
    os.mkdir(path1 + 'data_for_model/raw_data/raw_png/')
if not os.path.exists(path1 + 'data_for_model/raw_data/recording/'):
    os.mkdir(path1 + 'data_for_model/raw_data/recording/')
# -----------------------------------------
# this folder is to record each trial's information
# -----------------------------------------
fidin = open(path1 + 'data_for_model/raw_data/overview.txt', 'a')

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
    
    # -----------------------------------------
    # a few signals that terminate the runnning process
    # -----------------------------------------
    signal_stop = False
    signal_collision = False
    signal_merge = False
    signal_error = False

    # -----------------------------------------
    # a list that contains all spawned actors
    # -----------------------------------------
    actor_list = []

    # -----------------------------------------
    # The collection is relevant to parameters
    # -----------------------------------------
    params = {
        'target_lane': 21, # target lane id
        'min_num': 10, # min number of other vehicles
        'max_num': 20, # max number of other vehicles
        'density': 1.5, # density of spawn points
        'min_range': 5.0, # min distance between ego car and other vehicles
        'max_range': 60.0, # max distance between ego car and other vehicles
        'index_img': 5, # index of saved image
        'index_frame': 5,
        'min_dis': 10, # min distance between ego car and destination
        'max_dis': 10, # max distance ego car can run
        'threshold_dis': 5, # min distance between ego car and other vehicles
        'max_angle': 30, # min angle between ego car and other vehicles
        'small_throttle': 0.3 # a small throttle value
        }

    try:
        # -----------------------------------------
        # set target_lane
        # -----------------------------------------
        target_lane = params['target_lane']
        print('target_lane: {}'.format(target_lane))

        # -----------------------------------------
        # use .log to record running process (video)
        # -----------------------------------------
        recording = path1 + 'data_for_model/raw_data/recording/' + str(init_time) + '.log'
        client.start_recorder(recording)

        # -----------------------------------------
        # compute source and destination navigation goals
        # -----------------------------------------
        source_state, dest_state = generate_navigation(path1, target_lane)
        print('source_state: {}, dest_state: {}'.format(source_state, dest_state))

        # -----------------------------------------
        # create our ego car
        # -----------------------------------------
        car_ego_manager = EgoCarManager(world)
        car_ego_make = 'vehicle.audi.tt'
        car_ego_color = '0,0,255'
        car_ego_name = 'car_ego'
        car_ego = car_ego_manager.create_car(car_ego_make, car_ego_color, car_ego_name, source_state)
        actor_list.append(car_ego)

        # -----------------------------------------
        # implement run mode
        # -----------------------------------------
        agent_ego = BasicAgent(car_ego)
        agent_ego.set_destination((dest_state[0], dest_state[1], 1.2))

        # -----------------------------------------
        #  create four cameras
        # -----------------------------------------
        camera_manager = CameraManager(world)
        camera, *cameras_288 = camera_manager.set_up_sensors(car_ego)
        actor_list.append(camera)
        for cam in cameras_288:
            actor_list.append(cam)

        # -----------------------------------------
        #  add four encoders
        # -----------------------------------------
        supervised_encoder_front = SupervisedEncoder()
        supervised_encoder_back = SupervisedEncoder()
        supervised_encoder_right = SupervisedEncoder()
        supervised_encoder_left = SupervisedEncoder()

        # -----------------------------------------
        #  add collision sensor
        # -----------------------------------------
        collision_manager = CollisionManager(world)
        collision_sensor = collision_manager.set_up_sensors(car_ego)
        actor_list.append(collision_sensor)

        # -----------------------------------------
        # set traffic manager
        # -----------------------------------------
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_hybrid_physics_mode(False)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.global_percentage_speed_difference(5)
        # traffic_manager.set_global_distance_to_leading_vehicle(5.0)

        # -----------------------------------------
        # set the number of other vehicles
        # -----------------------------------------
        num_of_vehicles = random.randint(params['min_num'],params['max_num'])
        print('number of other vehicles: {}'.format(num_of_vehicles))
        
        # -----------------------------------------
        # generate surrounding spawn points
        # -----------------------------------------
        spawn_points = world.get_map().generate_waypoints(params['density'])
        # spawn_points = world.get_map().get_spawn_points()

        # -----------------------------------------
        # process waypoint (format) to transform (format)
        # -----------------------------------------
        spawn_points_qualified = []
        for point in spawn_points:
            temp = carla.Transform(carla.Location(x=point.transform.location.x, y=point.transform.location.y, z=1.2),
                        carla.Rotation(pitch=point.transform.rotation.pitch, yaw=point.transform.rotation.yaw, roll=point.transform.rotation.roll))
            spawn_points_qualified.append(temp)
        spawn_points = spawn_points_qualified

        # -----------------------------------------
        # filter rule 1: spawn points should be within a circle range
        # -----------------------------------------
        min_dis = params['min_range']
        max_dis = params['max_range']
        spawn_points_qualified = []
        for point in spawn_points:
            loc_x = point.location.x
            loc_y = point.location.y
            if max_dis > math.sqrt((source_state[0] - loc_x) ** 2 + (source_state[1] - loc_y) ** 2) > min_dis:
                spawn_points_qualified.append(point)
            else:
                continue
        spawn_points = spawn_points_qualified

        # -----------------------------------------
        # filter rule 2: spawn points should be similar to the car direction
        # -----------------------------------------
        max_angle = params['max_angle']
        spawn_points_qualified = []
        for point in spawn_points:
            loc_yaw = point.rotation.yaw
            if abs(loc_yaw - source_state[2]) < max_angle:
                spawn_points_qualified.append(point)
            else:
                continue
        spawn_points = spawn_points_qualified

        # -----------------------------------------
        # finalize the spawn points
        # -----------------------------------------
        number_of_spawn_points = len(spawn_points)
        if num_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif num_of_vehicles >= number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, num_of_vehicles, number_of_spawn_points)
            num_of_vehicles = number_of_spawn_points - 1

        # -----------------------------------------
        # use command to apply actions on batch of actors
        # -----------------------------------------
        vehicles_id_list = []
        
        # define car make and model for other vehicles
        blueprints_vehicle = world.get_blueprint_library().filter("vehicle.*")
        blueprints_vehicle = [x for x in blueprints_vehicle if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints_vehicle = sorted(blueprints_vehicle, key=lambda bp: bp.id)

        # define running mode
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= num_of_vehicles:
                break

            # set car make, model, and color
            blueprint = random.choice(blueprints_vehicle)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            # set autopilot
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot all together
            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        # disable lane change
        vehicles = world.get_actors().filter('vehicle.*')
        for eachcar in vehicles:
            if eachcar.attributes.get('role_name') != 'car_ego':
                traffic_manager.auto_lane_change(eachcar, False)

        # execute the command
        for (i, response) in enumerate(client.apply_batch_sync(batch, synchronous_master)):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_id_list.append(response.actor_id)

        # -----------------------------------------
        # create a synchronous mode
        # -----------------------------------------
        signal_start = 0 # 0: not start, 1: start 
        index_frame = 0 # counter for the frame
        index_img = 0 # counter for the image
        with CarlaSyncMode(world, camera, *cameras_288, fps=10) as sync_mode:
            while (not agent_ego.done()) and (not signal_stop) and (not signal_collision) and (not signal_merge):
                # -----------------------------------------
                # advance the simulation and wait for the data.
                # -----------------------------------------
                for _ in range(1):  # wait a few frames
                    snapshot, image_rgb, *images = sync_mode.tick(timeout=1.0)
                    supervised_encoder_front.run_step(images[0], "Front", str(snapshot))
                    supervised_encoder_back.run_step(images[1], "Back", str(snapshot))
                    supervised_encoder_right.run_step(images[2], "Right", str(snapshot))
                    supervised_encoder_left.run_step(images[3], "Left", str(snapshot))

                # -----------------------------------------
                # run our ego car before other car starts
                # -----------------------------------------
                control = agent_ego.run_step()
                vehicles = world.get_actors().filter('vehicle.*')
                if signal_start == 0:
                    for eachcar in vehicles:
                        if eachcar.attributes.get('role_name') != 'car_ego':
                            velocity_car_selected = math.sqrt(eachcar.get_velocity().x ** 2 + eachcar.get_velocity().y ** 2)
                            if velocity_car_selected > 0.01:
                                signal_start = 1
                                break
                if signal_start == 0:
                    control.throttle = params['small_throttle']
                control.manual_gear_shift = False
                car_ego.apply_control(control)

                # -----------------------------------------
                # encode images (5th frame)
                # -----------------------------------------
                if index_img == params['index_img']:
                    save_flag_value = True
                else:
                    save_flag_value = False
                index_img += 1
                supervised_encoder_front.run_step(images[0], "Front", str(snapshot), save_flag=save_flag_value)
                supervised_encoder_back.run_step(images[1], "Back", str(snapshot), save_flag=save_flag_value)
                supervised_encoder_right.run_step(images[2], "Right", str(snapshot), save_flag=save_flag_value)
                supervised_encoder_left.run_step(images[3], "Left", str(snapshot), save_flag=save_flag_value)

                # -----------------------------------------
                # check if our ego car achieves the destination
                # -----------------------------------------
                min_dis = params['min_dis']  # minimal distance
                ego_loc = car_ego.get_location()  # current location of our ego car
                temp_dist = math.sqrt((ego_loc.x - dest_state[0]) ** 2 + (ego_loc.y - dest_state[1]) ** 2)
                if temp_dist < min_dis:
                    signal_stop = True
                # another method to check if our ego car achieves the destination
                if agent_ego.done():
                    print("our ego car achieves destination")

                # -----------------------------------------
                # check if our ego car has merged
                # -----------------------------------------
                max_dis = params['max_dis']  # maximal distance
                temp_dist = math.sqrt((ego_loc.x - source_state[0]) ** 2 + (ego_loc.y - source_state[1]) ** 2)
                if temp_dist > max_dis:
                    signal_merge = True

                # -----------------------------------------
                # check if our ego car is too close to other cars
                # -----------------------------------------
                threshold_dis = params['threshold_dis']  # minimal distance
                temp_dist_list = []
                vehicles = world.get_actors().filter('vehicle.*')
                for eachcar in vehicles:
                    if eachcar.attributes.get('role_name') != 'car_ego':
                        eachcar_loc = eachcar.get_location()
                        temp_dist_list.append(math.sqrt((ego_loc.x - eachcar_loc.x) ** 2 + (ego_loc.y - eachcar_loc.y) ** 2))
                if min(temp_dist_list) < threshold_dis:
                    signal_collision = True

                # -----------------------------------------
                # check if our ego car has a collision
                # -----------------------------------------
                if collision_manager.flag == 1:
                    signal_collision = True
                
                # -----------------------------------------
                # check if the code has error
                # -----------------------------------------
                if int(time.time()) - init_time > 120:
                    signal_error = True
                    break

            # -----------------------------------------
            # record overall information
            # -----------------------------------------
            if not signal_collision:
                fidin.write('no collision: %s\n' % str(init_time))
            else:
                fidin.write('exist collision: %s\n' % str(init_time))
            fidin.flush()
    finally:
        # -----------------------------------------
        # reset the world
        # -----------------------------------------
        world.apply_settings(origin_settings)
        
        # -----------------------------------------
        # destroy all vehicles
        # -----------------------------------------
        print('destroying our ego car')
        for actor in actor_list:
            actor.destroy()
        print('destroying other cars')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_id_list])
        
        # -----------------------------------------
        # stop recording
        # -----------------------------------------
        client.stop_recorder()
        time.sleep(0.5)


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
