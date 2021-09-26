import gym
from gym import spaces
import numpy as np
import os
import cloudpickle
import pickle
import vizdoom as vzd
import cv2
import multiprocessing as mp
from collections import OrderedDict, deque
from copy import copy
import psutil
from .action_distance import action_distance
from .checks import *

class VecEnv(gym.Env):
    def __init__(self, num_envs):
        #note: do NOT place the communication setup (creation of Pipes, Processes, etc.) inside the __init__()
        #method, otherwise, it will not be possible to serialize and save the environment
        super(VecEnv, self).__init__()
        self.num_envs = num_envs
        self.closed = False
    
    def reset(self):
        #resets all sub-environments and returns a batch (a list or tuple) of initial observations, one
        #observation per sub-environment
        pass
        
    def step(self, actions):
        #takes an action for each sub-environment and returns a batch (a list or tuple) of results, one
        #result per sub-environment. Each result is a 4-tuple packing (next_observation, reward, done, info)
        pass

    def close(self, **kwargs):
        #closes all sub-environments, communications (if they exist) and release resources        
        #note: This will be automatically called when garbage is collected or the program is exited
        pass
        
    def seed(self, seeds=None):
        #sets the seed used by each sub-environment
        #seeds: list of int (of length 'num_envs'), or int, optional. If 'seeds' is an int, then each sub-
        #environment uses 'seeds + n' as seed value, where 'n' is the index of the sub-environment (between
        #'0' and 'num_envs - 1')
        pass
    
    def render(self, mode):
        #returns one or more images (numpy arrays) to be rendered on the screen
        #if many images are returned:
        # - the number of images must be equal to 'num_envs' 
        # - they all must have the same shape
        # - they must be packed within an iterable sequence (numpy array, list, tuple, etc.)
        #mode: must be one of {'human', 'rgb_array', 'agent'}
        #'human' and 'rgb_array' are the same as for any gym environment. 'agent' is the same as for any
        #environment of this suite, that is, it denotes images of what the learner actually perceives
        pass

class VecWrapper(VecEnv):
    def __init__(self, env, num_envs=None):
        assert isinstance(env, VecEnv) or (isinstance(env, EnvMaker) and env.num_envs > 1) or (
            type(num_envs) == int and num_envs > 0)
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.num_envs = num_envs or self.env.num_envs

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @property
    def spec(self):
        return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self, **kwargs):
        return self.env.close(**kwargs)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped

#EnvMaker is a special class that can create either a basic gym.Env environment (no sub-environments) or a
#VecEnv-like environment (from sparserl)
class EnvMaker(gym.Wrapper):
    #when generating an environment use super().__init__(num_envs) or equivalent somewhere sensible (usually 
    #at the end of the __init__() method)
    def __init__(self):
        env = self._make()
        self.is_vec = isinstance(env, VecEnv)
        if self.is_vec:
            self.num_envs = env.num_envs
        super().__init__(env)
    
    def _make(self):
        #must return an Env object
        raise NotImplementedError
    
    def get_config(self):
        #(optional)
        #produces a human-readable dict of the settings used for instantiating an object of this class
        #(including defaults). Useful only for logging purposes
        pass
    
    def _check_inputs(self):
        #(optional)
        #verifies arguments
        #performs minor processing of arguments (e.g. recasting, repeating), if needed
        pass
        
class NoRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return 0.0
        
class WarpFrameEnv(gym.Wrapper):
    def __init__(self, env, dims):
        gym.Wrapper.__init__(self, env)
        self._dims = tuple(dims)
        last_dim = env.observation_space.shape[-1]
        self.observation_space = spaces.Box(low=0, high=255, shape=(dims[1], dims[0], last_dim), 
                                            dtype=env.observation_space.dtype)
        self._warp_fn = self._observation_rgb if last_dim == 3 else self._observation_gray

    def _observation_rgb(self, frame):
        frame = cv2.resize(frame, self._dims, interpolation=cv2.INTER_AREA)
        return frame
    
    def _observation_gray(self, frame):
        frame = cv2.resize(frame, self._dims, interpolation=cv2.INTER_AREA)
        return frame[:, :, None]
    
    def render(self, mode='human', **kwargs):
        if mode == 'agent':
            frame = self.env.render(mode, **kwargs)
            if frame is None:
                frame = self.env.render('rgb_array', **kwargs)
            return self._warp_fn(frame)
        return self.env.render(mode, **kwargs)
    
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self._warp_fn(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._warp_fn(observation), reward, done, info
        
class AddBackgroundEnv(gym.Wrapper):
    def __init__(self, env, dims):
        gym.Wrapper.__init__(self, env)
        self._dims = tuple(dims)
        height, width, channels = env.observation_space.shape
        self._coverage = (height / dims[0] if height < dims[0] else 1.0, width / dims[1] if width < dims[1] else 1.0)
        self._game_slice = (slice(0, height), slice(0, width), slice(0, channels))
        self.observation_space = spaces.Box(low=0, high=255, shape=(dims[1], dims[0], channels), 
                                            dtype=env.observation_space.dtype)

    def _observation(self, frame):
        self._background[self._game_slice] = frame
        return self._background.copy()
    
    def reset(self, **kwargs):
        self._background = np.zeros(self.observation_space.shape, dtype=np.uint8)
        frame = self.env.reset(**kwargs)
        return self._observation(frame)

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        return self._observation(frame), reward, done, info
        
    def render(self, mode='human', **kwargs):
        if mode == 'agent':
            frame = self.env.render(mode, **kwargs)
            if frame is None:
                frame = self.env.render('rgb_array', **kwargs)
            return self._observation(frame)
        else:
            frame = self.env.render('rgb_array', **kwargs)
            height, width = frame.shape[0:2]
            background = np.zeros((int(height / self._coverage[0]), int(width / self._coverage[1]), 3), dtype=np.uint8)
            render_slice = (slice(0, height), slice(0, width), slice(0, 3))
            background[render_slice] = frame
            if mode == 'rgb_array':
                return background
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self._viewer is None:
                    self._viewer = rendering.SimpleImageViewer()
                self._viewer.imshow(background)
                return self._viewer.isopen
        
def stacked_space(space, k):
    new_dtype = space.dtype
    if type(space) == spaces.Tuple:
        return spaces.Tuple(tuple([stacked_space(sp, k) for sp in space.spaces]))
    elif type(space) == spaces.Dict:
        return spaces.Dict(OrderedDict({key: stacked_space(sp, k) for key, sp in space.spaces.items()}))
    else:
        if isinstance(space, spaces.Discrete):
            low = np.zeros(k, dtype=new_dtype)
            high = np.tile(space.n, k).astype(new_dtype)
        elif isinstance(space, spaces.MultiDiscrete):
            low = np.zeros(np.size(space.nvec) * k, dtype=new_dtype)
            high = np.tile(space.nvec, k).astype(new_dtype)
        elif isinstance(space, spaces.MultiBinary):
            low = np.zeros(space.shape[0] * k, dtype=new_dtype)
            high = np.ones(space.shape[0] * k, dtype=new_dtype)
        elif isinstance(space, spaces.Box):
            low = np.tile(space.low, k).astype(new_dtype)
            high = np.tile(space.high, k).astype(new_dtype)
        return spaces.Box(low=low, high=high, dtype=new_dtype)

#to be used only when observations are numpy arrays or scalars
class FrameStackEnv(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        self.observation_space = stacked_space(env.observation_space, k)

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self._k):
            self._frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self._frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        ###is converting deque to list necessary?
        return LazyFrames(list(self._frames))

class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]
        
def worker(remote, parent_remote, serialized_env):
    ###don't forget to reseed inside the worker to get different envs (tested for python and numpy so far)
    ###set the seed of the environment inside the worker as well to guarantee reproducibility
    parent_remote.close()
    env = cloudpickle.loads(serialized_env)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            action, kwargs = data
            ob, reward, done, info = env.step(action)
            if done:
                ob = env.reset(**kwargs)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            remote.send(env.reset(**data))
        elif cmd == 'render':
            mode, kwargs = data
            remote.send(env.render(mode, **kwargs))
        elif cmd == 'seed':
            np.random.seed(data)
            remote.send(env.seed(data))
        elif cmd == 'close':
            remote.send(env.close(**data))
        elif cmd == 'disconnect': #separating close and disconnect is not necessary, but reads better
            remote.close()
            break
        else:
            raise NotImplementedError
    
class VectorizeEnv(VecWrapper):
    def __init__(self, env, num_envs):
        self.num_envs = num_envs
        super().__init__(env, self.num_envs)
        self.closed = True
        
    def _connect(self):
        self._remotes, self._work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self._ps = []
        for work_remote, remote in zip(self._work_remotes, self._remotes):
            self.env.seed(np.random.randint(2**32))
            self._ps.append(mp.Process(target=worker, args=(work_remote, remote, cloudpickle.dumps(
                self.env))))
        for p in self._ps:
            p.daemon = True
            p.start()
        for remote in self._work_remotes:
            remote.close()
        self.closed = False
    
    def reset(self, **kwargs):
        self._last_kwargs = kwargs
        if self.closed:
            self._connect() #connect() must set 'closed' to False before returning
        for remote in self._remotes:
            remote.send(('reset', kwargs))
        return [remote.recv() for remote in self._remotes]
        
    def step(self, actions):
        for remote, action in zip(self._remotes, actions):
            remote.send(('step', (action, self._last_kwargs)))
        return [remote.recv() for remote in self._remotes]

    def render(self, mode='human', **kwargs):
        if mode == 'human':
            self._remotes[0].send(('render', (mode, kwargs)))
            return self._remotes[0].recv()
        else:
            for pipe in self._remotes:
                pipe.send(('render', (mode, kwargs)))
            imgs = [pipe.recv() for pipe in self._remotes]
            return imgs
            
    def seed(self, seeds):
        if self.closed:
            self._connect()
        if type(seeds) in {tuple, list}:
            _seeds = seeds
        elif seeds is None:
            _seeds = [None] * self.num_envs
        else:
            _seeds = [seeds + n  for n in range(self.num_envs)]
        for seed, remote in zip(_seeds, self._remotes):
            remote.send(('seed', seed))
        return [remote.recv() for remote in self._remotes]
            
    def close(self, **kwargs):
        if self.closed:
            return
        for remote in self._remotes:
            remote.send(('close', kwargs))
        ack = [remote.recv() for remote in self._remotes]
        for remote in self._remotes:
            remote.send(('disconnect', None))
        for p in self._ps:
            p.join()
        self.closed = True
        return ack

#to be used only when input observations are numpy arrays or scalars
class FrameStackVecEnv(VecWrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self._k = k
        self._frames = [deque([], maxlen=k)] * self.num_envs
        self.observation_space = stacked_space(env.observation_space, k)
        
    def reset(self, **kwargs):
        obses = self.env.reset(**kwargs)
        for i, (obs, frame) in enumerate(zip(obses, self._frames)):
            for _ in range(self._k):
                frame.append(obs)
            obses[i] = LazyFrames(list(frame))
        return obses

    def step(self, actions):
        results = self.env.step(actions)
        for result, frame in zip(results, self._frames):
            if frame:
                frame.append(result[0])
            else:
                for _ in range(self._k):
                    frame.append(result[0])
            result[0] = LazyFrames(list(frame))
            if result[2]:
                frame.clear()
        return results
    
def get_digit_offsets(station, num_digits, random_sz, mods):
    offset = []
    for canonical_digit in range(num_digits):
        count = 0
        dec_count = 0
        for i, station_digit in enumerate(station):
            count += int(station_digit > canonical_digit) * mods[i]
            count += station_digit * i * mods[max(i - 1, 0)]
            count += int(station_digit == canonical_digit) * dec_count
            count = count % random_sz
            dec_count += station_digit * mods[i]
        offset.append(count)
    return offset
    
def incdec_station(number, delta, goup, num_digits):
    new_number = []
    l = len(number)
    if goup:
        new_number.extend(number[:delta])
        for i, digit in enumerate(number[delta:]):
            new_digit = digit + 1
            if new_digit < num_digits:
                new_number.append(new_digit)
                next_idx = delta + i + 1
                if next_idx < l:
                    new_number.extend(number[next_idx:])
                break
            else:
                new_number.append(0)
    else:
        new_number.extend(number[:delta])
        for i, digit in enumerate(number[delta:]):
            new_digit = digit - 1
            if new_digit > -1:
                new_number.append(new_digit)
                next_idx = delta + i + 1
                if next_idx < l:
                    new_number.extend(number[next_idx:])
                break
            else:
                new_number.append(num_digits - 1)
    return new_number
    
def incdec_stride(number, goup, num_digits, max_number):
    new_number = []
    l = len(number)
    if goup:
        if all([d1 == d2 for (d1, d2) in zip(number, max_number)]):
            new_number = [0] * l
        else:
            for i, digit in enumerate(number):
                new_digit = digit + 1
                if new_digit < num_digits:
                    new_number.append(new_digit)
                    next_idx = i + 1
                    if next_idx < l:
                        new_number.extend(number[next_idx:])
                    break
                else:
                    new_number.append(0)
    else:
        if sum(number) == 0:
            new_number.extend(max_number)
        else:
            for i, digit in enumerate(number):
                new_digit = digit - 1
                if new_digit > -1:
                    new_number.append(new_digit)
                    next_idx = i + 1
                    if next_idx < l:
                        new_number.extend(number[next_idx:])
                    break
                else:
                    new_number.append(num_digits - 1)
    return new_number

def station2str(station, stride, fill):
    emb = ''
    for s in station:
        semb = str(s)
        emb += '0' * (fill - len(semb)) + semb
    emb += str(stride)
    return emb

class TVEnv(gym.Wrapper):
    #coverage & partitions are to be defined in terms of width - height, in that order
    def __init__(self, env, mode, coverage=(1.0, 1.0), partitions=(8, 8), num_colors=16, allow_tv_off=True, 
                 color_format='rgb', rendering=None, rnd_batch=10000):
        gym.Wrapper.__init__(self, env)
        assert all([0.0 < c <= 1.0 for c in coverage])
        assert len(env.observation_space.shape) == 3
        height, width, num_channels = env.observation_space.shape
        assert height > 0 and width > 0
        assert num_channels in {1, 3}
        self._viewer = None
        folder_path = os.path.abspath(os.path.dirname(__file__)) + "/assets"
        self._colormap = None
        if color_format == 'original':
            self._colormap = pickle.load(open(folder_path + '/originalmap', 'rb'))
        div0, div1 = self._partitions = list(partitions)[::-1]
        self._num_colors = num_colors
        num_tv_actions = {'broken': 0, 'noisy': 1, 'sasr': 1, 'madr': 4}[mode] + int(allow_tv_off)
        self._extra_action = False
        self._env_actions = env.action_space.n
        if num_tv_actions > 0:
            self.action_space = spaces.Discrete(env.action_space.n + num_tv_actions)
            self._extra_action = True
        if mode == 'broken' and not allow_tv_off:
            self._operation = 'broken_on'
        elif mode == 'broken' and allow_tv_off:
            self._operation = 'broken_off'
        elif mode == 'noisy' and not allow_tv_off:
            self._operation = 'noisy_on'
        elif mode == 'noisy' and allow_tv_off:
            self._operation = 'noisy_off'
        elif mode == 'sasr' and not allow_tv_off:
            self._operation = 'sasr_on'
        elif mode == 'sasr' and allow_tv_off:
            self._operation = 'sasr_off'
        elif mode == 'madr' and not allow_tv_off:
            self._operation = 'madr_on'
        elif mode == 'madr' and allow_tv_off:
            self._operation = 'madr_off'
        self._coverage = coverage
        hmid, wmid = int(height * (1.0 - coverage[1])), int(width * (1.0 - coverage[0]))
        self._tv_slice = (slice(hmid, height), slice(wmid, width), slice(0, num_channels))
        tv_height, tv_width = height - hmid, width - wmid
        self._tv_shape = (tv_height, tv_width, num_channels)
        r_spacing, c_spacing = tv_height // div0, tv_width // div1
        self._row_sizes = np.array([r_spacing] * (div0 - 1) + [tv_height - r_spacing * (div0 - 1)])
        self._col_sizes = np.array([c_spacing] * (div1 - 1) + [tv_width - c_spacing * (div1 - 1)])
        if num_channels == 1:
            borders = np.linspace(0, 255, num_colors + 1)
            self._palette = ((borders[0:-1] + borders[1:]) / 2).astype(np.uint8)[:, None]
        else:
            r = int(num_colors ** (1 / 3)) + 1
            options = ((r, r, r - 2), (r, r - 1, r - 1), (r, r, r - 1), (r, r, r))
            closest = options[[o[0] * o[1] * o[2] >= num_colors for o in options].index(True)]
            palette_channels = []
            for num in closest:
                borders = np.linspace(0, 255, num + 1)
                palette_channels.append(((borders[0:-1] + borders[1:]) / 2).astype(np.uint8))
            self._palette = np.array(np.meshgrid(*palette_channels)).T.reshape(-1, 3)
        self._num_partitions = div0 * div1
        all_slices = []
        rid = 0
        for rsz in self._row_sizes:
            cid = 0
            new_rid = rid + rsz
            for csz in self._col_sizes:
                new_cid = cid + csz
                all_slices.append((slice(rid, new_rid), slice(cid, new_cid), slice(0, num_channels)))
                cid = new_cid
            rid = new_rid
        if mode in {'broken', 'noisy'}:
            self._tv_buffer = []
            self._rnd_batch = rnd_batch
            self._station_slices = all_slices[:]
        elif mode == 'sasr':
            self._start_station = [0] * self._num_partitions
            self._station_slices = all_slices[:]
            if rendering is None or rendering == 'static':
                self._rendering = 'static'
            else:
                self._rendering = 'dynamic'
                self._stride_base = 5
                self._random_sz = 1000000
                assert self._random_sz > 1 and self._random_sz > num_colors
                self._mods = [1, num_colors]
                for i in range(2, self._num_partitions):
                    self._mods.append((self._mods[1] * self._mods[-1]) % self._random_sz)
                #avoid modifying numpy's global seed
                prng_state = np.random.get_state()
                np.random.seed(0)
                self._color_permutations = np.random.randint(0, num_colors, (num_colors, self._random_sz), dtype=np.uint8)
                np.random.set_state(prng_state)
        else:
            self._stride_base = 5
            stride_cells = 0
            while self._stride_base ** stride_cells < self._num_partitions - stride_cells:
                stride_cells += 1
            self._num_partitions -= stride_cells
            self._max_stride = []
            r = self._num_partitions - 1
            for i in range(stride_cells - 1, 0, -1):
                d, r = divmod(r, self._stride_base ** i)
                self._max_stride.insert(0, d)
            self._max_stride.insert(0, r)
            self._start_station = [0] * self._num_partitions
            self._start_stride = [0] * stride_cells
            borders = np.linspace(0, 255, self._stride_base + 1)
            self._stride_palette = ((borders[0:-1] + borders[1:]) / 2).astype(np.uint8)[:, None]
            if num_channels == 3:
                self._stride_palette = np.tile(self._stride_palette, 3)
            self._station_slices = all_slices[:-stride_cells]
            self._stride_slices = all_slices[-stride_cells:]
            if rendering == 'static':
                self._rendering = 'static'
            else:
                self._rendering = 'dynamic'
                self._random_sz = 1000000
                assert self._random_sz > 1 and self._random_sz > num_colors
                self._mods = [1, num_colors]
                for i in range(2, self._num_partitions):
                    self._mods.append((self._mods[1] * self._mods[-1]) % self._random_sz)
                #avoid modifying numpy's global seed
                prng_state = np.random.get_state()
                np.random.seed(0)
                self._color_permutations = np.random.randint(0, num_colors, (num_colors, self._random_sz), dtype=np.uint8)
                np.random.set_state(prng_state)
        
    def reset(self):
        obs = self.env.reset()
        if self._operation in {'broken_off', 'noisy_off', 'sasr_off', 'madr_off'}:
            self._tv_state = True
            self._tvon = 0
        if self._operation in {'sasr_on', 'sasr_off'}:
            self._totdist = 0
            self._maxdist = -1
            self._record = dict()
            self._station = self._start_station
            self._last_tv = np.zeros(self._tv_shape, dtype=np.uint8)
            if self._rendering == 'static':
                for slc, digit in zip(self._station_slices, self._station):
                    self._last_tv[slc] = self._palette[digit]
            else:
                digit_offsets = get_digit_offsets(self._station, self._num_colors, self._random_sz, self._mods)
                for slc, digit in zip(self._station_slices, self._station):
                    self._last_tv[slc] = self._palette[self._color_permutations[digit][digit_offsets[digit]]]
                    digit_offsets[digit] = (digit_offsets[digit] + 1) % self._random_sz
            obs[self._tv_slice] = self._last_tv
        elif self._operation in {'madr_on', 'madr_off'}:
            self._totdist = 0
            self._maxdist = -1
            self._record = dict()
            self._station = self._start_station
            self._stride = self._start_stride
            self._last_tv = np.zeros(self._tv_shape, dtype=np.uint8)
            if self._rendering == 'static':
                for slc, digit in zip(self._station_slices, self._station):
                    self._last_tv[slc] = self._palette[digit]
            else:
                digit_offsets = get_digit_offsets(self._station, self._num_colors, self._random_sz, self._mods)
                for slc, digit in zip(self._station_slices, self._station):
                    self._last_tv[slc] = self._palette[self._color_permutations[digit][digit_offsets[digit]]]
                    digit_offsets[digit] = (digit_offsets[digit] + 1) % self._random_sz
            for slc, digit in zip(self._stride_slices, self._stride):
                self._last_tv[slc] = self._stride_palette[digit]
            obs[self._tv_slice] = self._last_tv
        else:
            if not self._tv_buffer:
                self._tv_buffer = list(np.random.randint(0, self._num_colors, (
                    self._rnd_batch, self._num_partitions)))
            obs[self._tv_slice] = self._last_tv = self._palette[np.repeat(np.repeat(np.reshape(
                self._tv_buffer.pop(), self._partitions), self._row_sizes, axis=-2), self._col_sizes,\
                axis=-1)]
        return obs
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self._operation == 'broken_on':
            if not self._tv_buffer:
                self._tv_buffer = list(np.random.randint(0, self._num_colors, (
                    self._rnd_batch, self._num_partitions)))
            for slc, color in zip(self._station_slices, self._tv_buffer.pop()):
                self._last_tv[slc] = self._palette[color]
            obs[self._tv_slice] = self._last_tv
        elif self._operation == 'broken_off':
            if action >= self._env_actions:
                self._tv_state = not self._tv_state
            info['step_tvon'] = self._tv_state
            if self._tv_state:
                if not self._tv_buffer:
                    self._tv_buffer = list(np.random.randint(0, self._num_colors, (
                        self._rnd_batch, self._num_partitions)))
                for slc, color in zip(self._station_slices, self._tv_buffer.pop()):
                    self._last_tv[slc] = self._palette[color]
                obs[self._tv_slice] = self._last_tv
                self._tvon += 1
            if done:
                info['episode']['tvon'] = self._tvon
        elif self._operation == 'noisy_on':
            if action >= self._env_actions:
                if not self._tv_buffer:
                    self._tv_buffer = list(np.random.randint(0, self._num_colors, (
                        self._rnd_batch, self._num_partitions)))
                for slc, color in zip(self._station_slices, self._tv_buffer.pop()):
                    self._last_tv[slc] = self._palette[color]
            obs[self._tv_slice] = self._last_tv
        elif self._operation == 'noisy_off':
            tv_action = 0 if action < self._env_actions else (action - self._env_actions + 1)
            if tv_action == 1:
                self._tv_state = not self._tv_state
            info['step_tvon'] = self._tv_state
            if self._tv_state:
                if tv_action in {0, 1}:
                    obs[self._tv_slice] = self._last_tv
                else:
                    if not self._tv_buffer:
                        self._tv_buffer = list(np.random.randint(0, self._num_colors, (
                            self._rnd_batch, self._num_partitions)))
                    for slc, color in zip(self._station_slices, self._tv_buffer.pop()):
                        self._last_tv[slc] = self._palette[color]
                    obs[self._tv_slice] = self._last_tv
                self._tvon += 1
            if done:
                info['episode']['tvon'] = self._tvon
        elif self._operation == 'sasr_on':
            tv_action = 0 if action < self._env_actions else (action - self._env_actions + 1)
            if tv_action == 0:
                obs[self._tv_slice] = self._last_tv
            else:
                self._station = incdec_station(self._station, 0, True, self._num_colors)
                if self._rendering == 'static':
                    for slc, digit in zip(self._station_slices, self._station):
                        self._last_tv[slc] = self._palette[digit]
                else:
                    digit_offsets = get_digit_offsets(self._station, self._num_colors, self._random_sz, self._mods)
                    for slc, digit in zip(self._station_slices, self._station):
                        self._last_tv[slc] = self._palette[self._color_permutations[digit][digit_offsets[digit]]]
                        digit_offsets[digit] = (digit_offsets[digit] + 1) % self._random_sz
                obs[self._tv_slice] = self._last_tv
            dist = sum([stn * self._num_colors ** i for i, stn in enumerate(self._station)])
            self._totdist += dist
            if dist > self._maxdist:
                self._maxdist = dist
            if dist in self._record.keys():
                self._record[dist] += 1
            else:
                self._record[dist] = 1
            if done:
                info['episode']['maxdist'] = self._maxdist
                info['episode']['totdist'] = self._totdist
                info['episode']['record'] = self._record
        elif self._operation == 'sasr_off':
            tv_action = 0 if action < self._env_actions else (action - self._env_actions + 1)
            if tv_action == 1:
                self._tv_state = not self._tv_state
            info['step_tvon'] = self._tv_state
            if self._tv_state:
                if tv_action in {0, 1}:
                    obs[self._tv_slice] = self._last_tv
                else:
                    self._station = incdec_station(self._station, 0, True, self._num_colors)
                    if self._rendering == 'static':
                        for slc, digit in zip(self._station_slices, self._station):
                            self._last_tv[slc] = self._palette[digit]
                    else:
                        digit_offsets = get_digit_offsets(self._station, self._num_colors, self._random_sz, self._mods)
                        for slc, digit in zip(self._station_slices, self._station):
                            self._last_tv[slc] = self._palette[self._color_permutations[digit][digit_offsets[digit]]]
                            digit_offsets[digit] = (digit_offsets[digit] + 1) % self._random_sz
                    obs[self._tv_slice] = self._last_tv
                self._tvon += 1
                dist = sum([stn * self._num_colors ** i for i, stn in enumerate(self._station)])
                self._totdist += dist
                if dist > self._maxdist:
                    self._maxdist = dist
                if dist in self._record.keys():
                    self._record[dist] += 1
                else:
                    self._record[dist] = 1
            if done:
                info['episode']['tvon'] = self._tvon
                info['episode']['maxdist'] = self._maxdist
                info['episode']['totdist'] = self._totdist
                info['episode']['record'] = self._record
        elif self._operation == 'madr_on':
            tv_action = 0 if action < self._env_actions else (action - self._env_actions + 1)
            if tv_action == 0:
                obs[self._tv_slice] = self._last_tv
                dec_stride = sum([stride * self._stride_base ** i for i, stride in enumerate(self._stride)])
            elif tv_action in {1, 2}:
                self._stride = incdec_stride(self._stride, tv_action==1, self._stride_base, self._max_stride)
                for slc, digit in zip(self._stride_slices, self._stride):
                    self._last_tv[slc] = self._stride_palette[digit]
                dec_stride = sum([stride * self._stride_base ** i for i, stride in enumerate(self._stride)])
                self._station = incdec_station(self._station, (dec_stride + tv_action - 1) % self._num_partitions, tv_action==1, self._num_colors)
                if self._rendering == 'static':
                    for slc, digit in zip(self._station_slices, self._station):
                        self._last_tv[slc] = self._palette[digit]
                else:
                    digit_offsets = get_digit_offsets(self._station, self._num_colors, self._random_sz, self._mods)
                    for slc, digit in zip(self._station_slices, self._station):
                        self._last_tv[slc] = self._palette[self._color_permutations[digit][digit_offsets[digit]]]
                        digit_offsets[digit] = (digit_offsets[digit] + 1) % self._random_sz
                obs[self._tv_slice] = self._last_tv
            elif tv_action in {3, 4}:
                dec_stride = sum([stride * self._stride_base ** i for i, stride in enumerate(self._stride)])
                self._station = incdec_station(self._station, dec_stride, tv_action==3, self._num_colors)
                if self._rendering == 'static':
                    for slc, digit in zip(self._station_slices, self._station):
                        self._last_tv[slc] = self._palette[digit]
                else:
                    digit_offsets = get_digit_offsets(self._station, self._num_colors, self._random_sz, self._mods)
                    for slc, digit in zip(self._station_slices, self._station):
                        self._last_tv[slc] = self._palette[self._color_permutations[digit][digit_offsets[digit]]]
                        digit_offsets[digit] = (digit_offsets[digit] + 1) % self._random_sz
                obs[self._tv_slice] = self._last_tv
            dist = action_distance(self._station, dec_stride, self._num_colors)
            self._totdist += dist
            if dist > self._maxdist:
                self._maxdist = dist
            station_emb = station2str(self._station, dec_stride, 2)
            if station_emb in self._record.keys():
                self._record[station_emb] += 1
            else:
                self._record[station_emb] = 1
            if done:
                info['episode']['maxdist'] = self._maxdist
                info['episode']['totdist'] = self._totdist
                info['episode']['record'] = self._record
        elif self._operation == 'madr_off':
            tv_action = 0 if action < self._env_actions else (action - self._env_actions + 1)
            if tv_action == 1:
                self._tv_state = not self._tv_state
            info['step_tvon'] = self._tv_state
            if self._tv_state:
                if tv_action in {0, 1}:
                    obs[self._tv_slice] = self._last_tv
                    dec_stride = sum([stride * self._stride_base ** i for i, stride in enumerate(self._stride)])
                elif tv_action in {2, 3}:
                    self._stride = incdec_stride(self._stride, tv_action==2, self._stride_base, self._max_stride)
                    for slc, digit in zip(self._stride_slices, self._stride):
                        self._last_tv[slc] = self._stride_palette[digit]
                    dec_stride = sum([stride * self._stride_base ** i for i, stride in enumerate(self._stride)])
                    self._station = incdec_station(self._station, (dec_stride + tv_action - 2) % self._num_partitions, tv_action==2, self._num_colors)
                    if self._rendering == 'static':
                        for slc, digit in zip(self._station_slices, self._station):
                            self._last_tv[slc] = self._palette[digit]
                    else:
                        digit_offsets = get_digit_offsets(self._station, self._num_colors, self._random_sz, self._mods)
                        for slc, digit in zip(self._station_slices, self._station):
                            self._last_tv[slc] = self._palette[self._color_permutations[digit][digit_offsets[digit]]]
                            digit_offsets[digit] = (digit_offsets[digit] + 1) % self._random_sz
                    obs[self._tv_slice] = self._last_tv
                elif tv_action in {4, 5}:
                    dec_stride = sum([stride * self._stride_base ** i for i, stride in enumerate(self._stride)])
                    self._station = incdec_station(self._station, dec_stride, tv_action==4, self._num_colors)
                    if self._rendering == 'static':
                        for slc, digit in zip(self._station_slices, self._station):
                            self._last_tv[slc] = self._palette[digit]
                    else:
                        digit_offsets = get_digit_offsets(self._station, self._num_colors, self._random_sz, self._mods)
                        for slc, digit in zip(self._station_slices, self._station):
                            self._last_tv[slc] = self._palette[self._color_permutations[digit][digit_offsets[digit]]]
                            digit_offsets[digit] = (digit_offsets[digit] + 1) % self._random_sz
                    obs[self._tv_slice] = self._last_tv
                self._tvon += 1
                dist = action_distance(self._station, dec_stride, self._num_colors)
                self._totdist += dist
                if dist > self._maxdist:
                    self._maxdist = dist
                station_emb = station2str(self._station, dec_stride, 2)
                if station_emb in self._record.keys():
                    self._record[station_emb] += 1
                else:
                    self._record[station_emb] = 1
            if done:
                info['episode']['tvon'] = self._tvon
                info['episode']['maxdist'] = self._maxdist
                info['episode']['totdist'] = self._totdist
                info['episode']['record'] = self._record
        return obs, rew, done, info
    
    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        return self.env.close()
    
    def render(self, mode='human', **kwargs):
        if mode == 'agent':
            frame = self.env.render(mode, **kwargs)
            if self._operation in {'broken_off', 'noisy_off', 'sasr_off', 'madr_off'}:
                if self._tv_state:
                    frame[self._tv_slice] = self._last_tv
            else:
                frame[self._tv_slice] = self._last_tv
            return frame
        else:
            frame = self.env.render('rgb_array', **kwargs)
            height, width = frame.shape[0:2]
            hmid, wmid = int(height * (1.0 - self._coverage[1])), int(width * (1.0 - self._coverage[0]))
            tv_hsz, tv_wsz = height - hmid, width - wmid
            tv_height, tv_width, tv_channels = self._last_tv.shape
            render_slice = (slice(hmid, height), slice(wmid, width), slice(0, 3))
            tv_frame = self._last_tv
            if tv_height != tv_hsz or tv_width != tv_wsz:
                tv_frame = cv2.resize(tv_frame, (tv_wsz, tv_hsz), interpolation=cv2.INTER_AREA)
                if tv_frame.ndim < 3:
                    tv_frame = tv_frame[:, :, None]
            if tv_channels == 1:
                if self._colormap is not None:
                    tv_frame = cv2.LUT(np.tile(tv_frame, 3), self._colormap[None])
                else:
                    tv_frame = np.tile(tv_frame, 3)                
            if self._operation in {'broken_off', 'noisy_off', 'sasr_off', 'madr_off'}:
                if self._tv_state:
                    frame[render_slice] = tv_frame
            else:
                frame[render_slice] = tv_frame
            if mode == 'rgb_array':
                return frame
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self._viewer is None:
                    self._viewer = rendering.SimpleImageViewer()
                self._viewer.imshow(frame)
                return self._viewer.isopen
    
class DoomMwhInfoEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._visited_sectors = set()
        
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        xpos, ypos = tuple(info['game_variables'])
        for key, val in vertices.items():
            xlow, xhigh = val['X']
            ylow, yhigh = val['Y']
            if xlow <= xpos <= xhigh and ylow <= ypos <= yhigh:                
                self._visited_sectors.add(key)
                break
        if done:
            if 'episode' not in info.keys():
                info['episode'] = {}
            info['episode'].update(visited_sectors=copy(self._visited_sectors))
            self._visited_sectors.clear()
        return obs, rew, done, info

vertices = {0: {'X': (432, 576), 'Y': (-704, -640)}, 
            1: {'X': (432, 496), 'Y':  (-640, -544)}, 
            2: {'X': (384, 544), 'Y':  (-544, -384)},  
            3: {'X': (432, 496), 'Y':  (-384, -256)}, 
            4: {'X': (384, 544), 'Y':  (-256, -96)},  
            5: {'X': (320, 384), 'Y':  (-208, -144)},
            6: {'X': (160, 320), 'Y':  (-256, -96)},
            7: {'X': (544, 608), 'Y':  (-208, -144)},  
            8: {'X': (608, 768), 'Y':  (-256, -96)},  
            9: {'X': (656, 720), 'Y':  (-96, -32)}, 
            10: {'X': (432, 496), 'Y':  (-96, -32)},
            11: {'X': (384, 544), 'Y':  (-32, 128)},
            12: {'X': (544, 608), 'Y':  (16, 80)}, 
            13: {'X': (608, 768), 'Y':  (-32, 128)}, 
            14: {'X': (768, 960), 'Y':  (16, 80)}, 
            15: {'X': (960, 1120), 'Y':  (-32, 128)}, 
            16: {'X': (1008, 1072), 'Y':  (-96, -32)}, 
            17: {'X': (960, 1120), 'Y':  (-256, -96)}, 
            18: {'X': (992, 1088), 'Y':  (-416, -256)}}

class DoomMyWayHomeEnv(gym.Env):
    def __init__(self, frame_skip, skip_pool, screen_dims, timeout, screen_format, enable_depth):
        assert frame_skip >= skip_pool
        width, height = screen_dims
        ###to be updated for deployment
        folder_path = os.path.abspath(os.path.dirname(__file__)) + "/assets"
        self._config_path = folder_path + "/my_way_home.cfg"
        self._resolution = getattr(vzd.ScreenResolution, 'RES_' + str(width) + 'X' + str(height))
        self._format = getattr(vzd.ScreenFormat, screen_format)
        self._colormap = None
        if screen_format == 'GRAY8':
            self._colormap = pickle.load(open(folder_path + '/graymap', 'rb'))
        elif screen_format == 'DOOM_256_COLORS8':
            self._colormap = pickle.load(open(folder_path + '/originalmap', 'rb'))
        self._max_episode_steps = timeout
        self._enable_depth = enable_depth
        channels = {'RGB24': 3, 'RGBA32': 4, 'GRAY8': 1, 'DOOM_256_COLORS8': 1}[screen_format] + int(
            enable_depth)
        self._add_dim = channels <= 2
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, channels), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self._frame_skip = frame_skip
        self._skip_rng = list(range(1, skip_pool))
        self._maxpool = skip_pool > 1
        self._skip_tics = frame_skip - skip_pool + 1
        if self._maxpool:
            self._obs_buffer = np.zeros((skip_pool,) + self.observation_space.shape, dtype=np.float64)
        self._dummy_obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self._dummy_vars = np.zeros(3)
        self.game = None
        self._viewer = None
        self._seed = np.random.randint(2**32)
        
    def reset(self):
        if self.game is None:
            self.game = vzd.DoomGame()
            self.game.load_config(self._config_path)
            self.game.set_screen_resolution(self._resolution)
            self.game.set_screen_format(self._format)
            self.game.set_depth_buffer_enabled(self._enable_depth)
            self.game.set_seed(self._seed)
            self.game.set_episode_timeout(self._max_episode_steps)
            self.game.set_window_visible(False)
            self.game.init()
        self._step_count = 0
        self._visited_rooms = set([0])
        self.game.new_episode()
        state = self.game.get_state()
        obs = state.screen_buffer
        if self._add_dim:
            obs = obs[:, :, None]
        if self._enable_depth:
            depth = state.depth_buffer[:, :, None]
            obs = np.concatenate((obs, depth), axis=-1)
        return obs
    
    def step(self, action):
        game_action = [0] * 3
        if action < 3:
            game_action[action] = 1
        room = -1
        if self._maxpool:
            reward = self.game.make_action(game_action, self._skip_tics)
            done = self.game.is_episode_finished()
            if not done:
                new_state = self.game.get_state()
                obs = new_state.screen_buffer
                if self._add_dim:
                    obs = obs[:, :, None]
                if self._enable_depth:
                    depth = new_state.depth_buffer[:, :, None]
                    obs = np.concatenate((obs, depth), axis=-1)
                self._obs_buffer[0] = obs
                for i in self._skip_rng:
                    reward += self.game.make_action(act, 1)
                    done = self.game.is_episode_finished()
                    if done:
                        break
                    new_state = self.game.get_state()
                    obs = new_state.screen_buffer
                    if self._add_dim:
                        obs = obs[:, :, None]
                    if self._enable_depth:
                        depth = new_state.depth_buffer[:, :, None]
                        obs = np.concatenate((obs, depth), axis=-1)
                    self._obs_buffer[i] = obs
            obs = self._obs_buffer.max(axis=0)
        else:
            reward = self.game.make_action(game_action, self._frame_skip)
            done = self.game.is_episode_finished()
            try:
                new_state = self.game.get_state()
                obs = new_state.screen_buffer
                if self._add_dim:
                    obs = obs[:, :, None]
                if self._enable_depth:
                    depth = new_state.depth_buffer[:, :, None]
                    obs = np.concatenate((obs, depth), axis=-1)
                game_vars = new_state.game_variables
                for key, val in vertices.items():
                    xlow, xhigh = val['X']
                    ylow, yhigh = val['Y']
                    if xlow <= game_vars[0] <= xhigh and ylow <= game_vars[1] <= yhigh:                
                        room = key
                        break
                if room >= 0:
                    self._visited_rooms.add(room)
            except:
                obs = self._dummy_obs
        self._step_count += 1
        if done:
            vecrooms = np.zeros(len(vertices), dtype=np.int32)
            vecrooms[list(self._visited_rooms)] = 1
            return obs, reward, done, {'episode': {'l': self._step_count, 'r': reward, 'visited_rooms': vecrooms}}
        else:
            return obs, reward, done, {}
    
    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        return
            
    def seed(self, this_seed=None):
        if this_seed is not None:
            assert isinstance(this_seed, int)
            if self.game is not None:
                self.game.set_seed(this_seed)
            self._seed = this_seed
            return [this_seed]
        else:
            return [self._seed]
    
    def render(self, mode='human'):
        if self.game is not None:
            if mode == 'agent' and self._maxpool:
                raise NotImplemented
            else:
                if self.game.is_episode_finished():
                    if self._enable_depth:
                        obs = self._dummy_obs[..., :-1]
                    else:
                        obs = self._dummy_obs
                else:
                    state = self.game.get_state()
                    obs = state.screen_buffer
                    if self._add_dim:
                        obs = obs[:, :, None]
                if mode == 'agent':
                    return obs
                else:
                    if self._colormap is not None:
                        obs = cv2.LUT(np.tile(obs, 3), self._colormap[None])
                    if mode == 'rgb_array':
                        return obs
                    elif mode == 'human':
                        raise NotImplemented
        else:
            msg = "Exception raised by %s: " % self.__class__.__name__
            msg += "The environment needs to be reset at least once before rendering."
            raise RuntimeError(msg)
            
available_resolutions = {(160, 120), (200, 125), (200, 150), (256, 144), (256, 160), (256, 192), (320, 180), 
                         (320, 200), (320, 240), (320, 256), (400, 225), (400, 250), (400, 300), (512, 288), 

                         (512, 320), (512, 384), (640, 360), (640, 400), (640, 480), (800, 450), (800, 500), 
                         (800, 600), (1024, 576), (1024, 640), (1024, 768), (1280, 720), (1280, 800), 
                         (1280, 960), (1280, 1024), (1400, 787), (1400, 875), (1400, 1050), (1600, 900), 
                         (1600, 1000), (1600, 1200), (1920, 1080)}

class MyWayHomeTVEnv(EnvMaker):
    def __init__(self, 
                 tv=None, 
                 frame_skip=4, 
                 skip_pool=1, 
                 color_format='gray', 
                 screen_dims=(84, 84), 
                 game_coverage=(0.5, 0.5), 
                 tv_coverage=(1.0, 1.0), 
                 frame_stack=4, 
                 max_episode_steps=2100, 
                 num_envs=0, 
                 no_reward=False, 
                 tv_partitions=None, 
                 tv_colors=None, 
                 allow_tv_off=True, 
                 tv_rendering=None, 
                 game_resolution=None, 
                 visited_sectors=False):
        self._name = self.__class__.__name__
        self._tv = tv
        self._frame_skip = frame_skip
        self._skip_pool = skip_pool
        self._color_format = color_format
        self._screen_dims = screen_dims
        self._game_coverage = game_coverage
        self._tv_coverage = tv_coverage
        self._frame_stack = frame_stack
        self._max_episode_steps = max_episode_steps
        self.num_envs = num_envs
        self._no_reward = no_reward
        self._tv_partitions = tv_partitions
        self._tv_colors = tv_colors
        self._allow_tv_off = allow_tv_off
        self._tv_rendering = tv_rendering
        self._game_resolution = game_resolution
        self._visited_sectors = visited_sectors
        self._check_inputs()
        super().__init__()
        
    def _make(self):
        apply_warping = False
        apply_background = any([c < 1.0 for c in self._game_coverage])
        game_dims = (160, 120)
        game_dims_on_screen = (160, 120)
        if self._screen_dims is not None:
            game_dims_on_screen = tuple([int(d * c) for d, c in zip(self._screen_dims, self._game_coverage)])
        w = h = None
        if self._game_resolution is not None:
            w, h = self._game_resolution
        elif self._screen_dims is not None:
            w, h = game_dims_on_screen
        if w is not None and h is not None:
            highest_f = 0.0
            for res in available_resolutions:
                f0 = w / res[0] if w <= res[0] else res[0] / w
                f1 = h / res[1] if h <= res[1] else res[1] / h
                f = f0 * f1
                if f > highest_f:
                    game_dims = res
                    highest_f = f
            if self._game_resolution is None:
                apply_warping = not (1.0 - 1e-6 < highest_f < 1.0 + 1e-6)
            else:
                if self._screen_dims is not None:
                    apply_warping = any([gdim != int(sdim * c) for sdim, gdim, c in zip(self._screen_dims, game_dims, self._game_coverage)])
                if any([gres != gdim for gres, gdim in zip(self._game_resolution, game_dims)]):
                    msg = "WARNING (from %s): The desired 'game_resolution' is not allowed " % self._name
                    msg += "by VizDoom. A resolution of %sX%s will be set instead, which is " % game_dims
                    msg += "the closest in terms of aspect ratio."
                    print(msg)
        self._game_resolution = game_dims
        self._screen_dims = self._screen_dims or tuple([int(d / c) for d, c in zip(game_dims, self._game_coverage)])
        screen_format = {'rgb': 'RGB24', 'rgba': 'RGBA32', 'gray': 'GRAY8', 'original': 'DOOM_256_COLORS8'}[
            self._color_format]
        env = DoomMyWayHomeEnv(self._frame_skip, self._skip_pool, game_dims, self._max_episode_steps, 
                               screen_format, False)
        if apply_warping:
            env = WarpFrameEnv(env, game_dims_on_screen)
        if apply_background:
            env = AddBackgroundEnv(env, self._screen_dims)
        if self._tv is not None:
            env = TVEnv(env, self._tv, self._tv_coverage, self._tv_partitions, self._tv_colors, self._allow_tv_off, self._color_format, self._tv_rendering)
        if self._no_reward:
            env = NoRewardEnv(env)
        if self._visited_sectors:
            env = DoomMwhInfoEnv(env)
        if self.num_envs == 1:
            if self._frame_stack > 1:
                env = FrameStackEnv(env, self._frame_stack)
        else:
            env = VectorizeEnv(env, self.num_envs)
            if self._frame_stack > 1:
                env = FrameStackVecEnv(env, self._frame_stack)
        return env
    
    def get_config(self):
        config = {'environment': self._name, 
                  'tv': self._tv, 
                  'frame_skip': self._frame_skip,
                  'skip_pool': self._skip_pool, 
                  'color_format': self._color_format, 
                  'screen_dims': self._screen_dims, 
                  'game_coverage': self._game_coverage,   
                  'frame_stack': self._frame_stack, 
                  'max_episode_steps': self._max_episode_steps, 
                  'num_envs': self.num_envs, 
                  'no_reward': self._no_reward, 
                  'game_resolution': self._game_resolution}          
        if self._tv is not None:
            config.update({'tv_coverage': self._tv_coverage, 
              'tv_partitions': self._tv_partitions, 
              'tv_colors': self._tv_colors, 
              'allow_tv_off': self._allow_tv_off})
            if self._tv in {'sasr', 'madr'}:
                config.update({'tv_rendering': self._tv_rendering})
        return config
        
    def _check_inputs(self):
        raise_error(self._name, 'tv', (is_none(self._tv), is_inset(self._tv, ('broken', 'noisy', 'sasr', 'madr'))))
        raise_error(self._name, 'frame_skip', (is_integer(self._frame_skip, 'pos'),))
        raise_error(self._name, 'skip_pool', (is_integer(self._skip_pool, 'pos'),))
        if self._skip_pool > self._frame_skip:
            raise_error(self._name, 'skip_pool', ("lower than 'frame_skip'",))
        raise_error(self._name, 'color_format', (is_inset(self._color_format, ('rgb', 'gray', 'original')),))
        raise_error(self._name, 'screen_dims', (is_none(self._screen_dims), is_itemizable(self._screen_dims, (tuple, list), (2, eq))))
        if self._screen_dims is not None:
            raise_error(self._name, "Every item in 'screen_dims'", 
                        (is_each_item(self._screen_dims, (lambda x: is_integer(x, 'pos'),)),), quoted=False)
        raise_error(self._name, 'game_coverage', (is_itemizable(self._game_coverage, (tuple, list), (2, eq)),))
        raise_error(self._name, "Every item in 'game_coverage'", (is_each_item(self._game_coverage, (lambda x: is_real_inrange(x, (0.0, 1.0)),)),), quoted=False)
        raise_error(self._name, 'frame_stack', (is_integer(self._frame_stack, 'pos'),))
        raise_error(self._name, 'max_episode_steps', (is_large_int(self._max_episode_steps, 'pos'),))
        self._max_episode_steps = int(self._max_episode_steps)
        raise_error(self._name, 'num_envs', (is_integer(self.num_envs, 'nonneg'),))
        self.num_envs = self.num_envs if self.num_envs > 0 else psutil.cpu_count(logical=False)
        raise_error(self._name, 'no_reward', (is_bool(self._no_reward),), err=TypeError)
        raise_error(self._name, 'game_resolution', (is_none(self._game_resolution), is_itemizable(self._game_resolution, (tuple, list), (2, eq))))
        if self._game_resolution is not None:
            raise_error(self._name, "Every item in 'game_resolution'", 
                        (is_each_item(self._game_resolution, (lambda x: is_integer(x, 'pos'),)),), quoted=False)
        raise_error(self._name, 'visited_sectors', (is_bool(self._visited_sectors),), err=TypeError)
        if self._tv is not None:
            raise_error(self._name, 'tv_coverage', (is_itemizable(self._tv_coverage, (tuple, list), (2, eq)),))
            raise_error(self._name, "Every item in 'tv_coverage'", (is_each_item(self._tv_coverage, (lambda x: is_real_inrange(x, (0.0, 1.0)),)),), quoted=False)
            raise_error(self._name, 'tv_partitions', (is_none(self._tv_partitions), is_itemizable(self._tv_partitions, (tuple, list), (2, eq))))
            if self._tv_partitions is not None:
                raise_error(self._name, "Every item in 'tv_partitions'", 
                            (is_each_item(self._tv_partitions, (lambda x: is_integer(x, 'pos'),)),), quoted=False)
            else:
                self._tv_partitions = (4, 4) if self._tv == 'sasr' else (8, 8)
            raise_error(self._name, 'tv_colors', (is_none(self._tv_colors), is_integer(self._tv_colors, 'pos')))
            if self._tv_colors is None:
                self._tv_colors = 2 if self._tv == 'sasr' else 16
            raise_error(self._name, 'allow_tv_off', (is_bool(self._allow_tv_off),), err=TypeError)
            if self._tv in {'sasr', 'madr'}:
                raise_error(self._name, 'tv_rendering', (is_none(self._tv_rendering), is_inset(self._tv_rendering, ('static', 'dynamic'))))
                if self._tv_rendering is None:
                    self._tv_rendering = {'sasr': 'static', 'madr': 'dynamic'}[self._tv]
