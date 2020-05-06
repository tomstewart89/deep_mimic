import sys
import pybullet_data

from pybullet_envs.deep_mimic.learning.rl_world import RLWorld
from pybullet_utils.arg_parser import ArgParser
from pybullet_envs.deep_mimic.env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv


if __name__ == '__main__':

  enable_draw = True
  timestep = 1. / 240.

  args = sys.argv[1:]

  arg_parser = ArgParser()
  arg_parser.load_args(args)
  arg_file = arg_parser.parse_string('arg_file', "run_humanoid3d_spinkick_args.txt")
  arg_parser.load_file(pybullet_data.getDataPath() + "/args/" + arg_file)
  
  env = PyBulletDeepMimicEnv(arg_parser, enable_draw)
  world = RLWorld(env, arg_parser)

  world.reset()

  total_reward = 0
  steps = 0

  while True:

    world.update(timestep)
    total_reward += world.env.calc_reward(agent_id=0)

    steps+=1

    if world.env.is_episode_end() or steps>= 1000:
      total_reward = 0
      steps = 0
      world.end_episode()
      world.reset()

  world.shutdown()
