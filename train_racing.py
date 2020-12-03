import gym
import deepq
import sys


def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    sys.path.append('./sdc-gym/sdc_gym/gym/envs/box2d/')


    env = gym.make("CarRacing-v0")
    deepq.learn(env)
    env.close()


if __name__ == '__main__':
    main()

