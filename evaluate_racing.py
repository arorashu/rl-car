import gym
import deepq
import deepq_double
import sys

def main():
    """ 
    Evaluate a trained Deep Q-Learning agent 
    """ 
    sys.path.append('./sdc_gym/gym/envs/box2d/')
    env = gym.make("CarRacing-v0")
    deepq.evaluate(env)
    env.close()

if __name__ == '__main__':
    main()
