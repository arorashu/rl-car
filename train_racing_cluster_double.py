from pyvirtualdisplay import Display
import gym
import deepq_double
import sys


def main():
    """ 
    Train a Deep Q-Learning agent in headless mode on the cluster
    """ 
    sys.path.append('./sdc-gym/sdc_gym/gym/envs/box2d/')
    display = Display(visible=0, size=(800,600))
    display.start()
    env = gym.make("CarRacing-v0")
    #deepq.learn(env)
    deepq_double.learn(env,
                   gamma=0.99,
                   model_identifier='agent-act-7-double')
    env.close()
    display.stop()


if __name__ == '__main__':
    main()

