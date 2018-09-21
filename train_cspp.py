import gym

from baselines import deepq

def callback():

def main():
    env =
    model =
    act = deepq.learn(

    )
    print("Saving model to cspp.pkl")
    act.save("cspp_model.pkl")
if __name__ == '__main__':
    main()
