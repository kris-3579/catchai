from __future__ import division, print_function
import tensorflow as tf
from keras.models import load_model
import catchai
import catchai_impelemt
import numpy as np
import os

DATA_DIR = "/Users/kristofreid/Desktop/catch_ai"
model = load_model(os.path.join(DATA_DIR, "rl-network.h5"))
model.compile(optimizer= "adam", loss= "mse")

game = catchai.MyWrappedGame()


num_games, num_wins = 0, 0
for e in range(100):
    game.reset()

    # get first state

    a_0 = 1
    x_t, r_0, game_over = game.step(a_0)
    s_t = catchai_impelemt.preprocess_images(x_t)

    while not game_over:
        s_tm1 = s_t
        q = model.predict(s_t)[0]
        a_t = np.argmax(q)
        x_t, r_t, game_over = game.step(a_t)
        s_t = catchai_impelemt.preprocess_images(x_t)
        if r_t == 1:
            num_wins += 1

    num_games += 1

    print("Game: {:03d}, Wins: {:03d}".format(num_games, num_wins), end="r")
print("")




