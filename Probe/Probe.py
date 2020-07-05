"""Probe for AIPlayground.
Author: Mehdi Mousavi - Last Edit: June 2020
For More Customizations, Look at Probe_utils.py"""

import os
import sys
import time

import numpy as np
from pykeyboard import PyKeyboard

from Probe_utils import MK, Probe, WindowMgr, KeyUtils

LINUX = 'linux'
WINDOWS = 'win32'

# Switch focus to the game window
if sys.platform == LINUX:
    # In Linux, we can shift focus to a window by using the wmctrl program

    gameName = "Loft (64-bit, PCD3D_SM5)"
    os.system('wmctrl -a ' + gameName)
elif sys.platform == WINDOWS:
    # In Windows, we can use the clascs defined above
    w = WindowMgr()

    # Note, I don't have Unreal set up in my Windows machine, so I used an alternate example
    w.find_window_wildcard(".*AI_Play")
    w.set_foreground()

"""Global variables You can Change to sense the environment"""

nSteps = 2000   # The number of Random steps Generated (Always make this a huge number)
count = 0
stepsize = 16
current_location = np.zeros(2)
steps = np.zeros(nSteps)

imagesets = 1000    # Number of image sequences you want to take


"""Initiating The Files."""

moves_file = "ProbeLog\look_rotate.txt"

ku = KeyUtils()
pkey = Probe()
k = PyKeyboard()
# generate_moves(nSteps, moves_file) IF you are reading moves you need to Comment this out.

""" Gathering Data """

file = open(moves_file)

stepcount=0

for action in list(file.readlines()):
    if imagesets > 0:
        look, move, intensity, shots = action.split(',')
        look = int(look)
        move = int(move)
        intensity = int(intensity)
        shots = int(shots)
        print('step = ' + str(stepcount))
        stepcount += 1

        if look == 5:  # UP
            k.tap_key(k.scroll_lock_key, n=2, interval=0.01)  #change N to intensity later
            print('Up look')

        elif look == 6:  # DOWN
            k.tap_key(k.home_key, n=2, interval=0.01)
            print('Down look')

        elif look in MK['LOOK_RIGHT']:
            k.tap_key(k.page_up_key, n=intensity, interval=0.01)

        else:  # LEFT
            k.tap_key(k.insert_key, n=intensity, interval=0.01)

        if move == 1:

            pkey.global_step(num_pad_key=8, step_size=stepsize)

        elif move == 2:
            pkey.global_step(num_pad_key=2, step_size=stepsize)

        elif move == 3:
            pkey.global_step(num_pad_key=6, step_size=stepsize)

        elif move == 4:
            pkey.global_step(num_pad_key=4, step_size=stepsize)

        if shots == 1:
            imagesets = imagesets - 1
            rotate = 2

            """ 284 Rotations will result in a 360 Degree rotation ingame.  """

            while rotate >= 0:
                k.tap_key(k.page_up_key, n=94,interval=0.005)
                pkey.takepics()
                rotate = rotate-1
                #print('\r rotating...' + str(rotate) + '\r')
                print('rotates left: ' + str(rotate) + '/284 | ' + str(imagesets) + ' image sets remain')




        elif shots == 0:
            print('\r')


# Pause at the end to avoid transitioning away from the game too abruptly
time.sleep(0.5)

# Switch focus back to the Python window
if sys.platform == LINUX:
    # Using Alt-Tab in Linux. A more robust way would be to identify the
    # window from which the script was launched
    ku.alt_tab()
elif sys.platform == WINDOWS:
    # Assumes that the Python script was launched from pycharm
    w.find_window_wildcard(".*PyCharm*.")
    w.set_foreground()
