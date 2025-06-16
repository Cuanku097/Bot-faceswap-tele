#!/bin/bash
tmux kill-session -t swapbot 2>/dev/null
cd ~/faceswap
tmux new-session -d -s swapbot 'python3 bot_swap.py'
