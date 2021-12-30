Welcome to my fork of OpenAi's Spinning Up repo
===============================================

When I started experimenting with spinning up for continuous control tasks, I noticed a couple of things that the original repo doesn't have. So I decided to create my own fork and add the stuff that I needed. Going forward, I'd like to use this repo to explore and document some of my own research ideas and do some cool RL experiments. Feel free to reach out and let me know what you think.

Stuff that I've added
---------------------
Some of the changes that my fork has to offer are:
- GPU support for the pytorch algorithms.
- A more flexible way to create the models behind (some of) the pytorch algorithms.
- Support for multi modal obervation spaces, including CNN models for pixel based observations.
- A bunch of gym environment wrappers and techniques that I found useful - e.g., HER or random network distillation.

Some other cool RL projects that you might find interesting  
-----------------------------------------------------------
As I'd like to use this repo for my own experiments and can't really guarantee that everything will always be 100% correct, I thought that I could at least collect some links to other cool RL projects that I came across. So, if nothing else, the following list might at least help you to find a project that works for your RL tasks:
- To get a really nice collection of RL algoritms, check out https://github.com/DLR-RM/stable-baselines3.
- The stable baselines above can even be combined with a bunch of tools and pretrained weights from https://github.com/DLR-RM/rl-baselines3-zoo.
- Another pretty powerful looking RL project is https://github.com/parilo/tars-rl.
- And here's a framework that I liked to use when working with discrete action spaces: https://github.com/unixpickle/anyrl-py.
