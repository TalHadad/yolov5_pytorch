This repository contains a PIP package which is an OpenAI environment for
simulating an enironment in which a cat is moving inside an image.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```

## Usage

```
import gym
import gym_mouse

env = gym.make('Mouse-v0')
```

See https://github.com/matthiasplappert/keras-rl/tree/master/examples for some
examples.


## The Environment

Imagine you are a mouse tying to avoid a cat.
