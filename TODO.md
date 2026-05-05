# TODO

## Meeting April 09th
Main task figure out a good plan of juan on vllm to differentiate the agents with vLLM - in other words refine the prompt to better differentiate the agents and their roles in the system. This will help in better understanding and implementation of the system.

Classification goals:
- classify the prompts in stages
- classify steps that are difficult and those to that are easy.

Once you have performed classification, you can also chose the architecture to run on
- heavier models are on blackwell
- lighter models are on L40

## Meeting May 05th
From what we have:
- Test GridSearch on our approach, attach Mattia's framework
- work the a40, bench matte a copule of top-p and a couple of beam search and best of n, n=1 / 3

Focus on best of n = 3, check our code is fine with the majority voting
Comparisons with best f n = 1, n = 3.
Then compare the 3 approaches (best-of n and other beam, ...)
Also bayesian optimisation