# deep-learning-from-human-preferences

## Summary

- [deep-learning-from-human-preferences](#deep-learning-from-human-preferences)
  - [Summary](#summary)
  - [Project](#project)
    - [Example of human feedback](#example-of-human-feedback)
    - [Benchmark](#benchmark)
    - [How it works](#how-it-works)
      - [Frame gather from Gymnasium API](#frame-gather-from-gymnasium-api)
      - [Rewards](#rewards)
      - [Reward Model](#reward-model)
      - [Training policy $\pi$](#training-policy-pi)
  - [References](#references)


## Project

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;The paper "Deep Learning from Human Preferences" by Christiano et al. (2017) explores the concept of training machine learning models using human feedback to achieve more aligned and desirable outcomes. The authors introduce a framework where deep reinforcement learning models are trained not just on predefined reward signals, but also on human evaluations of the model's performance. This approach allows the model to learn complex behaviors that are more closely aligned with human values and preferences, leading to more nuanced and effective decision-making in various tasks.
</p>

### Example of human feedback

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;Select preferable segment or tell if they are incomparable.
</p>

<div align="center">
    <img src="media/HumanFeedbackInterface.gif" width=90%>
    <div style="font-size:0.8em;">
        Human feedback interface.
    </div>
</div>

### Benchmark

&nbsp;&nbsp;&nbsp;&nbsp;To validate the implementation a benchmark was made. The following results are using a gymnasium env on Enduro game from Atari.

<div align="center">
    <img src="media/benchmark.png">
    <div style="font-size:0.8em;">
        Accumulative rewards gather during train on Enduro from Atari.
    </div>
</div>

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;The model can be improved by optimizing the hparams and collecting more human feedbacks. Note that, given that humans has to give theirs feedbacks to the machine, the training process is consederably slow in terms of time.
</p>

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;During the training process, only 290 feedbacks (preference between segments) where given. On original papers, they collected 5.5k feedbacks.
</p>

<div align="center">
    <img src="media/D_size_during_train.png">
    <div style="font-size:10px;">
        Gathered human feedbacks count during train.
    </div>
</div>

### How it works

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;In the following will be presented the steps of training process:
</p>

#### Frame gather from Gymnasium API
<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;On the current DLFHP implementation was used Gymnasium, which is a fork of OpenAI’s Gym library. "OpenAI Gym is a toolkit for reinforcement learning research. It includes a growing collection of benchmark problems that expose a common interface, and a website where people can share their results
and compare the performance of algorithms. This whitepaper discusses the components of OpenAI Gym
and the design decisions that went into the software" (Brockman et al., 2016).
</p>

<p align="justify">

&nbsp;&nbsp;&nbsp;&nbsp;An agent, policy $\pi$, can interact with env from Gymnasium API, gathering a observation, from that observation the agents takes an action and, from that action, receives a reward and a new observation.

</p>

<div align="center">
    <img src="media/raw_observation.png">
    <div style="font-size:10px;">
        Observation gathered from Enduro Env.
    </div>
</div>

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;To simplify the observation to the CNN model and improve a faster learning, the observation is preprocessed to grayscale, resized to (80,80) and normalized between range [0,1] by dividing each pixel by 255.
</p>

<div align="center">
    <img src="media/preprocessed_observation.png">
    <div style="font-size:10px;">
        Rescaled grayscale observation.
    </div>
</div>

<p align="justify">

&nbsp;&nbsp;&nbsp;&nbsp;These observation are feed to an policy $\pi$ that will be trained using Reinforcement Learning, using rewards as feedbacks on how good they actions were. The policy $\pi$ are feed with sequences of $n$ observations.

</p>

#### Rewards

<p align="justify">

&nbsp;&nbsp;&nbsp;&nbsp;The Gymnasium API already provides rewards for each step an policy $\pi$ takes on it envs, but we will modify it to use our Reward Model that predicts rewards for each timestep. This model will be feed with the same observation that our policy $\pi:o \rightarrow a$ receives, and the action $a$ taken by the $\pi$ on that same observation will be feed together into the Reward Model. The policy $\pi$ are trained using the rewards gather from the Reward Model.

</p>

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;The rewards given by the Gymnasium API gonna be used to compare two agents: one trained using the the Reward Model an another trained using the true rewards (rewards from Gymnasium API). While training the model trained using feedbacks will never use the true rewards.
</p>

#### Reward Model

<p align="justify">

&nbsp;&nbsp;&nbsp;&nbsp;Using feedbacks from human to train an policy $\pi$ was a complex task that could not help to solve many complex environments. The paper "Deep Learning from Human Preferences" by Christiano et al. (2017) aims to solve that problem. "We show that this approach can effectively solve complex RL tasks without access to the reward function, including Atari games and simulated robot locomotion, while providing feedback on less than 1% of our agent’s interactions with the environment"(Christiano et al., 2017).

&nbsp;&nbsp;&nbsp;&nbsp;To predict the rewards the model must have a observation $o_t$ and a action $a_t$ that the policy $\pi$ takes on $o_t$, so reward $r_t=\hat{r}(o_t, a_t)$.

&nbsp;&nbsp;&nbsp;&nbsp;Adjusting reward function $\hat{r}$ can be done using human preferences. The preferences are made based on segments $\sigma = ((o_0, a_0), (o_1, a_1), ..., (o_{k-1}, a_{k-1}))$. The humans must select a segment they prefer over another segment $\sigma^1 \succ \sigma^2$ or tell if they are incomparable. Preferences of segments $\sigma$ are stored on a tuple $D$ as tuple $(\sigma^1, \sigma^2, \mu)$, the $\mu$ is a distribution of segments $\sigma$ preference. A crossentropy loss can train the model predicting the probabilities $\hat{P}[\sigma^1 \succ \sigma^2]$ and  $\mu$ as labels.

</p>

#### Training policy $\pi$

<p align="justify">

&nbsp;&nbsp;&nbsp;&nbsp;Using the preprocessed observation $o_t$ the policy $\pi$ takes an action $a_t$ receiving an $\hat{r}_t$ and a new observation $o$ for $k$ timesteps, these experiences are stored on a tuple $T=(\sigma^{0},\sigma^{1},..., \sigma^{k-1})$. After $k$ timesteps the policy $\pi$ is updated using PPO method (Schulman et al, 2017) on predicted reward $r$ from Reward Model $\hat{r}$. Then humans select they preferences between random sampled segment $\sigma$ from tuple $T$ and store the preferences  $\sigma^1 \succ \sigma^2$ on tuple $D$ so that the Reward Model $\hat{r}$ can be updated. After this process, the loop continues until timesteps $k=30.000$.

</p>

## References
- Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems (pp. 4299-4307).
- Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). OpenAI Gym. arXiv preprint arXiv:1606.01540. Retrieved from https://arxiv.org/abs/1606.01540
- Schulman, John, et al. "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347 (2017). https://arxiv.org/abs/1707.06347