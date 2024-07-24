# deep-learning-from-human-preferences

## About

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
In the following will be presented the steps of training process:
</p>

#### Frame gather from Gymnasium API
<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;On the current DLFHP implementation was used Gymnasium, which is a fork of OpenAI’s Gym library. "OpenAI Gym is a toolkit for reinforcement learning research. It includes a growing collection of benchmark problems that expose a common interface, and a website where people can share their results
and compare the performance of algorithms. This whitepaper discusses the components of OpenAI Gym
and the design decisions that went into the software" (Brockman et al., 2016).
</p>

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;An agent can interact with env from Gymnasium API, gathering a observation, from that observation the agents takes an action and, from that action, receives a reward and a new observation.
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
&nbsp;&nbsp;&nbsp;&nbsp;These observation are feed to an agent that will be trained using Reinforcement Learning, using rewards as feedbacks on how good they actions were. The model has a LSTM layer and is feed with sequences of observations.
</p>

#### Rewards

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;The Gymnasium API already provides rewards for each step an agent takes on it envs, but we will modify it to use our Reward Model that predicts rewards for each timestep. This model will be feed with the same observation that our agent receives, and the action taken by the agent on that same observation will be feed together into the Reward Model. Our agent are trained using the rewards gather from the Reward Model.
</p>

#### Training the Reward Model

todo ...

#### todo ...

todo ...

...

## References
- Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems (pp. 4299-4307).
- Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). OpenAI Gym. arXiv preprint arXiv:1606.01540. Retrieved from https://arxiv.org/abs/1606.01540