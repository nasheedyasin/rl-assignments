import torch
import random
import gymnasium as gym

from itertools import count
from tqdm import tqdm, trange
from convokit import Corpus
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM

from typing import Optional

from modelling import DialogAgent
from rewards import PersuasionRewards
from environment import PersuasionEnvironment


class PPOTrainer(object):
    def __init__(
        self, 
        data_path: str,
        mpath: str,
        gamma: float,
        rewards_module: PersuasionRewards,
        num_epochs: int, # Number of times to run through all episodes
        batch_size: int = 16, # Episodes in a batch
        mini_batch_size: int = 1,
        human_proxy_path: Optional[str] = None,
        purpose_text: str = "Convince people to donate to charities.",
        enable_cuda: bool = True
    ):
        # Check if CUDA is enabled
        if enable_cuda and torch.cuda.is_available(): self.device = 'cuda'
        else: 'cpu'

        self.gamma = gamma
        self.num_epochs = num_epochs
        self.purpose_text = purpose_text
        self.rewards_module = rewards_module
        self.mini_batch_size = mini_batch_size

        self.episode_durations = []
        self.episode_rewards = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.action = []
        self.states = []
        self.true_states = [] 
        self.terminate = []
        self.batch_size = batch_size
        self.eps_clip=0.2 

        # Init the data
        corpus = Corpus(data_path)
        self.episodes = [conv for conv in corpus.iter_conversations()]

        # Setup the tokenizers
        self.env_tokenizer = AutoTokenizer.from_pretrained(
            human_proxy_path if human_proxy_path is not None else mpath
        )
        self.env_tokenizer.padding_side = 'left'
        self.env_tokenizer.truncation_side = 'left'        
        self.state_tokenizer = AutoTokenizer.from_pretrained(mpath)
        self.state_tokenizer.padding_side = 'left'
        self.state_tokenizer.truncation_side = 'left'
        self.action_tokenizer = AutoTokenizer.from_pretrained(mpath)

        # Init the model
        self.human_proxy = AutoModelForCausalLM.from_pretrained(
            human_proxy_path if human_proxy_path is not None else mpath
        ).to(self.device)
        self.dialog_agent = DialogAgent(mpath).to(self.device)

        # Setup Optimization
        self.optimizer = self.dialog_agent.configure_optimizers()['optimizer']

    def train(self):
        epoch_iterator = trange(self.num_epochs, desc='Training Epoch:')
        
        for epoch in epoch_iterator:
            # Shuffle the episodes
            random.shuffle(self.episodes)
            step_iterator = trange(
                start=0,
                stop=len(self.episodes),
                step=self.batch_size,
                desc='Training Step:',
                leave=False
            )

            for step in step_iterator:
                step_episodes = self.episodes[step: step+self.batch_size]
                for episode in step_episodes:
                    episode_reward = 0
                    env = PersuasionEnvironment(
                        episode,
                        self.human_proxy,
                        self.env_tokenizer,
                        self.rewards_module,
                        self.purpose_text,
                        render_mode='human'
                    )

                    state, info = env.reset()
                    true_state = info['true_state']

                    for time_step in count():
                        self.states.append(state)
                        self.true_states.append(true_state)
                        state_tokens = self.state_tokenizer(
                            state, truncation=True,
                            return_tensors='pt'
                        )
                        true_state_tokens = self.state_tokenizer(
                            true_state, truncation=True,
                            return_tensors='pt'
                        )

                        action_tokens, log_prob, _ = \
                            self.dialog_agent.get_action_and_values(state_tokens)
                        value = self.dialog_agent.get_values(true_state_tokens)

                        self.log_probs.append(log_prob) # 1-D Tensor
                        self.values.append(value) # 0-D Tensor

                        action = self.action_tokenizer.batch_decode(
                            action_tokens, skip_special_tokens=True)[0]

                        self.action.append(action)

                        state, reward, terminated, truncated, info = env.step(action)
                        true_state = info['true_state']

                        self.rewards.append(reward)

                        done = terminated or truncated
                        episode_reward += reward

                        self.terminate.append(done)

                        if done:
                            self.episode_durations.append(time_step)
                            self.episode_rewards.append(episode_reward/time_step)
                            break

                # Update once per step
                self.update()

                self.rewards = []
                self.log_probs = []
                self.values = []
                self.states = []
                self.true_states = []
                self.action = []
                self.terminate = []

    def update(self):
        policy_loss = []
        value_loss = []
        n_time_steps = len(self.rewards)
        # Accumulate gradients over the batch
        accelerator = Accelerator(gradient_accumulation_steps=n_time_steps)

        q_values = torch.empty(n_time_steps, dtype=torch.float).to(self.device)

        # Calculate GAE based Advantage
        value_t_p_1 = 0 # Since we always ensure completed episodes
        for t in reversed(range(n_time_steps)):
            if self.terminate[t]: value_t_p_1 = 0

            q_value = self.rewards[t] + self.gamma * value_t_p_1
            q_values[t] = q_value
            # The value for time step t+1
            value_t_p_1 = self.values[t]

        values = torch.stack(self.values)
        advantage = q_values - values

        # Normalizing advantage for stability
        mean_ = torch.mean(advantage)
        std_ = torch.std(advantage)
        advantage = (advantage-mean_) / (std_ + 1e-5)

        log_probs_old = torch.cat(self.log_probs)

        for mini_step in range(0, n_time_steps, self.mini_batch_size):
            minib_state_tokens = self.state_tokenizer(
                self.states[mini_step: self.mini_batch_size],
                 padding=True, truncation=True, return_tensors='pt'
            )
            minib_true_state_tokens = self.state_tokenizer(
                self.true_states[mini_step: self.mini_batch_size],
                 padding=True, truncation=True, return_tensors='pt'
            )
            minib_action_tokens = self.state_tokenizer(
                self.action[mini_step: self.mini_batch_size],
                padding=True, truncation=True, return_tensors='pt'
            )
            minib_advantage = advantage[mini_step: self.mini_batch_size]
            minib_log_probs_old = log_probs_old[mini_step: self.mini_batch_size]

            with accelerator.accumulate(self.dialog_agent):
                minib_log_probs = self.dialog_agent.get_log_probs(
                    minib_state_tokens,
                    minib_action_tokens
                )
                minib_values = self.dialog_agent.get_values(minib_true_state_tokens)

                # Taking this ratio as an approximation to KLdivergence
                ratios = (minib_log_probs-minib_log_probs_old.detach()).exp()

                surr1 = ratios * minib_advantage.detach()
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) \
                    * minib_advantage.detach()

                policy_loss = -torch.min(surr1, surr2).sum()

                criterion = torch.nn.SmoothL1Loss()
                value_loss = criterion(minib_values, q_values)

                self.optimizer.zero_grad()
                policy_loss.backward()
                value_loss.backward()
                self.optimizer.step()

    def evaluate(self, num_episodes):
        episodes = self.episodes[: num_episodes]
        ep_iterator = tqdm(episodes, desc='Evaluation Episode:')

        for episode in ep_iterator:
            episode_reward = 0
            env = PersuasionEnvironment(
                episode,
                self.human_proxy,
                self.env_tokenizer,
                self.rewards_module,
                self.purpose_text,
                render_mode='human'
            )

            state, _ = env.reset()

            for time_step in count():
                state_tokens = self.state_tokenizer(
                    state, truncation=True,
                    return_tensors='pt'
                )

                action_tokens, _, _ = \
                    self.dialog_agent.get_action_and_values(state_tokens)

                action = self.action_tokenizer.batch_decode(
                    action_tokens, skip_special_tokens=True)[0]

                state, reward, terminated, truncated, _ = env.step(action)

                self.rewards.append(reward)

                done = terminated or truncated
                episode_reward += reward

                self.terminate.append(done)

                if done:
                    self.episode_durations.append(time_step)
                    self.episode_durations.append(episode_reward/time_step)
                    break

    def plot(self, mode='Training'):
        import matplotlib.pyplot as plt

        plt.rcParams["figure.figsize"] = (30,14)
        plt.rcParams["font.size"] = 22

        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.set_xlabel(f'{mode} Conversations')
        ax1.set_ylabel('Running Duration (Turns)')
        ax1.plot(self.episode_durations, linewidth=6, label='Per Episode', color='#0077b6')
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel(f'{mode} Conversations')
        ax2.set_ylabel('Epiosde Avg Reward (reward/turn)')
        ax2.plot(self.episode_durations, linewidth=6, label='Running Average', color='#420c09')
        ax2.legend()
        ax2.grid(True)
