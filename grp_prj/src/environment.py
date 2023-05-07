import torch
import string
import gymnasium as gym

from gymnasium import spaces
from convokit import Conversation, Utterance
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Any, Dict, List, Optional, Tuple

from rewards import PersuasionRewards


class PersuasionEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        conervsation: Conversation,
        human_proxy: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_module: PersuasionRewards,
        purpose_text: str = "",
        max_length=768,
        render_mode: Optional[str] = None,
    ):
        self.purpose_text = purpose_text
        self.human_proxy = human_proxy
        self.tokenizer = tokenizer
        self.reward_module = reward_module
        self.max_length = max_length
        self.device = self.human_proxy.device
        self.render_mode = render_mode

        self.observation_space = gym.spaces.Text(
            2**13, charset=string.digits+string.ascii_letters+string.punctuation+' '
        )
        self.action_space = gym.spaces.Text(
            2**13, charset=string.digits+string.ascii_letters+string.punctuation+' '
        )

        # We truncate from the left to keep latest context intact
        self.tokenizer.truncation_side = 'left'
        # GPT Models don't have a padding requirement, hence this is not set
        # GPT Models have all special tokens set to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get the role markers
        self.role_markers = self.tokenizer.special_tokens_map['additional_special_tokens']

        # Enumerate the conversation utterances
        # Persuader is label (int) 0 and Persuadee is label (int) 1
        self.gold_persuader_utts: List[Utterance] = list()
        self.gold_persuadee_utts: List[Utterance] = list()
        for utt in conervsation.iter_utterances():
            if utt.meta['role'] == 0: self.gold_persuader_utts.append(utt)
            else: self.gold_persuadee_utts.append(utt)

        self.max_turns = len(self.gold_persuader_utts)

    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        # Initialize the env state
        self.state = f'{self.tokenizer.bos_token} {self.purpose_text} '\
            f'{self.role_markers[0]}'

        self.turn = 0
        self.action_t_minus_1 = ""

        return self.state, {'true_state': self.state}

    def step(
        self, action: Optional[str]="", sanity_test_mode: bool = False
    ):
        assert self.turn < len(self.gold_persuader_utts), \
            "Calling step after episode truncation/termination."

        # We keep track of the true state seperately
        # Otherwise, since we are doing imitation learning, we doctor the state to
        # resemble gold conversation history/trajectory
        gold_action = self.gold_persuader_utts[self.turn].text
        # Sometimes on the last turn, the pursuadee does not reply
        if self.turn == len(self.gold_persuadee_utts): gold_response = ""
        else: gold_response = self.gold_persuadee_utts[self.turn].text
        true_state = f'{self.state} {action} {self.role_markers[1]}'

        self.state = f'{self.state} {gold_action} '\
            f'{self.role_markers[1]} {gold_response} {self.role_markers[0]}'

        # Truncating the state to be at most 'max_length'
        state_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(self.state))
        state_ids = self.tokenizer.truncate_sequences(
            state_ids,
            num_tokens_to_remove=len(state_ids)-self.max_length
        )[0]
        self.state = self.tokenizer.decode(state_ids)

        # While sanity testing we pass the same text for both
        # action and gold response
        if sanity_test_mode: action = gold_action

        batch = (
            self.action_t_minus_1,
            action,
            gold_action
        )

        reward = self.reward_module.predict_step(batch).item()

        # While sanity testing we just pick up the golden persuadee
        # utterance for the time step
        if sanity_test_mode:
            true_state = self.state
        else:
            # Tokenize the `true_state`
            prompt_ids = self.tokenizer(
                true_state,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            # Move to the human_proxy's device
            prompt_ids = {k: v.to(self.device) for k, v in prompt_ids.items()}

            with torch.no_grad():
                # We use contrastive sampling
                # refer https://huggingface.co/blog/introducing-csearch
                true_state_ids = self.human_proxy.generate(
                    **prompt_ids,
                    penalty_alpha=0.6,
                    top_k=4, max_new_tokens=128
                )

            # Truncating the true_state to be at most `max_length`
            true_state_len = len(true_state_ids[0])
            true_state_ids = self.tokenizer.truncate_sequences(
                true_state_ids,
                num_tokens_to_remove=true_state_len-self.max_length
            )[0]

            # The `eos` token is the last `input_id` and we want to skip that.
            true_state = self.tokenizer.decode(true_state_ids[0][:-1])
            true_state = f'{true_state} {self.role_markers[0]}'

        # Proceed with the next step
        self.action_t_minus_1 = action
        self.turn += 1

        if self.turn == self.max_turns: terminated = truncated = True
        else: terminated = truncated = False

        return (
            self.state, reward,
            terminated, truncated,
            {'true_state': true_state}
        )

    def render(self, is_jupyter: bool = True):
        if not self.render_mode == "human": raise NotImplementedError('Coming soon!')

        import re

        # Clear the cell op
        if is_jupyter:
            from IPython.display import clear_output
            clear_output(wait=True)
        else:
            import os, platform
            if platform.system() == 'Windows':
                os.system('cls')
            else:
                os.system('clear')

        delimiter_pattern = "|".join(map(re.escape, self.role_markers))
        result = re.findall(f'(.*?{delimiter_pattern}.*)', self.state)

        print(*result, sep='\n')
