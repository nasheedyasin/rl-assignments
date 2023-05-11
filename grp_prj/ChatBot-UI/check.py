from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# tokenizer = AutoTokenizer.from_pretrained("nasheed/rl-grp-prj-gpt2-baseagent")

# model = AutoModelForCausalLM.from_pretrained("nasheed/rl-grp-prj-gpt2-baseagent")

# tokenizer = AutoTokenizer.from_pretrained("nasheed/rl-grp-prj-gpt2-base-persuader")
# model = AutoModelForCausalLM.from_pretrained("nasheed/rl-grp-prj-gpt2-base-persuader")

persuader, persuadee = tokenizer.special_tokens_map['additional_special_tokens']
base_text = 'Convince the persuadee to donate to a childrens charity.'+tokenizer.special_tokens_map['eos_token']
persuader, persuadee = tokenizer.encode(persuader,return_tensors='pt'), tokenizer.encode(persuadee,return_tensors='pt')

# Let's chat for 5 lines
for step in range(10):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, persuadee, new_user_input_ids, persuader], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids,penalty_alpha=0.6, top_k=4, max_new_tokens=128)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
