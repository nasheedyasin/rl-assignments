import random
import json
import os

import torch
import nltk



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


# tokenizer = AutoTokenizer.from_pretrained("nasheed/rl-grp-prj-gpt2-base-persuader")
# model = AutoModelForCausalLM.from_pretrained("nasheed/rl-grp-prj-gpt2-base-persuader")
# persuader, persuadee = tokenizer.special_tokens_map['additional_special_tokens']
# count = 0

# ## 
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # tokenizer = AutoTokenizer.from_pretrained("nasheed/rl-grp-prj-gpt2-baseagent")
# # model = AutoModelForCausalLM.from_pretrained("nasheed/rl-grp-prj-gpt2-baseagent")
# # persuader, persuadee = tokenizer.special_tokens_map['additional_special_tokens']


# base_text = 'Convince the persuadee to donate to a childrens charity.'+tokenizer.special_tokens_map['eos_token']
# persuader, persuadee = tokenizer.encode(persuader,return_tensors='pt'), tokenizer.encode(persuadee,return_tensors='pt')

PATH = "data.txt"

open(PATH, "a").close()

# Save the tensor data to the specified path



bot_name = "Sam"
resetnum =10

data = {
    "user": [],
    "response": [] 
}
with open('output.json', 'w') as outfile:
    # dump the dictionary object to the output file as a JSON object
    json.dump(data, outfile)

def get_response(msg):
    '''
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    '''
    global count
    with open('output.json', 'r') as json_data:
        lastdata = json.load(json_data)


    if ((msg == 'clear') | (len(lastdata['user'])>resetnum)) :
        with open('output.json', 'r') as infile:
            # load the JSON data into a Python object
            data = json.load(infile)

            # delete the value from the Python object
            del data["user"]
            del data ["response"]

        # open the same file for writing
        with open('output.json', 'w') as outfile:
            # write the modified Python object back to the file
            data = {
             "user": [],
             "response": [] 
                }
            json.dump(data, outfile)
            return 'cleared...start new session started!!....'


    # encode the new user input, add the eos_token and return a tensor in Pytorch
    #new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')
    #new_user_input_ids = tokenizer.encode(tokenizer.eos_token + msg, return_tensors='pt')
    new_user_input_ids = tokenizer.encode(msg + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = new_user_input_ids
    if len(lastdata['user']) ==0:
       # print("BULL")
        bot_input_ids = new_user_input_ids
        if os.path.exists(PATH):
            os.remove(PATH)
            # Create a new empty file with the same name
        open(PATH, "a").close()
    else:
        data = torch.load(PATH) 
        #print("DialoGPT: {}".format(tokenizer.decode(data, skip_special_tokens=True)))

        #bot_input_ids = torch.cat([data, new_user_input_ids], dim=-1)
        bot_input_ids = torch.cat([data, new_user_input_ids], dim=-1) 
        
        print("History Input: {}".format(tokenizer.decode(bot_input_ids[0], skip_special_tokens=False)))

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    torch.save(chat_history_ids, PATH)

    # pretty print last ouput tokens from bot
    #print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
#    print(bot_input_ids,chat_history_ids)
    # open the JSON file for reading

    
    new_response = format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))

    lastdata["user"].append(msg)
    lastdata["response"].append(new_response)
    with open('output.json', 'w') as outfile:
        json.dump(lastdata, outfile)
    
    return new_response 


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break
        

        resp = get_response(sentence)
#        resp = 'your are good' 
        print(resp)

