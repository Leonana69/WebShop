import json

with open('text_data/fork_leaf_responses_copy.json', 'r') as f:
    string = f.read()
    # count the 'join_init' in string
    count = string.count('join_init')
    print(count)