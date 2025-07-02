import json

# tools = [{
#     "name": "search_shop",
#     "description": "Search for a range of items in shop database",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "keywords": {
#                 "type": "string",
#                 "description": "Item to search for"
#             },
#             "range": {
#                 "type": "string",
#                 "description": "Range of results to return"
#             }
#         },
#         "required": ["keywords", "range"]
#     },
# }]

# print(repr(json.dumps(tools)))

with open('text_data/fork_leaf_responses.json') as f:
    dataset = json.load(f)
    for item in dataset:
        convs = item['conversations']
        new_convs = []
        gpt = ''
        for conv in convs:
            if conv['from'] == 'human':
                new_convs.append(conv)
                continue

            if conv['from'] == 'gpt':
                gpt += conv['value'] + '\n'
            elif conv['from'] == 'function_call':
                gpt += '<|function_call_begin|>' + conv['value'] + '<|function_call_end|><|eot_id|>\n'
            elif conv['from'] == 'observation':
                gpt += '<|start_header_id|>' + 'ipython' + '<|end_header_id|>\n'
                value = conv['value']
                gpt += f'{{"items": {value}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'
            else:
                raise ValueError(f"Unknown conversation type: {conv['from']}")
        
        new_convs.append({
            'from': 'gpt',
            'value': gpt
        })
        item['conversations'] = new_convs
    
    with open('text_data/fork_leaf_responses.json', 'w') as f:
        json.dump(dataset, f, indent=4)