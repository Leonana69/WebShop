import gym
import copy, re
import sys, os, json
from openai import OpenAI
from tqdm import tqdm
import random
sys.path.append('..')
from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.utils import DEBUG_PROD_SIZE
env = gym.make('WebAgentTextEnv-v0', observation_mode='text_rich', num_products=DEBUG_PROD_SIZE)
env.reset()

FILE_FOLDER = "text_data/"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_LOG_FILE = os.path.join(CURRENT_DIR, FILE_FOLDER + "chat_log.txt")

def llm_request(prompt: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )

    with open(CHAT_LOG_FILE, "a") as f:
        f.write(prompt + "\n---\n")
        f.write(response.model_dump_json(indent=2) + "\n---\n")
    return response.choices[0].message.content

def get_item(keywords: str, rg: str):
    print(f'[S] search_range[{keywords}, {rg}]')
    env.step(f'search_range[{keywords}, {rg}]')
    return env.observation.split('[button] Next > [button_]')[1].strip()

def generate_user_requests(n: int = 10):
    requests = []
    for i in tqdm(range(n // 5)):
        with open('user_request_prompt.txt') as f:
            prompt = f.read()
        response = llm_request(prompt)
        js = json.loads(response.lstrip('```json').rstrip('```'))
        requests = requests + js

    print(requests)
    with open(FILE_FOLDER +'user_requests.json', 'w') as f:
        json.dump(requests, f, indent=4)

def generate_main_responses():
    with open(FILE_FOLDER +'user_requests.json') as f:
        requests = json.load(f)
    with open('main_response_prompt.txt') as f:
        prompt = f.read()

    with open(FILE_FOLDER +'main_responses.json', 'r') as f:
        main_responses = json.loads(f.read())

    requests = requests[:50]
    for req in tqdm(requests):
        k = ' '.join(req['keywords'])

        r0 = random.randint(1, 5)
        r1 = random.randint(6, 10)
        search_result_0 = get_item(k, f'{r0}-{r0}').replace('\n', ' ')
        search_result_1 = get_item(k, f'{r1}-{r1}').replace('\n', ' ')

        _prompt = prompt.replace("{user_request}", req['value'])
        _prompt = _prompt.replace("{search_result_0}", search_result_0)
        _prompt = _prompt.replace("{search_result_1}", search_result_1)
        response = llm_request(_prompt)
        js = json.loads(response.lstrip('```json').rstrip('```'))
        mr = {}
        mr["keywords"] = req['keywords']
        mr["conversations"] = []
        user_input = {"from": "human", "value": req['value']}
        gpt_output = {"from": "gpt", "value": js['response']}
        mr["conversations"].append(user_input)
        mr["conversations"].append(gpt_output)
        
        main_responses.append(mr)

        # save to file
        with open(FILE_FOLDER + 'main_responses.json', 'w') as f:
            json.dump(main_responses, f, indent=4)
    
def process_main_responses():
    with open(FILE_FOLDER + 'main_responses.json') as f:
        main_responses = json.load(f)
    
    processed_responses = []
    for mr in main_responses:
        mr.pop('keywords')
        mr['tools'] = '[]'
        processed_responses.append(mr)

    with open(FILE_FOLDER + 'processed_main_responses.json', 'w') as f:
        json.dump(processed_responses, f, indent=4)

def _generate_fork_response(prompt, conv):
    k = ' '.join(conv['keywords'])
    conv.pop('keywords')
    
    r0 = random.randint(1, 5)
    r1 = random.randint(6, 10)
    search_result_0 = get_item(k, f'{r0}-{r0}').replace('\n', ' ')
    search_result_1 = get_item(k, f'{r1}-{r1}').replace('\n', ' ')

    search =  {
        "child_0_search_result": search_result_0,
        "child_1_search_result": search_result_1
    }

    _prompt = prompt.replace("{user_request}", json.dumps({**conv, **search}))
    response = llm_request(_prompt)
    js = json.loads(response.lstrip('```json').rstrip('```'))
    print('-------------------')
    print(js['response'])
    return js['response']

def generate_fork_responses(depth: int = 0):
    with open('./fork_response_prompt.txt') as f:
        prompt = f.read()

    if depth == 0:
        with open(FILE_FOLDER + '/main_responses.json') as f:
            main_responses = json.load(f)
    else:
        with open(f'{FILE_FOLDER}fork_responses_{depth}.json') as f:
            main_responses = json.load(f)

    with open(f'{FILE_FOLDER}fork_responses_{depth+1}.json', 'r') as f:
        fork_response = json.loads(f.read())
    
    for mr in tqdm(main_responses):
        print(">>> Generating fork responses for: ", mr['conversations'][0])
        user_input = mr['conversations'][0]
        s: str = mr['conversations'][1]['value']

        if depth > 0:
            f1 = s.find('<|fork_init')
            f2 = s.find('<|fork_start|>', f1+1)
            s = s[:f1] + s[f2 + 14:]

        f1 = s.find('<|fork|>', 1)
        f2 = s.find('<|fork|>', f1+1)

        child1 = s[:f1] + '<|fork_start|>\n'
        
        fi1 = s.find('<|fork_init', 1)
        fi2 = s.find('<|fork_init', fi1+1)
        child2 = s[:fi1] + s[fi2:f2] + '<|fork_start|>\n'

        # print(child1)
        # print(child2)
        # return

        keywords = mr['keywords']

        conv1 = copy.deepcopy(mr)
        conv1['conversations'][1]['value'] = child1
        rp = _generate_fork_response(prompt, conv1)
        conv1['conversations'][1]['value'] = child1 + rp
        conv1['keywords'] = keywords

        conv2 = copy.deepcopy(mr)
        conv2['conversations'][1]['value'] = child2
        rp = _generate_fork_response(prompt, conv2)
        conv2['conversations'][1]['value'] = child2 + rp
        conv2['keywords'] = keywords

        fork_response.append(conv1)
        fork_response.append(conv2)

        with open(f'{FILE_FOLDER}fork_responses_{depth+1}.json', 'w') as f:
            json.dump(fork_response, f, indent=4)

def _generate_leaf_response(prompt, conv, search_range):
    k = ' '.join(conv['keywords'])
    conv.pop('keywords')

    search = {
        "observation": get_item(k, search_range).replace('\n', ' ')
    }
    _prompt = prompt.replace("{user_request}", json.dumps({**conv, **search}))
    response = llm_request(_prompt)
    js = json.loads(response.lstrip('```json').rstrip('```'))
    print('-------------------')
    print(js)
    return js

def generate_fork_leaf_responses(depth: int = 2):
    with open('./fork_leaf_prompt.txt') as f:
        prompt = f.read()

    with open(f'{FILE_FOLDER}fork_responses_{depth}.json') as f:
        fork_responses = json.load(f)

    with open(f'{FILE_FOLDER}fork_leaf_responses.json', 'r') as f:
        fork_leaf_response = json.loads(f.read())

    for fr in tqdm(fork_responses):
        print(">>> Generating fork leaf responses for: ", fr['conversations'][0])
        user_input = fr['conversations'][0]
        s: str = fr['conversations'][1]['value']

        ff1 = s.find('<|fork_init')
        ff2 = s.find('<|fork_start|>', ff1+1)
        s = s[:ff1] + s[ff2 + 14:]

        f1 = s.find('<|fork|>', 1)
        f2 = s.find('<|fork|>', f1+1)

        child1 = s[:f1] + '<|fork_start|>\n'
        range1 = re.findall(r"\d+-\d+", child1[child1.find('<|fork_init'):])[0]

        fi1 = s.find('<|fork_init', 1)
        fi2 = s.find('<|fork_init', fi1+1)
        child2 = s[:fi1] + s[fi2:f2] + '<|fork_start|>\n'
        range2 = re.findall(r"\d+-\d+", child2[child2.find('<|fork_init'):])[0]

        # print(child1)
        # print(child2)
        # return

        keywords = fr['keywords']

        conv1 = copy.deepcopy(fr)
        conv1['conversations'][1]['value'] = child1
        rp = _generate_leaf_response(prompt, conv1, range1)
        conv1['conversations'][1]['value'] = child1 + rp[0]['value']
        conv1['conversations'] = conv1['conversations'] + rp[1:]
        # conv1['keywords'] = keywords
        conv1['tools'] = '[{\"name\": \"search_shop\", \"description\": \"Search for a range of items in shop database\", \"parameters\": {\"type\": \"object\", \"properties\": {\"keywords\": {\"type\": \"string\", \"description\": \"Item to search for\"}, \"range\": {\"type\": \"string\", \"description\": \"Range of results to return\"}}, \"required\": [\"keywords\", \"range\"]}}]'

        conv2 = copy.deepcopy(fr)
        conv2['conversations'][1]['value'] = child2
        rp = _generate_leaf_response(prompt, conv2, range2)
        conv2['conversations'][1]['value'] = child2 + rp[0]['value']
        conv2['conversations'] = conv2['conversations'] + rp[1:]
        # conv2['keywords'] = keywords
        conv2['tools'] = '[{\"name\": \"search_shop\", \"description\": \"Search for a range of items in shop database\", \"parameters\": {\"type\": \"object\", \"properties\": {\"keywords\": {\"type\": \"string\", \"description\": \"Item to search for\"}, \"range\": {\"type\": \"string\", \"description\": \"Range of results to return\"}}, \"required\": [\"keywords\", \"range\"]}}]'

        fork_leaf_response.append(conv1)
        fork_leaf_response.append(conv2)

        with open(f'{FILE_FOLDER}fork_leaf_responses.json', 'w') as f:
            json.dump(fork_leaf_response, f, indent=4)

if __name__ == '__main__':
    # ret = llm_request("I'm looking for x-large, red color women faux fur lined winter warm jacket coat, and price lower than 80.00 dollars. Please check the first 10 search results in parallel. Based on user's request, I'll use jacket coat as keywords to search for the products. I'll split the search into parallel tasks to speed up processing. <|fork_init, child_0|> Child_0 will analyze results 1-5. <|fork|>\n<|fork_init, child_1|> Child_1 will analyze results 6-10. <|fork|>\nGathering results from child_0...\n<|wait, child_0|>According to my analysis, the item B09KP78G37: \"Women Faux Fur Lined Jacket Coat Winter Warm Thick Fleece Outwear Trench Zipper Plus Size Long Sleeve Plush Overcoat\" priced at $47.41 to $59.07 is the best match.<|wait_end|>\nGathering results from child_1...<|wait, child_1|>According to my analysis, the item B07ZXBGDXF: \"Women's Coat, FORUU Winter Faux Fur Fleece Outwear Warm Lapel Biker Motor Aviator Jacket\" priced at $21.49 to $24.99 is the best match.<|wait_end|>\nNow I'll compare the two best matches to provide a recommendation.")
    # print(repr(ret))
    # get_item('I need a stainless steel double-wall insulated water bottle, 32 ounces, and under 25 dollars. Please check the first 20 search results in parallel.', '1-10')
    
    ### Step 1: Generate user requests
    # generate_user_requests(100)

    ### Step 2: Generate main responses
    # generate_main_responses()
    # process_main_responses()

    ### Step 3: Generate fork responses
    # generate_fork_responses(0)
    # generate_fork_responses(1)

    ### Step 4: Generate fork leaf responses
    generate_fork_leaf_responses()
    # print(repr(get_item('jacket coat', f'{11}-{15}')))
#     print(repr(llm_request("""I'm looking for x-large, red color women faux fur lined winter warm jacket coat, and price lower than 80.00 dollars. Can you analyze the following options step by step and pick the best one for me?
# ```
# [button] B09QQP3356 [button_]\nHAUKLIE Men\'s Sports Waffle Ribbed Polo Shirts Summer Short Sleeve Cotton Muscle Quarter-Zip Henley T-Shirt Tunics Tops\n$10.99\n\n[button] B00ZDEDVBI [button_]\nStar Wars I Am Furry Chewbacca Womens Costume Zip-Up Jacket\n$69.99\n\n[button] B07V3WXX85 [button_]\nPUMA Men\'s Amplified Hooded Fleece Jacket\n$55.0\n\n[button] B09QYYN4S4 [button_]\nAUTOKOLA Portable & Practical Hooks, Vintage Style Cast Iron Wall-Mounted Coat Rack Coat Hooks Hat Hooks Hall Tree, 3 3/4", Coffee GG002\n$14.99\n\n[button] B08DXL22JN [button_]\nCicy Bell Womens Casual Blazers Open Front Long Sleeve Work Office Jackets Blazer\n$48.99```""")))
