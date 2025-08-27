import json
import requests


DEFAULT_MODEL = 'gemma3'
VKLLAMA_GENERATE_URL = 'http://{address}/api/generate'
VKLLAMA_CHAT_URL = 'http://{address}/api/chat'


def chat(model, system, address, seed):
    messages = []

    if system:
        messages.append({'role': 'system', 'content': system})

    while True:
        # input prompt
        try:
            prompt = input('> ')
        except EOFError as _:
            print()
            return

        # append message
        messages.append({'role': 'user', 'content': prompt})

        # generate
        payload = {
            'model': model,
            'seed': seed,
            'stream': True,
            'messages': messages
        }

        try:
            response = requests.post(VKLLAMA_CHAT_URL.format(address=address), json=payload, stream=True)
            response.raise_for_status()

            answer = ''
            for chunk in response.iter_lines():
                msg = json.loads(chunk)
                answer += msg['message']['content']
                print(msg['message']['content'], end='', flush=True)
            print()

            # append answer
            messages.append({'role': 'assistant', 'content': answer.strip()})
        except Exception as e:
            print(f'An unexpected error occurred: {e}')
            return


def generate(prompt, system, model, address, seed, stream):
    payload = {
        'model': model,
        'seed': seed,
        'stream': stream,
        'prompt': prompt.strip()
    }

    # use system prompt
    if system:
        payload['system'] = system.strip()

    # generate
    try:
        response = requests.post(VKLLAMA_GENERATE_URL.format(address=address), json=payload, stream=stream)
        response.raise_for_status()

        for chunk in response.iter_lines():
            msg = json.loads(chunk)
            print(msg['response'], end='', flush=True)
        print()
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


def run(args):
    prompt = ' '.join(args.prompt)

    if prompt:
        generate(prompt, args.sys, args.model, args.address, args.seed, args.stream)
    else:
        chat(args.model, args.sys, args.address, args.seed)
