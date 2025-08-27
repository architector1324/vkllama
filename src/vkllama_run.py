import json
import requests
import datetime


DEFAULT_MODEL = 'gemma3'
VKLLAMA_GENERATE_URL = 'http://{address}/api/generate'
VKLLAMA_CHAT_URL = 'http://{address}/api/chat'

COMMANDS = [
    {
        'cmd': '/clear',
        'help': 'clear char context'
    },
    {
        'cmd': '/exit (/bye, /quit)',
        'help': 'exit'
    },
    {
        'cmd': '/help (/?)',
        'help': 'print help'
    },
    {
        'cmd': '/sys',
        'help': 'set system prompt'
    },
    {
        'cmd': '/ctx',
        'help': 'get/set context window'
    },
    {
        'cmd': '/save',
        'help': 'save chat to file (leave empty for automatic filename)'
    },
    {
        'cmd': '/load',
        'help': 'load chat from json file'
    },
    {
        'cmd': '/continue',
        'help': 'continue model answer'
    },
    {
        'cmd': '/hack',
        'help': 'start answer as model and continue'
    },
    {
        'cmd': '/json',
        'help': 'show full chat in json (use `/json pretty` for ident)'
    }
]


def chat(model, system, address, seed):
    messages = []
    ctx = 2048

    if system:
        messages.append({'role': 'system', 'content': system})

    while True:
        model_turn = False

        # input prompt
        try:
            prompt = input('> ').strip()
        except EOFError as _:
            print()
            return
        except KeyboardInterrupt as _:
            print()
            continue

        # commands
        if prompt == '/clear':
            messages = [{'role': 'system', 'content': system}] if system else []
            print('Chat cleared.')
            continue
        elif prompt in ('/exit', '/bye', '/quit'):
            return
        elif prompt in ('/help', '/?'):
            print('Commands:')
            for cmd in COMMANDS:
                print(f'  {cmd["cmd"]}: {cmd["help"]}')
            continue
        elif prompt.startswith('/sys'):
            parts = prompt.split(' ')
            if len(parts) < 2:
                print('Please, provide a system prompt!')
                continue

            system = ' '.join(parts[1:]).strip()

            if len(messages) > 0 and messages[0]['role'] == 'system':
                messages[0]['content'] = system
            else:
                messages = [{'role': 'system', 'content': system}]
            print('System prompt set.')
            continue
        elif prompt.startswith('/ctx'):
            parts = prompt.split(' ')
            if len(parts) == 1:
                print(ctx)
                continue
            
            try:
                ctx = int(parts[1])
                print(f'Context set: {ctx}')
                continue
            except ValueError as e:
                print(f'An unexpected error occurred: {e}')
                continue
        elif prompt.startswith('/save'):
            parts = prompt.split(' ')
            filename = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json' if len(parts) == 1 else parts[1].strip()

            with open(filename, 'w') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
                print(f'Chat saved: "{filename}".')
            continue
        elif prompt.startswith('/load'):
            parts = prompt.split(' ')
            if len(parts) != 2:
                print('Please, provide filename!')
                continue

            filename = parts[1].strip()

            try:
                with open(filename, 'r') as f:
                    messages = json.load(f)

                    print(f'Chat loaded: "{filename}".')

                    for msg in messages:
                        if msg['role'] == 'user':
                            print(f'> {msg["content"].strip()}')
                        elif msg['role'] == 'assistant':
                            print(f'{msg["content"].strip()}\n\n')
                continue
            except json.JSONDecodeError as e:
                print(f'An unexpected error occurred: {e}')
        elif prompt.startswith('/continue'):
            model_turn = True
        elif prompt.startswith('/hack'):
            parts = prompt.split(' ')
            prompt = ' '.join(parts[1:]).strip()
            messages.append({'role': 'user', 'content': prompt})

            # model prompt
            try:
                model_prompt = input(f'[{model}] > ').strip()
                messages.append({'role': 'assistant', 'content': model_prompt})
            except EOFError as _:
                print()
                return
            except KeyboardInterrupt as _:
                print()
                continue
            model_turn = True
        elif prompt.startswith('/json'):
            parts = prompt.split(' ')
            indent = 2 if len(parts) > 1 and parts[1].startswith('pretty') else None
            chat = json.dumps(messages, indent=indent, ensure_ascii=False)
            print(chat)
            continue

        # append message
        if not model_turn:
            messages.append({'role': 'user', 'content': prompt})

        # generate
        payload = {
            'model': model,
            'seed': seed,
            'stream': True,
            'messages': messages,
            'options': {'num_ctx': ctx}
        }

        answer = ''

        try:
            response = requests.post(VKLLAMA_CHAT_URL.format(address=address), json=payload, stream=True)
            response.raise_for_status()

            for chunk in response.iter_lines():
                msg = json.loads(chunk)
                answer += msg['message']['content']
                print(msg['message']['content'], end='', flush=True)
            print('\n')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')
            return
        except KeyboardInterrupt as _:
            print()
            
        # append answer
        if messages[-1]['role'] == 'user':
                messages.append({'role': 'assistant', 'content': answer.strip()})
        else:
            messages[1]['content'] += answer.strip()


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
