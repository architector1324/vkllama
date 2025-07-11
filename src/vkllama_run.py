import json
import requests


DEFAULT_MODEL = 'gemma3'
VKLLAMA_GENERATE_URL = 'http://{address}/api/generate'


def run(args):
    prompt = ' '.join(args.prompt)
    payload = {
        'model': args.model,
        'seed': args.seed,
        'stream': args.stream,
        'prompt': prompt
    }

    try:
        response = requests.post(VKLLAMA_GENERATE_URL.format(address=args.address), json=payload, stream=args.stream)
        response.raise_for_status()

        for chunk in response.iter_lines():
            msg = json.loads(chunk)
            print(msg['response'], end='', flush=True)
        print()
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
