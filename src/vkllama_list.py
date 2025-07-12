import json
import requests

VKLLAMA_MODELS_URL = 'http://{address}/api/tags'


def list_models(args):
    try:
        response = requests.get(VKLLAMA_MODELS_URL.format(address=args.address))
        response.raise_for_status()

        models = response.json()

        print('NAME\tID\tSIZE\tMODIFIED')
        for model in models['models']:
            print(f'{model['name']}\t{model['digest']}\t{model['size'] / (1024 * 1024 * 1024)} GB\t{model['modified_at']}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
