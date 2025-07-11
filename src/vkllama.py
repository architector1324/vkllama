import random
import argparse

import vkllama_run
import vkllama_serve


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vulkan LLM tool', add_help=False)
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    parser.add_argument('--help', action='help')

    # run
    run_parser = subparsers.add_parser('run', help='Run LLM model', add_help=False)
    run_parser.add_argument('-m', '--model', type=str, default=vkllama_run.DEFAULT_MODEL, help='Model')
    run_parser.add_argument('--seed', default=random.randint(0, 2**32 - 1), type=int, help='Specify a numerical seed for reproducible text generation. If not provided, a random seed will be used.')
    run_parser.add_argument('-s', '--stream', action='store_true', help='Stream output')
    run_parser.add_argument('-t', '--think', action='store_true', help='Enable advanced, iterative reasoning for the model to refine outputs. May increase processing time and token usage.')
    run_parser.add_argument('-a', '--address', default='0.0.0.0:11435', type=str, help='Server host address')
    run_parser.add_argument('prompt', nargs='+', type=str, help='Prompt for model')
    run_parser.add_argument('--help', action='help')

    # serve
    serve_parser = subparsers.add_parser('serve', help='Start LLM server', add_help=False)
    serve_parser.add_argument('--host', default='0.0.0.0', type=str, help='Server host address')
    # serve_parser.add_argument('-m', '--models', default=imagine_server_defs.DEFAULT_MODELS_PATH, type=str, help='SD models path')
    serve_parser.add_argument('-p', '--port', default=11435, type=int, help='Server port')
    # serve_parser.add_argument('-d', '--device', default=imagine_server_defs.DEFAULT_DEVICE, type=str,  choices=['cpu', 'cuda', 'mps'], help='Model compute device')
    serve_parser.add_argument('--help', action='help')

    args = parser.parse_args()

    if args.command == 'run':
        vkllama_run.run(args)
    elif args.command == 'serve':
        vkllama_serve.serve(args)
    else:
        parser.print_help()
