import random
import argparse
import vkllama_run


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vulkan LLM tool', add_help=False)
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    parser.add_argument('--help', action='help')

    # run
    run_parser = subparsers.add_parser('run', help='Run a model', add_help=False)
    run_parser.add_argument('-m', '--model', type=str, default=vkllama_run.DEFAULT_MODEL, help='Model')
    run_parser.add_argument('--seed', default=random.randint(0, 2**32 - 1), type=int, help='Specify a numerical seed for reproducible text generation. If not provided, a random seed will be used.')
    run_parser.add_argument('-s', '--stream', default=None, type=int, help='Stream steps to output')
    run_parser.add_argument('-t', '--think', action='store_true', help='Enable advanced, iterative reasoning for the model to refine outputs. May increase processing time and token usage.')
    run_parser.add_argument('prompt', nargs='+', type=str, help='Prompt for model')
    run_parser.add_argument('--help', action='help')

    args = parser.parse_args()

    if args.command == 'run':
        vkllama_run.run(args)
    else:
        parser.print_help()
