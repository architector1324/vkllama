import llama_cpp


DEFAULT_MODEL = 'gemma3'

MODELS = [
    {
        'name': 'gemma3',
        'path': '/home/arch/AI/models/llm/gemma-3-4b-it-Q4_K_M.gguf'
    },
    {
        'name': 'qwen3',
        'path': '/home/arch/AI/models/llm/Qwen3-8B-Q4_K_M.gguf'
    },
    {
        'name': 'qwen3:4b',
        'path': '/home/arch/AI/models/llm/Qwen3-4B-Q4_K_M.gguf'
    }
]

def run(args):
    # get prompt and model
    prompt = ' '.join(args.prompt)
    model = next((e for e in MODELS if e['name'] == args.model), None)

    if not model:
        print(f'error: model "{args.model}" not found!')
        return

    # init
    llm = llama_cpp.Llama(
        model_path=model['path'],
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False
    )

    out = llm.create_chat_completion(
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ],
        max_tokens=16,
        stream=args.stream
    )

    # run
    if args.stream:
        # stream output
        for msg in out:
            if 'content' in msg['choices'][0]['delta']:
                txt = msg['choices'][0]['delta']['content']
                print(txt, end='', flush=True)
        print()
    else:
        print(out['choices'][0]['message']['content'])
