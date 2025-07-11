import json
import random
import datetime
import llama_cpp
import http.server
import socketserver


DEFAULT_MODEL = 'gemma3'
MODELS_CONFIG = [
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


class VKLlamaRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/generate':
            self.handle_generate()
        else:
            self.send_error(404, "Not Found")

    def handle_generate(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_payload = json.loads(post_data.decode('utf-8'))

            # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
            model_name = request_payload.get('model', DEFAULT_MODEL)
            prompt = request_payload.get('prompt')
            stream = request_payload.get('stream', False)
    
            options = request_payload.get('options', {}) 
            max_tokens = options.get('num_predict', 128)
            temperature = options.get('temperature', 0.8)
            seed = options.get('seed', random.randint(0, 2**32 - 1))

            if not prompt:
                self.send_error(400, "Bad Request", "Missing 'prompt' in request body.")
                return

            # find model
            model_config = next((e for e in MODELS_CONFIG if e['name'] == model_name), None)

            if not model_config:
                self.send_error(404, "Not Found", f"Model '{model_name}' not found.")
                return

            model_path = model_config['path']

            # init llm
            llm = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                # n_ctx=2048
                seed=seed,
                verbose=False
            )

            # generate
            if stream:
                self.send_response(200)
                self.send_header('Content-type', 'application/x-ndjson') # Ollama uses ndjson for streaming
                self.end_headers()

                out = llm.create_chat_completion(
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )

                for chunk in out:
                    # streaming
                    msg = chunk['choices'][0]['delta'].get('content', '')
                    
                    ollama_chunk = {
                        "model": model_name,
                        "created_at": datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                        "response": msg,
                        "done": False
                    }

                    # last chunk
                    if chunk['choices'][0].get('finish_reason') is not None:
                         ollama_chunk['done'] = True
                         ollama_chunk['total_duration'] = 0 # dumb
                         ollama_chunk['load_duration'] = 0 # dumb
                         ollama_chunk['prompt_eval_count'] = 0 # dumb
                         ollama_chunk['eval_count'] = 0 # dumb

                    self.wfile.write(json.dumps(ollama_chunk).encode('utf-8') + b'\n')
                    # wfile.flush()
                self.wfile.flush() # send last chunk

            else:
                out = llm.create_chat_completion(
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )

                response_content = out['choices'][0]['message']['content']

                ollama_response = {
                    "model": model_name,
                    "created_at": datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                    "response": response_content,
                    "done": True,
                    "total_duration": 0, # dumb
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                    "eval_count": 0
                }

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(ollama_response).encode('utf-8'))

        except json.JSONDecodeError:
            self.send_error(400, "Bad Request", "Invalid JSON payload.")
        except KeyError as e:
            self.send_error(400, "Bad Request", f"Missing key in request: {e}")
        except Exception as e:
            print(f"Error handling /api/generate: {e}")
            self.send_error(500, "Internal Server Error", f"An error occurred: {e}")

    def log_message(self, format, *args):
        # print(f"[{self.log_date_time_string()}] {self.address_string()} - {format % args}")
        pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def serve(args):
    server_address = (args.host, args.port)
    httpd = ThreadedHTTPServer(server_address, VKLlamaRequestHandler)
    print(f'Starting vkllama server on http://{args.host}:{args.port}')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nServer is shutting down.')
        httpd.shutdown()
        httpd.server_close()
