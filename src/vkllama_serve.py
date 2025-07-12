import os
import json
import random
import hashlib
import datetime
import llama_cpp
import http.server
import socketserver


DEFAULT_MODEL = 'gemma3'
DEFAULT_MODELS_PATH = '~/.vkllama/models'

models_path = DEFAULT_MODELS_PATH


def get_models():
    expanded_models_path = os.path.expanduser(models_path)
    os.makedirs(expanded_models_path, exist_ok=True)

    # read models
    with open(f'{expanded_models_path}/models.json', 'r') as f:
        models_config = json.load(f)
        return models_config
    return None


def calculate_file_sha256(filepath):
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            # Read and update hash string in chunks
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f'Warning: Could not calculate SHA256 for {filepath}: {e}')
        return 'sha256:error_calculating_digest'


class VKLlamaRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/tags':
            self.handle_list_models()
        else:
            self.send_error(404, 'Not Found')

    def handle_list_models(self):
        try:
            expanded_models_path = os.path.expanduser(models_path)
            models = []
            for model_info in get_models():
                model_name = model_info['name']
                full_model_path = os.path.join(expanded_models_path, model_info['filename'])

                # get info
                size = 0
                modified_at = datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z' 
                digest = model_info['digest'] if 'digest' in model_info else None

                if os.path.exists(full_model_path):
                    size = os.path.getsize(full_model_path)
                    modified_at = datetime.datetime.fromtimestamp(os.path.getmtime(full_model_path)).isoformat(timespec='milliseconds') + 'Z'

                    if not digest:
                        digest = calculate_file_sha256(full_model_path)

                models.append({
                    'name': f'{model_name}:latest',
                    'modified_at': modified_at,
                    'size': size,
                    'digest': digest
                })

            response_payload = {'models': models}

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        except FileNotFoundError:
            self.send_error(500, 'Internal Server Error', 'models.json not found in the models directory.')
        except json.JSONDecodeError:
            self.send_error(500, 'Internal Server Error', 'Error parsing models.json. Check file format.')
        except Exception as e:
            print(f'Error handling /api/tags: {e}')
            self.send_error(500, 'Internal Server Error', f'An unexpected error occurred: {e}')

    def do_POST(self):
        if self.path == '/api/generate':
            self.handle_generate()
        else:
            self.send_error(404, 'Not Found')

    def handle_generate(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_payload = json.loads(post_data.decode('utf-8'))

            # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
            model_name = request_payload.get('model', DEFAULT_MODEL)
            prompt = request_payload.get('prompt')
            stream = request_payload.get('stream', True)

            # options
            options = request_payload.get('options', {})

            n_ctx = options.get('num_ctx', 2048)
            max_tokens = options.get('num_predict', 128)
            temperature = options.get('temperature', 0.8)
            top_p = options.get('top_p', 0.9)
            top_k = options.get('top_k', 40)
            frequency_penalty = options.get('frequency_penalty', 0.5)
            presence_penalty = options.get('presence_penalty', 0.5)
            seed = options.get('seed', random.randint(0, 2**32 - 1))

            if not prompt:
                self.send_error(400, 'Bad Request', 'Missing "prompt" in request body.')
                return

            # find model
            model_info = next((e for e in get_models() if e['name'] == model_name), None)

            if not model_info:
                self.send_error(404, 'Not Found', f'Model "{model_name}" not found.')
                return

            expanded_models_path = os.path.expanduser(models_path)
            model_path = os.path.join(expanded_models_path, model_info['filename'])

            # init llm
            llm = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=n_ctx,
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
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    temperature=temperature,
                    stream=True,
                )

                for chunk in out:
                    # streaming
                    msg = chunk['choices'][0]['delta'].get('content', '')
                    
                    ollama_chunk = {
                        'model': model_name,
                        'created_at': datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                        'response': msg,
                        'done': False
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
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    temperature=temperature,
                    stream=False,
                )

                response_content = out['choices'][0]['message']['content']

                ollama_response = {
                    'model': model_name,
                    'created_at': datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                    'response': response_content,
                    'done': True,
                    'total_duration': 0, # dumb
                    'load_duration': 0,
                    'prompt_eval_count': 0,
                    'eval_count': 0
                }

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(ollama_response).encode('utf-8'))

        except json.JSONDecodeError:
            self.send_error(400, 'Bad Request', 'Invalid JSON payload.')
        except KeyError as e:
            self.send_error(400, 'Bad Request', f'Missing key in request: {e}')
        except Exception as e:
            print(f'Error handling /api/generate: {e}')
            self.send_error(500, 'Internal Server Error', f'An error occurred: {e}')

    def log_message(self, format, *args):
        # print(f'[{self.log_date_time_string()}] {self.address_string()} - {format % args}')
        pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def serve(args):
    global models_path
    models_path = args.models

    server_address = (args.host, args.port)
    httpd = ThreadedHTTPServer(server_address, VKLlamaRequestHandler)
    print(f'Starting vkllama server on http://{args.host}:{args.port}')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nServer is shutting down.')
        httpd.shutdown()
        httpd.server_close()
