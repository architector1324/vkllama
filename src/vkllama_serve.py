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
                digest = model_info.get('digest', None)
                qlevel = model_info.get('quantization_level', None)
                psize = model_info.get('parameter_size', None)

                if os.path.exists(full_model_path):
                    size = os.path.getsize(full_model_path)
                    modified_at = datetime.datetime.fromtimestamp(os.path.getmtime(full_model_path)).isoformat(timespec='milliseconds') + 'Z'

                    if not digest:
                        digest = calculate_file_sha256(full_model_path)

                models.append({
                    'name': model_name,
                    'model': model_name,
                    'modified_at': modified_at,
                    'size': size,
                    'digest': digest,
                    'details': {
                        'parent_model': '',
                        'format': 'gguf',
                        'family': model_name,
                        'families': [model_name],
                        'quantization_level': qlevel,
                        'parameter_size': psize
                    }
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
        elif self.path == '/api/chat':
            self.handle_chat_completion()
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
            system_prompt = request_payload.get('system', None)
            stream = request_payload.get('stream', True)

            # options
            options = request_payload.get('options', {})

            n_ctx = options.get('num_ctx', 2048)
            max_tokens = options.get('num_predict', 2048)
            temperature = options.get('temperature', 0.8)
            top_p = options.get('top_p', 0.9)
            top_k = options.get('top_k', 40)
            frequency_penalty = options.get('frequency_penalty', 0.0)
            presence_penalty = options.get('presence_penalty', 0.0)
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

            # create messages
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})

            # generate
            if stream:
                self.send_response(200)
                self.send_header('Content-type', 'application/x-ndjson') # Ollama uses ndjson for streaming
                self.end_headers()

                out = llm.create_chat_completion(
                    messages=messages,
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
                    messages=messages,
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

    # handle_chat_completion method
    def handle_chat_completion(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_payload = json.loads(post_data.decode('utf-8'))

            model_name = request_payload.get('model', DEFAULT_MODEL)
            messages = request_payload.get('messages')
            stream = request_payload.get('stream', True) # Ollama's default for chat API is stream=True

            if not messages:
                self.send_error(400, 'Bad Request', 'Missing "messages" in request body.')
                return
            if not isinstance(messages, list) or not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
                self.send_error(400, 'Bad Request', 'Invalid "messages" format. Must be a list of objects with "role" and "content".')
                return

            options = request_payload.get('options', {})
            n_ctx = options.get('num_ctx', 2048)
            max_tokens = options.get('num_predict', 2048)
            temperature = options.get('temperature', 0.8)
            top_p = options.get('top_p', 0.9)
            top_k = options.get('top_k', 40)
            frequency_penalty = options.get('frequency_penalty', 0.0)
            presence_penalty = options.get('presence_penalty', 0.0)
            seed = options.get('seed', random.randint(0, 2**32 - 1))

            model_info = next((e for e in get_models() if e['name'] == model_name), None)
            if not model_info:
                self.send_error(404, 'Not Found', f'Model "{model_name}" not found.')
                return

            expanded_models_path = os.path.expanduser(models_path)
            model_path = os.path.join(expanded_models_path, model_info['filename'])

            llm = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=n_ctx,
                seed=seed,
                verbose=False
            )

            if stream:
                self.send_response(200)
                self.send_header('Content-type', 'application/x-ndjson')
                self.end_headers()

                response_generator = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    temperature=temperature,
                    stream=True,
                )

                final_finish_reason = None

                for chunk in response_generator:
                    delta = chunk['choices'][0]['delta']
                    message_content = delta.get('content', '')
                    current_finish_reason = chunk['choices'][0].get('finish_reason')

                    ollama_chunk = {
                        'model': model_name,
                        'created_at': datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                        'message': {
                            'role': 'assistant',
                            'content': message_content
                        },
                        'done': False
                    }
                    
                    self.wfile.write(json.dumps(ollama_chunk).encode('utf-8') + b'\n')
                    self.wfile.flush()

                    if current_finish_reason:
                        # Store the reason for the final chunk
                        final_finish_reason = current_finish_reason

                # Construct the final 'done: true' chunk with metrics.
                final_ollama_chunk = {
                    'model': model_name,
                    'created_at': datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                    'message': {
                        'role': 'assistant',
                        'content': '' # As per Ollama example, content is empty in final metrics chunk
                    },
                    'done': True,
                    'done_reason': final_finish_reason if final_finish_reason else 'stop', # Default to 'stop' if no specific reason
                    'total_duration': 0, # Dummy
                    'load_duration': 0,  # Dummy
                    'prompt_eval_count': 0, # Dummy
                    'prompt_eval_duration': 0, # Dummy
                    'eval_count': 0, # Dummy
                    'eval_duration': 0 # Dummy
                }
                self.wfile.write(json.dumps(final_ollama_chunk).encode('utf-8') + b'\n')
                self.wfile.flush()

            else: # Not streaming
                full_completion = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    temperature=temperature,
                    stream=False,
                )

                response_message = full_completion['choices'][0]['message']
                usage = full_completion['usage']
                finish_reason = full_completion['choices'][0].get('finish_reason', 'stop')

                ollama_response = {
                    'model': model_name,
                    'created_at': datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                    'message': {
                        'role': response_message['role'],
                        'content': response_message['content']
                    },
                    'done': True,
                    'done_reason': finish_reason,
                    'total_duration': 0, # Dummy
                    'load_duration': 0,  # Dummy
                    'prompt_eval_count': usage.get('prompt_tokens', 0),
                    'prompt_eval_duration': 0, # Dummy
                    'eval_count': usage.get('completion_tokens', 0),
                    'eval_duration': 0 # Dummy
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
            print(f'Error handling /api/chat: {e}')
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
