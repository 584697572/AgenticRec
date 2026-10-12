"""Persistent keyless observer around the unchanged Microsoft original App.

Accept public episode messages only over stdin; evaluation labels remain in the
supervisor. SDK traffic is relayed, never sent by this resource worker.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'reproduction/.runtime/InteRecAgent-compat'


def main():
    wire_in, wire_out = sys.stdin, sys.stdout

    def emit(value):
        wire_out.write(json.dumps(value, ensure_ascii=True) + '\n')
        wire_out.flush()

    if os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('Real credential must not enter original worker')
    captured = io.StringIO()
    clients, maps = [], []
    active = {}
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        import gradio as gr
        import httpx
        import numpy as np
        import torch
        from openai import OpenAI
        sys.path.insert(0, str(SOURCE))
        os.chdir(SOURCE)
        from llm4crs.utils import open_ai as upstream
        from llm4crs.utils import extract_integers_from_string
        from llm4crs.mapper import MapTool
        torch.set_num_threads(2)

        def seed(value):
            random.seed(value)
            np.random.seed(value)
            torch.manual_seed(value)

        class Relay(httpx.BaseTransport):
            def handle_request(self, request):
                if str(request.url) != 'https://api.deepseek.com/chat/completions' or request.method != 'POST':
                    raise RuntimeError('Unexpected worker network request')
                emit({'event': 'request', **active, 'body': json.loads(request.content)})
                packet = json.loads(wire_in.readline())
                if 'transport_error' in packet:
                    raise httpx.TransportError(packet['transport_error'])
                return httpx.Response(packet['status_code'], content=packet['content'].encode('utf-8'),
                                      headers={'content-type': 'application/json'})

        def client_factory(**kwargs):
            client = OpenAI(**kwargs, max_retries=0,
                            http_client=httpx.Client(transport=Relay(), follow_redirects=False))
            create = client.chat.completions.create

            def bounded_create(**kw):
                kw['max_tokens'] = min(kw.get('max_tokens', 512), 512)
                kw['temperature'] = 0
                return create(**kw, extra_body={'thinking': {'type': 'disabled'}})

            client.chat.completions.create = bounded_create
            clients.append(client)
            return client

        old_init = upstream.OpenAICall.__init__

        def bounded_init(self, *args, **kwargs):
            kwargs['retry_limits'] = 1
            kwargs['timeout'] = 60
            old_init(self, *args, **kwargs)

        old_map = MapTool.run

        def observe_map(self, inputs):
            before = list(self.buffer.get())
            try:
                count = extract_integers_from_string(inputs)[0]
            except Exception:
                count = 5
            output = old_map(self, inputs)
            maps.append({'ids': [int(i) for i in before[:min(count, self.max_rec_num)]],
                         'output': output})
            return output

        os.environ.update(OPENAI_API_KEY='stdio-relay-placeholder', OPENAI_API_TYPE='open_ai',
                          OPENAI_API_BASE='https://api.deepseek.com')
        sys.argv = ['app.py', '--engine', 'deepseek-flash', '--bot_type', 'chat',
                    '--plan_first', '1', '--langchain', '0', '--demo_mode', 'zero',
                    '--enable_reflection', '0', '--enable_shorten', '0',
                    '--reply_style', 'concise', '--max_output_tokens', '512']
        seed(42)
        with patch.object(upstream, 'OpenAI', client_factory), \
             patch.object(upstream.OpenAICall, '__init__', bounded_init), \
             patch.object(MapTool, 'run', observe_map), \
             patch.object(gr.Blocks, 'launch', lambda *args, **kwargs: None):
            started = time.perf_counter()
            emit({'event': 'progress', 'stage': 'loading_original_app'})
            app = runpy.run_path(str(SOURCE / 'app.py'), run_name='__main__')
            emit({'event': 'ready', 'cold_start_seconds': time.perf_counter()-started,
                  'real_key_in_worker': False, 'tool_names': app['tool_names']})
            state = None
            for line in wire_in:
                command = json.loads(line)
                if command.get('command') == 'stop':
                    break
                if set(command) != {'command', 'episode_id', 'turn', 'message', 'reset', 'seed'}:
                    raise ValueError('Worker accepts public message commands only')
                active.clear()
                active.update(episode_id=command['episode_id'], turn=command['turn'])
                if command['reset']:
                    app['bot'].clear()
                    app['candidate_buffer'].clear()
                    app['candidate_buffer'].clear_tracks()
                    seed(command['seed'])
                    state = [list(app['default_chat_value'])]
                    if app['bot'].memory.memory:
                        raise RuntimeError('Original episode memory did not reset')
                offset = len(maps)
                captured.seek(0)
                captured.truncate(0)
                clock = time.perf_counter()
                result = {'event': 'turn_result', **active, 'status': 'ERROR', 'answer': '',
                          'mapped_catalog_ids': [], 'tool_trace': []}
                try:
                    _, state, _ = app['user'](command['message'], state)
                    _, state = app['bot'].run_gr(state)
                    answer = state[-1][1]
                    result.update(status='OK' if isinstance(answer, str) and not answer.startswith('Something went wrong') else 'ERROR',
                                  answer=answer, mapped_catalog_ids=maps[-1]['ids'] if len(maps)>offset else [],
                                  map_observations=maps[offset:],
                                  tool_trace=app['candidate_buffer'].tracker.copy(),
                                  memory_entries=len(app['bot'].memory.memory))
                except Exception as error:
                    import traceback
                    result.update(error_type=type(error).__name__)
                    captured.write(traceback.format_exc())
                result.update(elapsed_seconds=time.perf_counter()-clock, worker_log=captured.getvalue())
                emit(result)
    for client in clients:
        client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
