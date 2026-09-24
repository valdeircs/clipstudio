"""AI duration is source-bounded; manual exports share the same source bounds."""
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import engine
spec = importlib.util.spec_from_file_location('duration_server', APP / 'server.py')
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)
PID = 'a' * 32


class DurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / PID
        self.folder.mkdir()
        self.paths = patch.object(srv, 'PROJECTS_DIR', Path(self.temp.name))
        self.paths.start()
        self.project = {'id': PID, 'status': 'ready', 'duration': 500, 'source': 'source.mp4',
                        'segments': [{'start': 0, 'end': 23, 'text': 'A complete idea.'}],
                        'clips': [{'id': 'clip-1', 'start': 0, 'end': 120, 'title': 'Complete message'}]}
        srv.atomic_json(self.folder / 'project.json', self.project)
        (self.folder / 'source.mp4').write_bytes(b'synthetic-source')
        self.jobs = []
        self.worker = patch.object(srv, 'WORKER', types.SimpleNamespace(submit=lambda *args: self.jobs.append(args)))
        self.worker.start()
        self.handler = object.__new__(srv.Handler)
        self.handler.send_json = lambda *args: None

    def tearDown(self):
        self.worker.stop()
        self.paths.stop()
        self.temp.cleanup()

    def test_ai_settings_ignore_legacy_duration_target(self):
        for length in (5, 45, 180, 'AI decides'):
            with self.subTest(length=length):
                settings = srv.analysis_settings({'length': length}, ai=True)
                self.assertEqual(settings['duration_mode'], 'ai')
                self.assertNotIn('length', settings)
        with self.assertRaises(srv.APIError):
            srv.analysis_settings({'length': 180})

    def test_queue_accepts_short_and_long_messages(self):
        for end in (23, 120, 420, 500):
            with self.subTest(end=end):
                srv.atomic_json(self.folder / 'project.json', self.project)
                self.handler.queue_render(PID, ['clip-1'], {'start': 0, 'end': end})
                saved = srv.read_project(PID)
                self.assertEqual(saved['clips'][0]['duration'], end)
                self.assertEqual(saved['status'], 'queued')

    def test_queue_rejects_ranges_outside_source(self):
        for start, end in ((0, .5), (-1, 120), (120, 100), (0, 500.01), (0, float('inf'))):
            with self.subTest(start=start, end=end), self.assertRaises(srv.APIError):
                self.handler.queue_render(PID, ['clip-1'], {'start': start, 'end': end})
        self.assertFalse(self.jobs)
        self.assertEqual(srv.read_project(PID)['status'], 'ready')

    def test_reselection_accepts_a_complete_23_second_source(self):
        self.project['duration'] = 23
        srv.atomic_json(self.folder / 'project.json', self.project)
        config = {'pipeline': 'local_ai', 'ai_provider': 'gemini', 'ai_api_key': 'synthetic-secret'}
        with patch.object(srv, 'connections', return_value=types.SimpleNamespace(load=lambda **kw: config)):
            self.handler.reselect_ai(PID, {'count': 1})
        self.assertEqual(len(self.jobs), 1)

    def test_long_render_uses_full_selection_not_90_seconds(self):
        commands = []
        def run(args, folder, **kwargs):
            commands.append(args)
            Path(args[-1]).write_bytes(b'synthetic-encoded-video')
            return ''
        def thumbnail(*args):
            Path(args[10]).write_bytes(b'synthetic-cover')
        with patch.object(engine, '_run', side_effect=run), \
             patch.object(engine, '_require', side_effect=lambda name: name), \
             patch.object(engine, '_thumbnail', side_effect=thumbnail), \
             patch.object(engine, '_probe', return_value={}):
            result = engine.render_clip(self.folder, self.project, self.project['clips'][0],
                                        {'captions': False, 'zoom_mode': 'wide'}, lambda *args: None)
        self.assertEqual(result['end'], 120)
        self.assertEqual(commands[0][commands[0].index('-t') + 1], '120.000')
        self.assertTrue((self.folder / result['video']).is_file())


if __name__ == '__main__':
    unittest.main()
