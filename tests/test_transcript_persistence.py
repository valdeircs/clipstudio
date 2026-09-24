"""Extraction remains usable after AI failure; no network or real media."""
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
spec = importlib.util.spec_from_file_location('transcript_server', APP / 'server.py')
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)
PID = 'e' * 32
KEY = 'synthetic-provider-secret'
SEGMENTS = [{'start': 0, 'end': 45, 'text': 'A complete source passage.'}]


class TranscriptPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.paths = patch.object(srv, 'PROJECTS_DIR', Path(self.temp.name))
        self.paths.start()
        self.folder = Path(self.temp.name) / PID
        srv.atomic_json(self.folder / 'project.json', {
            'id': PID, 'title': 'New video', 'status': 'queued', 'clips': [], 'source': 'source.mp4'})

    def tearDown(self):
        self.paths.stop()
        self.temp.cleanup()

    def run_analysis(self, analyze):
        with patch.object(srv, 'engine', return_value=types.SimpleNamespace(analyze_project=analyze)):
            srv.analyze_job(PID, {'pipeline': 'local_ai', '_api_config': {'ai_api_key': KEY}})
        return srv.read_project(PID)

    def test_failed_ranking_keeps_metadata_timing_source_and_redacted_response(self):
        def analyze(folder, request, progress):
            request['_transcript_observer']({
                'title': 'Sermon', 'duration': 60, 'language': 'pt', 'segments': SEGMENTS,
                'source_url': 'https://www.youtube.com/watch?v=abcdefghijk',
                'source': None, 'transcript': 'transcript.txt', 'transcript_srt': 'transcript.srt',
                'ai_api_key': KEY, 'id': 'bad', 'status': 'ready'})
            request['_response_observer']({'selection': {'reason': KEY}, 'ai_api_key': KEY})
            raise RuntimeError('Quote mismatch ' + KEY)
        p = self.run_analysis(analyze)
        self.assertEqual(p['id'], PID)
        self.assertEqual(p['title'], 'Sermon')
        self.assertEqual(p['segments'], SEGMENTS)
        self.assertEqual(p['source'], 'source.mp4')
        self.assertEqual(p['clips'], [])
        self.assertEqual(p['status'], 'error')
        self.assertIn('transcript is saved', p['error'])
        for name in ('project.json', 'ai-selection-response.json', 'diagnostic.log'):
            self.assertNotIn(KEY, (self.folder / name).read_text())

    def test_early_extraction_failure_does_not_claim_transcript_saved(self):
        def analyze(*args):
            raise RuntimeError('Captions unavailable')
        p = self.run_analysis(analyze)
        self.assertEqual(p['error'], 'Captions unavailable')
        self.assertNotIn('segments', p)

    def test_success_still_commits_suggestions(self):
        def analyze(folder, request, progress):
            request['_transcript_observer']({'title': 'Sermon', 'segments': SEGMENTS, 'duration': 60})
            return {'title': 'Sermon', 'segments': SEGMENTS, 'duration': 60,
                    'clips': [{'start': 0, 'end': 45, 'title': 'A complete source passage.'}]}
        p = self.run_analysis(analyze)
        self.assertEqual(p['status'], 'ready')
        self.assertIsNone(p['error'])
        self.assertEqual(p['clips'][0]['id'], 'clip-1')


if __name__ == '__main__':
    unittest.main()
