"""Offline reselection tests; all project files are synthetic and temporary."""
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
spec = importlib.util.spec_from_file_location('reselection_server', APP / 'server.py')
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)
PID = 'd' * 32
SECRET = 'private-test-key-never-persist'  # Synthetic redaction marker only.
CONFIG = {'ai_api_key': SECRET, 'ai_provider': 'gemini', 'ai_model': 'test', 'apify_token': 'unused-test-token'}
OLD = {'id': 'clip-1', 'start': 10, 'end': 40, 'video': 'clip-1.mp4', 'status': 'ready'}
PROJECT = {'id': PID, 'status': 'ready', 'pipeline': 'local', 'title': 'Saved video', 'duration': 120,
           'source': 'source.mp4', 'source_url': 'https://www.youtube.com/watch?v=abcdefghijk',
           'segments': [{'start': 0, 'end': 120, 'text': 'Saved transcript'}], 'clips': [OLD],
           'settings': {'length': 30, 'language': 'pt'}, '_request': {'url': 'https://www.youtube.com/watch?v=abcdefghijk', 'pipeline': 'local'}}
RESULT = {'clips': [{'id': 'clip-1', 'start': 40, 'end': 100, 'title': 'Complete thought', 'reason': 'Full explanation', 'score': 85}],
          'ai_provider': 'gemini', 'model': 'test', 'duration_mode': 'ai', 'min_clip_seconds': 30, 'max_clip_seconds': 90}

class ReselectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='clipstudio-reselection-test-')
        self.folder = Path(self.temp.name)/PID
        self.folder.mkdir()
        self.paths = patch.object(srv, 'PROJECTS_DIR', Path(self.temp.name))
        self.paths.start()
        srv.atomic_json(self.folder/'project.json', copy.deepcopy(PROJECT))
        (self.folder/'clip-1.mp4').write_bytes(b'existing-render')
        (self.folder/'source.mp4').write_bytes(b'existing-source')
        self.handler = object.__new__(srv.Handler)
        self.replies = []
        self.handler.send_json = lambda data, status=200: self.replies.append((copy.deepcopy(data), status))
        self.jobs = []
        self.worker = patch.object(srv, 'WORKER', types.SimpleNamespace(submit=lambda *args:self.jobs.append(args)))
        self.worker.start()
        self.config = patch.object(srv, 'connections', return_value=types.SimpleNamespace(load=lambda **kw:copy.deepcopy(CONFIG)))
        self.config.start()
    def tearDown(self):
        self.config.stop(); self.worker.stop(); self.paths.stop(); self.temp.cleanup()
    def run_job(self, rank):
        with patch.dict(sys.modules, {'ai_ranker': types.SimpleNamespace(rank_with_ai=rank)}):
            job, *args = self.jobs.pop()
            job(*args)
    def test_success_uses_saved_transcript_once_and_preserves_source_history_exports(self):
        calls=[]
        self.handler.reselect_ai(PID, {'count':5,'message_goal':'inspiring','content_context':'church'})
        queued=srv.read_project(PID)
        self.assertEqual(self.replies[-1][1],202)
        self.assertEqual(queued['clips'],[OLD])
        self.assertNotIn(SECRET, json.dumps(queued))
        def rank(*args): calls.append(args); return copy.deepcopy(RESULT)
        self.run_job(rank)
        new=srv.read_project(PID)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0], PROJECT['segments'])
        self.assertEqual(calls[0][2]['count'],5)
        self.assertEqual(calls[0][2]['message_goal'],'inspiring')
        self.assertEqual(calls[0][2]['content_context'],'church')
        self.assertEqual(new_preferences := srv.read_project(PID)['settings']['message_goal'],'inspiring')
        self.assertNotIn('apify_token', calls[0][3])
        self.assertEqual(new['status'],'ready')
        self.assertEqual(new['source'],'source.mp4')
        self.assertEqual(new['pipeline'],'local_ai')
        self.assertEqual(new['clip_history'][0]['clips'],[OLD])
        self.assertTrue(new['clips'][0]['id'].startswith('ai-'))
        self.assertNotEqual(new['clips'][0]['id'],OLD['id'])
        self.assertEqual((self.folder/'clip-1.mp4').read_bytes(),b'existing-render')
        self.assertNotIn(SECRET,json.dumps(new))
    def test_failed_ai_keeps_previous_clips_and_masks_key(self):
        self.handler.reselect_ai(PID, {})
        def fail(*args): raise RuntimeError('Provider rejected '+SECRET)
        self.run_job(fail)
        saved=srv.read_project(PID)
        self.assertEqual(saved['clips'],[OLD])
        self.assertEqual(saved['status'],'ready')
        self.assertEqual(saved['pipeline'],'local')
        self.assertIn('kept',saved['error'])
        self.assertNotIn(SECRET,(self.folder/'diagnostic.log').read_text())
        self.assertNotIn(SECRET,json.dumps(saved))
    def test_busy_duplicate_rejected_before_second_call(self):
        self.handler.reselect_ai(PID,{})
        with self.assertRaises(srv.APIError) as error:self.handler.reselect_ai(PID,{})
        self.assertEqual(error.exception.status,409)
        self.assertEqual(len(self.jobs),1)
    def test_short_source_rejected_without_queue(self):
        p=srv.read_project(PID);p['duration']=29.99;srv.atomic_json(self.folder/'project.json',p)
        with self.assertRaises(srv.APIError):self.handler.reselect_ai(PID,{})
        self.assertFalse(self.jobs)
    def test_new_caption_controls_round_trip_and_invalid_values_fail(self):
        for style in ('bold','clean','boxed','outline','neon','minimal'):
            value={'caption_style':style,'caption_size':'large','caption_position':'top'}
            self.assertEqual(srv.render_options(value),value)
        for field,value in [('caption_style','unknown'),('caption_size',[]),('caption_position','outside')]:
            with self.assertRaises(srv.APIError):srv.render_options({field:value})
    def test_message_goal_choices_validated_without_running_ai(self):
        for goal in srv.MESSAGE_GOALS:
            self.assertEqual(srv.selection_preferences({'message_goal':goal,'content_context':'church'})['message_goal'],goal)
        for invalid in ({'message_goal':'make up a story'},{'message_goal':[]},{'content_context':'unknown'}):
            with self.assertRaises(srv.APIError): self.handler.reselect_ai(PID,invalid)
        self.assertFalse(self.jobs)
    def test_zoom_options_are_bounded_and_preserved(self):
        for mode in ('auto','wide','medium','close','manual'):
            value={'zoom_mode':mode,'zoom':1.5,'vertical_position':25}
            self.assertEqual(srv.render_options(value),value)
        for invalid in ({'zoom_mode':[]},{'zoom_mode':'unsafe'},{'zoom':2.01},{'zoom':.99},{'zoom':True},{'zoom':float('inf')}):
            with self.assertRaises(srv.APIError):srv.render_options(invalid)
    def test_interrupted_review_keeps_old_exports_and_specific_recovery_message(self):
        self.handler.reselect_ai(PID,{})
        srv.recover_interrupted_jobs()
        p=srv.read_project(PID)
        self.assertEqual(p['clips'],[OLD])
        self.assertEqual(p['status'],'ready')
        self.assertIn('AI review was interrupted',p['error'])
        self.assertNotIn('_reselect_pending',p)

if __name__ == '__main__': unittest.main()
