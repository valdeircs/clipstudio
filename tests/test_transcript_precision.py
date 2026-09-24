"""Offline caption precision tests; no media, subtitle or AI requests are made."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_ranker
import engine
import local_transcript

URL = 'https://www.youtube.com/watch?v=abcdefghijk'
JSON3 = {'events': [{'tStartMs': 0, 'dDurationMs': 40000, 'segs': [
    {'utf8': 'First'}, {'utf8': ' thought.', 'tOffsetMs': 1000},
    {'utf8': ' Next', 'tOffsetMs': 20000}, {'utf8': ' thought.', 'tOffsetMs': 21000},
]}]}
INFO = {'id': 'abcdefghijk', 'title': 'Synthetic source', 'duration': 100, 'language': 'pt',
        'automatic_captions': {'pt-orig': [{'ext': 'json3', 'url': 'https://www.youtube.com/api/timedtext?lang=pt'}]},
        'subtitles': {}}
COARSE = {'segments': [{'start': 0, 'end': 40, 'text': 'First thought. Next thought.'}],
          'language': 'pt', 'duration': 100, 'transcript_origin': 'Local caption fixture'}
REQUEST = {'url': URL, 'pipeline': 'local_ai', 'language': 'pt',
           '_api_config': {'ai_provider': 'gemini', 'ai_api_key': 'synthetic-offline-marker'}}
AI = {'clips': [{'start': 0, 'end': 40, 'title': 'Synthetic idea'}], 'analysis_method': 'Offline fixture'}


class TranscriptPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='clipstudio-transcript-test-')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.caption_file = self.folder / 'captions-abcdefghijk.pt-orig.json3'
        self.caption_file.write_text(json.dumps(JSON3))
        self.cues = engine._json3(self.caption_file, 100)
        self.precise = {**COARSE, 'segments': self.cues,
                        'transcript_origin': 'Local YouTube captions (JSON3 source word timings)'}

    def cache(self, transcript):
        (self.folder / 'local-transcript.json').write_text(json.dumps(
            {'url': URL, 'language': 'pt', 'result': transcript}))

    def test_sentence_split_uses_actual_offsets_and_keeps_every_word(self):
        result = engine.sentence_aligned_segments(self.cues, 100)
        self.assertEqual([(x['start'], x['end'], x['text']) for x in result],
                         [(0, 20, 'First thought.'), (20, 40, 'Next thought.')])
        self.assertEqual([word['start'] for row in result for word in row['words']], [0, 1, 20, 21])
        self.assertEqual(' '.join(x['text'] for x in result), self.cues[0]['text'])
        self.assertEqual(engine.sentence_aligned_segments(result, 100), result)

    def test_untimed_phrase_never_gets_invented_word_times(self):
        result = engine.sentence_aligned_segments(COARSE['segments'], 100)
        self.assertEqual(result, COARSE['segments'])
        self.assertFalse(engine._has_word_timing(result))
        self.assertNotIn('words', result[0])

    def test_mismatched_optional_word_text_falls_back_without_text_changes(self):
        cue = {**COARSE['segments'][0], 'words': [
            {'start': 0, 'end': 10, 'word': 'Different'}, {'start': 10, 'end': 40, 'word': 'words.'}]}
        result = engine.sentence_aligned_segments([cue], 100)
        self.assertEqual(result, COARSE['segments'])

    def test_track_selection_prefers_original_and_never_another_language(self):
        info = copy.deepcopy(INFO)
        info['automatic_captions']['pt'] = [{'ext': 'json3', 'url': 'https://www.youtube.com/api/timedtext?tlang=pt'}]
        info['subtitles']['pt'] = [{'ext': 'json3', 'url': 'https://www.youtube.com/api/timedtext?lang=pt'}]
        self.assertEqual(engine._select_caption_track(info, 'pt'), ('pt-orig', True))
        del info['automatic_captions']['pt-orig']
        self.assertEqual(engine._select_caption_track(info, 'pt'), ('pt', False))
        self.assertIsNone(engine._select_caption_track(info, 'de'))

    def test_caption_request_is_exact_subtitles_only_and_reuses_json3(self):
        self.caption_file.unlink()
        def fake_run(args, folder, **kw):
            self.assertIn('--skip-download', args)
            self.assertIn('--write-auto-subs', args)
            self.assertIn('--no-write-subs', args)
            self.assertEqual(args[args.index('--sub-langs') + 1], '^pt\\-orig$')
            self.assertEqual(args[args.index('--sub-format') + 1], 'json3')
            self.assertIn('--load-info-json', args)
            self.assertNotIn('-f', args)
            self.caption_file.write_text(json.dumps(JSON3))
            return ''
        with patch.object(engine, '_binary', return_value='synthetic-yt-dlp'), patch.object(engine, '_run', side_effect=fake_run) as run:
            first = engine._local_word_captions(self.folder, URL, 'pt', INFO, lambda *args: None)
            second = engine._local_word_captions(self.folder, URL, 'pt', INFO, lambda *args: None)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(first['transcript_precision'], 'word')
        self.assertFalse(list(self.folder.glob('caption-request-*')))
        self.assertFalse(list(self.folder.glob('*.mp4')))

    def test_existing_word_cache_skips_caption_fetch_and_observes_before_one_ai_call(self):
        self.cache({**self.precise, 'apify_run_id': 'synthetic-accepted-run'})
        events = []
        response_observer = lambda *args: None
        def observer(metadata):
            self.assertTrue(all((self.folder / name).is_file() for name in ('transcript.txt', 'transcript.srt', 'transcript.json')))
            self.assertEqual(metadata['transcript_precision'], 'word')
            self.assertEqual(metadata['apify_run_id'], 'synthetic-accepted-run')
            self.assertIn('untimed phrases', metadata['transcript_precision_note'])
            self.assertEqual(len(metadata['segments']), 2)
            events.append('transcript')
            metadata['title'] = 'Observer mutation must not alter result'
        def rank(segments, duration, request, config):
            self.assertEqual(events, ['transcript'])
            self.assertIs(request['_response_observer'], response_observer)
            events.append('ai')
            return copy.deepcopy(AI)
        request = {**REQUEST, '_transcript_observer': observer, '_response_observer': response_observer}
        with patch.object(engine, '_metadata_only', return_value=INFO), patch.object(engine, '_local_word_captions') as captions, patch.object(local_transcript, 'fetch_transcript') as fallback, patch.object(ai_ranker, 'rank_with_ai', side_effect=rank) as ai:
            result = engine.analyze_api_project(self.folder, request, lambda *args: None)
        captions.assert_not_called(); fallback.assert_not_called()
        self.assertEqual(ai.call_count, 1)
        self.assertEqual(result['title'], INFO['title'])
        self.assertEqual(events, ['transcript', 'ai'])
        self.assertIsNone(result['source'])

    def test_malformed_json3_falls_back_without_inventing_word_timing(self):
        self.caption_file.write_text('["invalid JSON3 structure"]')
        with patch.object(engine, '_binary', return_value=None), patch.object(engine, '_run') as run:
            result = engine._local_word_captions(self.folder, URL, 'pt', INFO, lambda *args: None)
        self.assertIsNone(result)
        run.assert_not_called()

    def test_coarse_cache_upgrades_without_fetching_video_or_whisper(self):
        self.cache(COARSE)
        with patch.object(engine, '_metadata_only', return_value=INFO), patch.object(engine, '_local_word_captions', return_value=self.precise) as captions, patch.object(local_transcript, 'fetch_transcript') as fallback, patch.object(ai_ranker, 'rank_with_ai', return_value=AI), patch.object(engine, '_download') as media, patch.object(engine, '_transcribe') as whisper:
            result = engine.analyze_api_project(self.folder, REQUEST, lambda *args: None)
        self.assertEqual(captions.call_count, 1)
        fallback.assert_not_called(); media.assert_not_called(); whisper.assert_not_called()
        self.assertEqual(result['transcript_precision'], 'word')
        saved = json.loads((self.folder / 'local-transcript.json').read_text())
        self.assertTrue(saved['result']['segments'][0]['words'])

    def test_unavailable_json3_reuses_coarse_cache_and_marks_precision(self):
        self.cache(COARSE)
        with patch.object(engine, '_metadata_only', return_value=INFO), patch.object(engine, '_local_word_captions', return_value=None), patch.object(local_transcript, 'fetch_transcript') as fallback, patch.object(ai_ranker, 'rank_with_ai', return_value=AI):
            result = engine.analyze_api_project(self.folder, REQUEST, lambda *args: None)
        fallback.assert_not_called()
        self.assertEqual(result['transcript_precision'], 'cue')
        self.assertIn('cannot be inferred', result['transcript_precision_note'])

    def test_observer_failure_prevents_spending_an_ai_call(self):
        self.cache(self.precise)
        def fail(_): raise OSError('Synthetic storage failure')
        with patch.object(engine, '_metadata_only', return_value=INFO), patch.object(ai_ranker, 'rank_with_ai') as ai:
            with self.assertRaises(OSError):
                engine.analyze_api_project(self.folder, {**REQUEST, '_transcript_observer': fail}, lambda *args: None)
        ai.assert_not_called()
        self.assertTrue((self.folder / 'transcript.json').is_file())


if __name__ == '__main__':
    unittest.main()
