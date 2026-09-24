"""Offline AI contract tests using synthetic transcripts and credentials."""
import io
import json
import re
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_ranker as ranker


SEGMENTS = [{"start": i * 10, "end": (i + 1) * 10, "text": f"A complete source passage {i}."} for i in range(12)]
KEY = "test-secret-never-display"  # Synthetic redaction marker, not a real credential.


def legacy_selection(**updates):
    first = updates.pop("start_segment", 0)
    last = updates.pop("end_segment", 2)
    ranges = ranker._candidate_ranges(ranker._prepare_segments(SEGMENTS, 120), 120)
    match = next(((a, b) for a, b, _ in ranges
                     if type(first) is int and type(last) is int and (a, b) == (first, last)), None)
    range_id = f"w{match[0]}-{match[1]}" if match else "w999-999"
    source = " ".join(s['text'] for s in SEGMENTS[match[0]:match[1]+1]) if match else "invalid source words"
    words = re.findall(r"\w+", source)
    return {"range_id": range_id, "title": "A source idea",
            "opening_quote": " ".join(words[:3]), "closing_quote": " ".join(words[-3:]),
            "reason": "A clear opening leads into a complete explanation.",
            "start_reason": "The idea is introduced here.", "end_reason": "The explanation finishes here.",
            "score": 84, **updates}


def selection(**updates):
    first = updates.pop("start_segment", 0)
    last = updates.pop("end_segment", 2)
    return {"start_unit": first, "end_unit": last, "title": "A source idea",
            "reason": "A clear opening leads into a complete explanation.",
            "start_reason": "The idea is introduced here.", "end_reason": "The explanation finishes here.",
            "score": 84, **updates}


def gemini_response(clips=None):
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps({"clips": clips or [selection()], "coverage_note": "Offline fixture only contains enough independent moments for these options."})}]}}], "usageMetadata": {"promptTokenCount": 25, "candidatesTokenCount": 30, "totalTokenCount": 55}}


class RankerTest(unittest.TestCase):
    def run_gemini(self, response=None, segments=None, duration=120, request=None):
        with patch.object(ranker, "_post_json", return_value=response or gemini_response()) as post:
            result = ranker.rank_with_ai(segments or SEGMENTS, duration, request or {}, {"ai_provider": "gemini", "ai_api_key": KEY})
        return result, post

    def test_gemini_one_call_source_grounded(self):
        result, post = self.run_gemini()
        self.assertEqual(post.call_count, 1)
        url, headers, payload = post.call_args.args
        self.assertEqual(headers, {"x-goog-api-key": KEY})
        self.assertNotIn(KEY, url)
        self.assertNotIn(KEY, json.dumps(payload))
        self.assertNotIn(KEY, json.dumps(result))
        self.assertIn("untrusted quoted source", payload["systemInstruction"]["parts"][0]["text"])
        self.assertEqual(result["model"], "gemini-3.5-flash-lite")
        self.assertEqual(result["clips"][0]["start"], 0)
        self.assertEqual(result["clips"][0]["end"], 30)
        self.assertEqual(result["clips"][0]["text"], " ".join(s["text"] for s in SEGMENTS[:3]))
        self.assertEqual(result["usage"]["total_tokens"], 55)

    def test_openai_one_call_and_strict_schema(self):
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"clips": [selection()], "coverage_note": "Offline fixture only contains enough independent moments for these options."})}}]}
        with patch.object(ranker, "_post_json", return_value=response) as post:
            result = ranker.rank_with_ai(SEGMENTS, 120, {}, {"ai_provider": "openai", "ai_api_key": KEY})
        self.assertEqual(post.call_count, 1)
        url, headers, payload = post.call_args.args
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(headers, {"Authorization": "Bearer " + KEY})
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertFalse(payload["store"])
        self.assertEqual(result["ai_provider"], "openai")

    def test_invalid_indices_are_rejected(self):
        for first, last in [(-1, 2), (1, 999), (3, 1), (True, 2), (0, 2.0)]:
            with self.subTest(first=first, last=last), self.assertRaises(ranker.AIRankingError):
                self.run_gemini(gemini_response([selection(start_segment=first, end_segment=last)]))

    def test_overlap_keeps_the_higher_score_and_reports_fewer_options(self):
        result, post = self.run_gemini(gemini_response([
            selection(score=80), selection(start_segment=1, end_segment=3, score=95),
            selection(start_segment=6, end_segment=8, score=90)]))
        self.assertEqual([clip['start'] for clip in result['clips']], [10, 60])
        self.assertIn('Removed 1 overlapping', result['selection_warning'])
        self.assertIn(result['selection_warning'], result['coverage_note'])
        self.assertEqual(post.call_count, 1)

    def test_disjoint_clips_sorted_and_ids_assigned(self):
        result, _ = self.run_gemini(gemini_response([selection(score=60), selection(start_segment=5, end_segment=7, score=90)]))
        self.assertEqual([c["start"] for c in result["clips"]], [50, 0])
        self.assertEqual([c["id"] for c in result["clips"]], ["clip-1", "clip-2"])

    def test_long_complete_idea_is_allowed(self):
        result, _ = self.run_gemini(gemini_response([selection(end_segment=10)]))
        self.assertEqual(result["clips"][0]["end"], 110)

    def test_invalid_json_or_wrong_fields_rejected(self):
        for text in ["not JSON", '{"clips": []}', '{"clips": [], "execute": "evil"}']:
            response = gemini_response()
            response["candidates"][0]["content"]["parts"][0]["text"] = text
            with self.subTest(text=text), self.assertRaises(ranker.AIRankingError):
                self.run_gemini(response)

    def test_nonfinite_score_rejected(self):
        with self.assertRaises(ranker.AIRankingError):
            self.run_gemini(gemini_response([selection(score=float("nan"))]))

    def test_missing_explanation_rejected(self):
        with self.assertRaises(ranker.AIRankingError):
            self.run_gemini(gemini_response([selection(reason=" ")]))

    def test_truncated_response_rejected_without_fallback(self):
        response = gemini_response()
        response["candidates"][0]["finishReason"] = "MAX_TOKENS"
        with patch.object(ranker, "_post_json", return_value=response) as post:
            with self.assertRaises(ranker.AIRankingError):
                ranker.rank_with_ai(SEGMENTS, 120, {}, {"ai_provider": "gemini", "ai_api_key": KEY})
        self.assertEqual(post.call_count, 1)

    def test_bad_input_does_not_call_api(self):
        with patch.object(ranker, "_post_json") as post:
            cases = [([], 120, {}), (SEGMENTS, 7201, {}), (SEGMENTS, 120, {"count": 9}), (SEGMENTS, 120, {"message_goal": "invalid"}), (SEGMENTS, 120, {"count": 1.5}), (SEGMENTS, float("inf"), {})]
            for segments, duration, request in cases:
                with self.subTest(duration=duration, request=request), self.assertRaises(ranker.AIRankingError):
                    ranker.rank_with_ai(segments, duration, request, {"ai_provider": "gemini", "ai_api_key": KEY})
            post.assert_not_called()

    def test_large_transcript_not_truncated(self):
        segments = [{"start": 0, "end": 30, "text": "x" * 200001}]
        with patch.object(ranker, "_post_json") as post, self.assertRaisesRegex(ranker.AIRankingError, "too long"):
            ranker.rank_with_ai(segments, 60, {}, {"ai_provider": "gemini", "ai_api_key": KEY})
        post.assert_not_called()

    def test_model_cannot_inject_url(self):
        with patch.object(ranker, "_post_json") as post, self.assertRaises(ranker.AIRankingError):
            ranker.rank_with_ai(SEGMENTS, 120, {}, {"ai_provider": "gemini", "ai_api_key": KEY, "ai_model": "../evil?key=secret"})
        post.assert_not_called()

    def test_http_error_is_redacted_and_no_retry(self):
        opener = unittest.mock.Mock()
        opener.open.side_effect = urllib.error.HTTPError("https://api.openai.com", 401, KEY, {}, io.BytesIO(KEY.encode()))
        self.addCleanup(opener.open.side_effect.close)
        with patch.object(ranker.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(ranker.AIRankingError) as failure:
                ranker._post_json("https://api.openai.com/v1/chat/completions", {"Authorization": KEY}, {})
        self.assertNotIn(KEY, str(failure.exception))
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 90)

    def test_observer_captures_before_invalid_selection_without_credentials(self):
        snapshots = []
        response = gemini_response([selection(start_unit=999999, title=KEY)])
        response['usageMetadata']['unsafe_field'] = KEY
        with patch.object(ranker, '_post_json', return_value=response) as post:
            with self.assertRaises(ranker.AIRankingError):
                ranker.rank_with_ai(SEGMENTS, 120, {'_response_observer': snapshots.append},
                                    {'ai_provider': 'gemini', 'ai_api_key': KEY})
        self.assertEqual(post.call_count, 1)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(set(snapshots[0]), {'selection', 'prepared_units', 'selection_protocol', 'ai_provider', 'model', 'usage'})
        self.assertNotIn(KEY, json.dumps(snapshots))
        self.assertEqual(snapshots[0]['selection']['clips'][0]['start_unit'], 999999)
        self.assertEqual(snapshots[0]['usage']['total_tokens'], 55)

    def test_observer_failure_does_not_discard_response_or_retry(self):
        def broken(snapshot):
            raise OSError('Read-only diagnostic storage')
        with patch.object(ranker, '_post_json', return_value=gemini_response()) as post:
            result = ranker.rank_with_ai(SEGMENTS, 120, {'_response_observer': broken},
                                        {'ai_provider': 'gemini', 'ai_api_key': KEY})
        self.assertEqual(post.call_count, 1)
        self.assertEqual(len(result['clips']), 1)

    def test_observer_snapshot_is_detached_from_validation(self):
        def mutate(snapshot):
            snapshot['selection']['clips'][0]['start_unit'] = 999999
            snapshot['prepared_units'][0]['start'] = -100
        with patch.object(ranker, '_post_json', return_value=gemini_response()):
            result = ranker.rank_with_ai(SEGMENTS, 120, {'_response_observer': mutate},
                                        {'ai_provider': 'gemini', 'ai_api_key': KEY})
        self.assertEqual(result['clips'][0]['start'], 0)
        self.assertEqual(result['clips'][0]['end'], 30)

    def test_overlap_uses_unrounded_scores_and_keeps_at_least_one(self):
        result, _ = self.run_gemini(gemini_response([
            selection(score=84.3), selection(start_segment=1, end_segment=3, score=84.4)]))
        self.assertEqual(len(result['clips']), 1)
        self.assertEqual(result['clips'][0]['start'], 10)
        self.assertNotIn('_rank_score', result['clips'][0])

    def test_overlap_exact_tie_is_deterministic_in_either_response_order(self):
        a = selection(end_segment=2, score=84)
        b = selection(end_segment=3, score=84)
        for candidates in ([a, b], [b, a]):
            result, _ = self.run_gemini(gemini_response(candidates))
            self.assertEqual([(c['start'], c['end']) for c in result['clips']], [(0, 30)])

    def validate_legacy(self, candidate):
        return ranker._validate_clips({'clips': [candidate], 'coverage_note': ''},
                                     ranker._prepare_segments(SEGMENTS, 120), 120, 1)

    def test_legacy_numeric_id_cannot_be_mistaken_for_window_id(self):
        with self.assertRaisesRegex(ranker.AIRankingError, 'range outside'):
            self.validate_legacy(legacy_selection(range_id=16))

    def test_legacy_window_id_and_normalized_source_quotes(self):
        clips = self.validate_legacy(legacy_selection(
            opening_quote='A, COMPLETE source', closing_quote='source passage 2!'))
        self.assertEqual(clips[0]['range_id'], 'w0-2')
        self.assertEqual((clips[0]['start'], clips[0]['end']), (0, 30))

    def test_legacy_quotes_from_another_window_are_rejected(self):
        for changes in ({'opening_quote': 'A complete source passage 5'},
                        {'closing_quote': 'source passage 5'}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ranker.AIRankingError, 'quoted words'):
                self.validate_legacy(legacy_selection(**changes))

    def test_compact_unit_protocol_needs_no_provider_quotes_or_range_enumeration(self):
        with patch.object(ranker, '_candidate_ranges', side_effect=AssertionError('Legacy enumeration used')):
            result, post = self.run_gemini()
        payload = post.call_args.args[2]
        source = json.loads(payload['contents'][0]['parts'][0]['text'])
        schema = payload['generationConfig']['responseJsonSchema']
        fields = schema['properties']['clips']['items']['properties']
        self.assertEqual(fields['start_unit'], {'type': 'integer', 'minimum': 0, 'maximum': 11})
        self.assertEqual(fields['end_unit'], fields['start_unit'])
        self.assertNotIn('opening_quote', fields)
        self.assertNotIn('closing_quote', fields)
        self.assertNotIn('candidate_ranges', source)
        self.assertEqual(source['transcript'][0], [0, 0, 10, 'source_start', 'sentence', [], SEGMENTS[0]['text']])
        self.assertEqual(source['requested_clip_count'], 5)
        self.assertEqual(result['min_clip_seconds'], 1)
        self.assertEqual(result['max_clip_seconds'], 120)
        self.assertEqual(result['selection_protocol'], 'sentence_units_v2')
        self.assertIn('without a fixed target', result['analysis_method'])

    def test_23_second_complete_source_is_allowed_and_quotes_are_locally_derived(self):
        source = [{'start': 0, 'end': 23, 'text': 'A brief, complete insight ends here.'}]
        result, _ = self.run_gemini(gemini_response([selection(end_segment=0)]),
                                    segments=source, duration=23, request={'count': 1, 'length': 90})
        clip = result['clips'][0]
        self.assertEqual((clip['start'], clip['end']), (0, 23))
        self.assertEqual(clip['opening_quote'], source[0]['text'])
        self.assertEqual(clip['closing_quote'], source[0]['text'])
        self.assertEqual(result['max_clip_seconds'], 23)

    def test_long_resolution_remains_in_the_selected_message(self):
        source = [{'start': 0, 'end': 23, 'text': 'A difficult beginning.'},
                  {'start': 23, 'end': 105, 'text': 'The story develops.'},
                  {'start': 105, 'end': 142, 'text': 'But now reconciliation changes the meaning.'}]
        result, post = self.run_gemini(gemini_response([selection()]),
                                      segments=source, duration=142, request={'count': 1, 'length': 30})
        clip = result['clips'][0]
        self.assertEqual((clip['start'], clip['end']), (0, 142))
        self.assertIn(source[-1]['text'], clip['text'])
        self.assertTrue(clip['closing_quote'].endswith(source[-1]['text']))
        instruction = post.call_args.args[2]['systemInstruction']['parts'][0]['text']
        self.assertNotIn('within 90 seconds', instruction)
        self.assertNotIn('30–90', instruction)

    def test_unfinished_boundary_ids_are_rejected(self):
        prepared = ranker._prepare_segments(SEGMENTS, 120)
        for field, index in [('start_boundary', 0), ('end_boundary', 2)]:
            modified = [dict(unit) for unit in prepared]
            modified[index][field] = 'continuation'
            with self.subTest(field=field), self.assertRaisesRegex(ranker.AIRankingError, 'unfinished'):
                ranker._validate_unit_clips({'clips': [selection()], 'coverage_note': ''}, modified, 120, 1)

    def test_technical_subsecond_interval_is_rejected(self):
        source = [{'start': 0, 'end': .5, 'text': 'A very brief word.'}]
        with self.assertRaisesRegex(ranker.AIRankingError, 'one usable second'):
            self.run_gemini(gemini_response([selection(end_segment=0)]), segments=source, duration=1,
                            request={'count': 1})

    def test_ai_cannot_override_source_times_or_text(self):
        for extra in ({'opening_quote': 'Invented quote'}, {'start': 90}, {'range_id': 'w0-2'}):
            with self.subTest(extra=extra), self.assertRaises(ranker.AIRankingError):
                self.run_gemini(gemini_response([selection(**extra)]))

    def test_redirect_disabled(self):
        self.assertIsNone(ranker._NoRedirect().redirect_request(None, None, 302, None, None, "https://evil.example"))


if __name__ == "__main__":
    unittest.main()
