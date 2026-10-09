"""Small restoration checks, without network access or regenerating data."""

import hashlib
import json
import unittest

from experiments import restore_simulation_data as restore


class RestorationTests(unittest.TestCase):
    def test_download_is_bounded_to_summary_and_seven_results(self):
        files = restore.selected_files()
        self.assertEqual(len(files), 8)
        self.assertEqual(len(set(files)), 8)
        self.assertEqual(files[0], "runs/llm.parquet")
        for batch, filename in zip(restore.BATCHES, files[1:]):
            self.assertIn(f"max_num_seqs+{batch}+", filename)
            self.assertTrue(filename.endswith("/results.json"))
            self.assertIn("num_request_repeats+2+" if batch >= 256
                          else "num_request_repeats+1+", filename)
            self.assertNotIn("*", filename)

    def test_timestamp_normalization_is_in_memory_only(self):
        original = {"written_utc": restore.ORIGINAL_WRITTEN_UTC, "spec_hash": "same"}
        current = dict(original, written_utc="2026-10-08T15:04:19+00:00")
        encode = lambda obj: json.dumps(obj, indent=2, sort_keys=True).replace("\n", "\r\n").encode()
        before = encode(current)
        self.assertEqual(restore.original_timestamp_hash(before),
                         hashlib.sha256(encode(original)).hexdigest())
        self.assertEqual(before, encode(current))
        changed = dict(current, spec_hash="different")
        self.assertNotEqual(restore.original_timestamp_hash(encode(changed)),
                            hashlib.sha256(encode(original)).hexdigest())

    def test_saved_restoration_evidence_matches_recorded_original(self):
        path = restore.ROOT / "results/simulator-restoration/verification.json"
        if not path.exists():
            self.skipTest("Restoration not yet completed")
        record = json.loads(path.read_text(encoding="utf-8"))
        original = json.loads((restore.ROOT / "results/power-model-pilot-verified/manifest.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(record["dataset_revision"], restore.REVISION)
        self.assertEqual(record["openg2g_commit"], restore.COMMIT)
        self.assertEqual(record["package_version_differences"], {})
        self.assertTrue(record["raw_inputs_unchanged"])
        self.assertEqual(set(record["selected_raw_sha256"]), set(restore.selected_files()))
        self.assertTrue(record["manifest_matches_after_timestamp_only_normalization"])
        self.assertEqual(record["unexpected_prepared_files"], [])
        self.assertEqual(len(record["numerical_artifacts_match_original"]), 3)
        for key, matches in record["numerical_artifacts_match_original"].items():
            self.assertTrue(matches)
            self.assertEqual(record["prepared_data_sha256"][key], original["prepared_data_sha256"][key])
        # Where ignored local data is present, verify the actual restored bytes.
        prepared = restore.ROOT / "data/prepared"
        if prepared.exists():
            self.assertEqual(restore.fingerprint(prepared), record["prepared_data_sha256"])
            manifest = prepared / restore.SPEC_HASH / "_manifest.json"
            self.assertEqual(restore.original_timestamp_hash(manifest.read_bytes()),
                             original["prepared_data_sha256"][f"{restore.SPEC_HASH}/_manifest.json"])


if __name__ == "__main__":
    unittest.main()
