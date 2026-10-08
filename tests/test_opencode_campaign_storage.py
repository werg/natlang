import tempfile
import unittest
from pathlib import Path

from scripts.opencode_campaign_storage import ensure_parent


class CampaignStorageTests(unittest.TestCase):
    def test_ensure_parent_creates_missing_campaign_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "campaign-v1" / "nested" / "launch.json"

            parent = ensure_parent(target)

            self.assertEqual(parent, target.parent)
            self.assertTrue(parent.is_dir())
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
