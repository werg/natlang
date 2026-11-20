import tempfile
import unittest
from pathlib import Path

from audit_open_goal_references import write_train_cases


class TrainSelectorTest(unittest.TestCase):
    def test_uses_only_accepted_training_goals_and_preserves_an_existing_plan(self):
        result = {
            'cases': {
                'open-goal-confirmed': {'split': 'train'},
                'open-goal-unverified': {'split': 'train'},
                'open-goal-holdout': {'split': 'test'},
            },
            'proofs': {'open-goal-confirmed': {}, 'open-goal-holdout': {}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'capture-cases.txt'
            write_train_cases(destination, result)
            self.assertEqual(destination.read_text(), 'open-goal-confirmed\n')
            with self.assertRaises(FileExistsError):
                write_train_cases(destination, result)

            empty_destination = Path(tmp) / 'empty.txt'
            result['proofs'].pop('open-goal-confirmed')
            with self.assertRaises(ValueError):
                write_train_cases(empty_destination, result)
            self.assertFalse(empty_destination.exists())


if __name__ == '__main__':
    unittest.main()
