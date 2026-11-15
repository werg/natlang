import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'neuralese_data'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teacher_gen  # noqa: E402


def test_gemini_default_omits_the_seed_field():
    gemini = 'https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions'
    assert teacher_gen.default_chat_config(gemini) == {
        'top_p': 0.95, 'x_natlang_omit_seed': True, 'x_natlang_host_call_fields': teacher_gen.GEMINI_HOST_CALL_FIELDS}
    assert teacher_gen.default_chat_config('https://openrouter.ai/api/v1/chat/completions') == {'top_p': 0.95}
