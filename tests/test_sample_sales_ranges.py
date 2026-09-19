import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "sample_sales_ranges", Path(__file__).resolve().parents[1] / "scripts" / "sample_sales_ranges.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_range_sampler_preserves_embedded_newline_records():
    data = (b"company_id,conversation_id,message\n"
            b'saas-1,one,"hello\nworld"\n'
            b"saas-2,two,plain\n"
            b"saas-3,three,partial")
    assert module.complete_records(data, 0) == [
        b"company_id,conversation_id,message\n",
        b'saas-1,one,"hello\nworld"\n',
        b"saas-2,two,plain\n",
    ]
    assert module.complete_records(b"broken\n" + data[data.index(b"saas-2"):], 100) == [
        b"saas-2,two,plain\n",
    ]
