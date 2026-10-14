import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inspect_case", ROOT / "scripts/inspect_case.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CaseInventoryTest(unittest.TestCase):
    def test_supplied_case_outages_exclude_transformers_and_bridges(self):
        data = module.inventory(ROOT / "data_generation/pglib/pglib_opf_case500_goc.m")
        self.assertEqual(data["bus_count"], 500)
        self.assertEqual(data["branch_count"], 733)
        self.assertEqual(data["active_branch_count"], 728)
        self.assertEqual(data["eligible_branch_count"], 502)
        self.assertEqual(data["active_bridge_count"], 146)
        self.assertEqual(len(set(data["eligible_branch_ids"])), 502)


if __name__ == "__main__":
    unittest.main()
