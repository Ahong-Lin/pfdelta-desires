import json
import math
import os
from typing import Any

import torch
from tqdm import tqdm

from core.datasets.pfdelta_variants import PFDeltaCANOS, PFDeltaGNS, PFDeltaPFNet
from core.utils.registry import registry


class PFDeltaRhoMixin:
    """
    Reader mixin for PFDelta-rho datasets.

    The rho generator writes one directory with this shape:

        root_dir/
          raw/sample_*.json
          splits/train.json
          splits/val.json
          splits/test.json

    Each split file lists raw sample paths. This reader bypasses the standard
    PFDelta shuffle-file layout and processes exactly those listed samples.
    """

    default_processed_subdir = "processed/rho_edge_rho"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        processed_subdir: str | None = None,
        include_branch_rho: bool = True,
        **kwargs: Any,
    ):
        self.rho_root = root_dir
        self.processed_subdir = processed_subdir or self.default_processed_subdir
        super().__init__(
            root_dir=root_dir,
            split=split,
            include_branch_rho=include_branch_rho,
            **kwargs,
        )

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def raw_file_names(self) -> list:
        return [
            os.path.join("raw", "sample_1.json"),
            os.path.join("splits", "train.json"),
            os.path.join("splits", "val.json"),
            os.path.join("splits", "test.json"),
        ]

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_subdir)

    @property
    def processed_file_names(self) -> list:
        return [f"{self.split}.pt"]

    def download(self) -> None:
        # Local/generated dataset only.
        return

    def _split_file_path(self, split: str) -> str:
        return os.path.join(self.root, "splits", f"{split}.json")

    def _raw_sample_paths(self, split: str) -> list[str]:
        split_path = self._split_file_path(split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing rho split file: {split_path}")

        with open(split_path, "r") as f:
            split_data = json.load(f)

        records = split_data.get("records", [])
        if not records:
            raise ValueError(f"No records found in rho split file: {split_path}")

        sample_paths = []
        for record in records:
            rel_path = record["file"]
            sample_paths.append(os.path.join(self.root, rel_path))
        return sample_paths

    @staticmethod
    def _complex_bus_voltage(solution_data: dict, bus_id: int) -> complex:
        bus_sol = solution_data["bus"][str(bus_id)]
        vm = float(bus_sol["vm"])
        va = float(bus_sol["va"])
        return vm * complex(math.cos(va), math.sin(va))

    @staticmethod
    def _branch_flow_from_voltage(network_data: dict, solution_data: dict, branch: dict) -> dict:
        if int(branch.get("br_status", 1)) == 0:
            return {"pf": 0.0, "qf": 0.0, "pt": 0.0, "qt": 0.0}

        r = float(branch["br_r"])
        x = float(branch["br_x"])
        if abs(r) + abs(x) == 0.0:
            return {"pf": 0.0, "qf": 0.0, "pt": 0.0, "qt": 0.0}

        ys = 1.0 / complex(r, x)
        y_fr = complex(float(branch.get("g_fr", 0.0)), float(branch.get("b_fr", 0.0)))
        y_to = complex(float(branch.get("g_to", 0.0)), float(branch.get("b_to", 0.0)))

        tap_mag = float(branch.get("tap", 1.0) or 1.0)
        shift = float(branch.get("shift", 0.0) or 0.0)
        tap = tap_mag * complex(math.cos(shift), math.sin(shift))

        vf = PFDeltaRhoMixin._complex_bus_voltage(solution_data, int(branch["f_bus"]))
        vt = PFDeltaRhoMixin._complex_bus_voltage(solution_data, int(branch["t_bus"]))

        yff = (ys + y_fr) / (tap * tap.conjugate()).real
        yft = -ys / tap.conjugate()
        ytf = -ys / tap
        ytt = ys + y_to

        s_fr = vf * (yff * vf + yft * vt).conjugate()
        s_to = vt * (ytf * vf + ytt * vt).conjugate()
        return {
            "pf": float(s_fr.real),
            "qf": float(s_fr.imag),
            "pt": float(s_to.real),
            "qt": float(s_to.imag),
        }

    @staticmethod
    def _ensure_branch_solution(pm_case: dict) -> None:
        if "network" not in pm_case or "solution" not in pm_case:
            return

        solution_wrapper = pm_case["solution"]
        if not isinstance(solution_wrapper, dict):
            return
        solution_data = solution_wrapper.get("solution")
        if not isinstance(solution_data, dict):
            return

        network_data = pm_case["network"]
        branch_solution = solution_data.setdefault("branch", {})
        for branch_id, branch in network_data.get("branch", {}).items():
            branch_id = str(branch_id)
            if branch_id not in branch_solution:
                branch_solution[branch_id] = PFDeltaRhoMixin._branch_flow_from_voltage(
                    network_data,
                    solution_data,
                    branch,
                )

    def process(self) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)

        for split in [self.split]:
            sample_paths = self._raw_sample_paths(split)
            data_list = []

            for sample_path in tqdm(sample_paths, desc=f"Building rho {split} data"):
                with open(sample_path, "r") as f:
                    pm_case = json.load(f)

                self._ensure_branch_solution(pm_case)
                data = self.build_heterodata(pm_case, is_cpf_sample=False)
                if "rho_meta" in pm_case:
                    data.rho = torch.tensor([float(pm_case["rho_meta"]["rho"])])
                    data.load_setting_id = torch.tensor(
                        [int(pm_case["rho_meta"]["load_setting_id"])]
                    )
                    data.rho_sample_id = torch.tensor(
                        [int(pm_case["rho_meta"].get("sample_id", 0))]
                    )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                data_list.append(data)

            data, slices = self.collate(data_list)
            torch.save((data, slices), os.path.join(self.processed_dir, f"{split}.pt"))


@registry.register_dataset("pfdeltaRhoCANOS")
class PFDeltaRhoCANOS(PFDeltaRhoMixin, PFDeltaCANOS):
    """PFDelta-rho reader using the CANOS/FlowCAN graph representation."""

    default_processed_subdir = "processed/canos_rho_edge_rho"


@registry.register_dataset("pfdeltaRhoPFNet")
class PFDeltaRhoPFNet(PFDeltaRhoMixin, PFDeltaPFNet):
    """PFDelta-rho reader using the PFNet graph representation."""

    default_processed_subdir = "processed/pfnet_rho_edge_rho"


@registry.register_dataset("pfdeltaRhoGNS")
class PFDeltaRhoGNS(PFDeltaRhoMixin, PFDeltaGNS):
    """PFDelta-rho reader using the GNS graph representation."""

    default_processed_subdir = "processed/gns_rho_edge_rho"
