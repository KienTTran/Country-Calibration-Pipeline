"""masim_analysis.configure
=================================

Helpers to assemble MaSim input configuration dictionaries and write
input rasters used by the calibration and scenario pipelines.

This module contains:
- constants and small in-repo databases (`GENOTYPE_INFO`, `DRUG_DB`, etc.)
- utilities to validate and build a `raster_db` describing ASCII rasters used
    by MaSim
- `configure()` which returns a serializable dict that is written to YAML
    and consumed by the `MaSim` runtime.

Notes
-----
- Many of the large dictionaries in this module are expressive data tables
    (drugs, genotypes, therapies) and are used directly when composing the
    final configuration (`configure()` returns them under keys like
    ``drug_db`` / ``genotype_info``).
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

from ruamel.yaml import YAML

SEASONAL_MODEL = {"enable": True}
NODATA_VALUE = -9999
yaml = YAML()

# --- YAML Database Constants ---
# These are the dictionaries that define various constants parameters used by the simulation
# TODO: #10 Convert database dictionaries to dataclasses for better type checking and validation
GENOTYPE_INFO = {
    "loci": [
        {
            "locus_name": "pfcrt",
            "position": 0,
            "alleles": [
                {
                    "value": 0,
                    "allele_name": "K76",
                    "short_name": "K",
                    "can_mutate_to": [1],
                    "mutation_level": 0,
                    "daily_cost_of_resistance": 0.0,
                },
                {
                    "value": 1,
                    "allele_name": "76T",
                    "short_name": "T",
                    "can_mutate_to": [0],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.0005,
                },
            ],
        },
        {
            "locus_name": "pfmdr1",
            "position": 1,
            "alleles": [
                {
                    "value": 0,
                    "allele_name": "N86 Y184 one copy of pfmdr1",
                    "short_name": "NY--",
                    "can_mutate_to": [1, 2, 4],
                    "mutation_level": 0,
                    "daily_cost_of_resistance": 0.0,
                },
                {
                    "value": 1,
                    "allele_name": "86Y Y184 one copy of pfmdr1",
                    "short_name": "YY--",
                    "can_mutate_to": [3, 0, 5],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.0005,
                },
                {
                    "value": 2,
                    "allele_name": "N86 184F one copy of pfmdr1",
                    "short_name": "NF--",
                    "can_mutate_to": [3, 0, 6],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.0005,
                },
                {
                    "value": 3,
                    "allele_name": "86Y 184F one copy of pfmdr1",
                    "short_name": "YF--",
                    "can_mutate_to": [1, 2, 7],
                    "mutation_level": 2,
                    "daily_cost_of_resistance": 0.00099975,
                },
                {
                    "value": 4,
                    "allele_name": "N86 Y184 2 copies of pfmdr1",
                    "short_name": "NYNY",
                    "can_mutate_to": [0],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.005,
                },
                {
                    "value": 5,
                    "allele_name": "86Y Y184 2 copies of pfmdr1",
                    "short_name": "YYYY",
                    "can_mutate_to": [1],
                    "mutation_level": 2,
                    "daily_cost_of_resistance": 0.0055,
                },
                {
                    "value": 6,
                    "allele_name": "N86 184F 2 copies of pfmdr1",
                    "short_name": "NFNF",
                    "can_mutate_to": [2],
                    "mutation_level": 2,
                    "daily_cost_of_resistance": 0.0055,
                },
                {
                    "value": 7,
                    "allele_name": "86Y 184F 2 copies of pfmdr1",
                    "short_name": "YFYF",
                    "can_mutate_to": [3],
                    "mutation_level": 3,
                    "daily_cost_of_resistance": 0.006,
                },
            ],
        },
        {
            "locus_name": "K13 Propeller",
            "position": 2,
            "alleles": [
                {
                    "value": 0,
                    "allele_name": "R561",
                    "short_name": "R",
                    "can_mutate_to": [1],
                    "mutation_level": 0,
                    "daily_cost_of_resistance": 0.0,
                },
                {
                    "value": 1,
                    "allele_name": "561H",
                    "short_name": "H",
                    "can_mutate_to": [0],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.0005,
                },
            ],
        },
        {
            "locus_name": "Plasmepsin 2-3",
            "position": 3,
            "alleles": [
                {
                    "value": 0,
                    "allele_name": "Plasmepsin 2-3 one copy",
                    "short_name": "1",
                    "can_mutate_to": [1],
                    "mutation_level": 0,
                    "daily_cost_of_resistance": 0.0,
                },
                {
                    "value": 1,
                    "allele_name": "Plasmepsin 2-3 2 copies",
                    "short_name": "2",
                    "can_mutate_to": [0],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.0005,
                },
            ],
        },
    ]
}
DRUG_DB = {
    # Artemisinin
    0: {
        "name": "ART",  # or sometimes AR
        "half_life": 0.0,
        "maximum_parasite_killing_rate": 0.999,
        "n": 25,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [2],
        "selecting_alleles": [[1]],
        "k": 4,
        "EC50": {"..0..": 0.75, "..1..": 1.2},
    },
    # Amodiaquine
    1: {
        "name": "AQ",
        "half_life": 9.0,
        "maximum_parasite_killing_rate": 0.95,
        "n": 19,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [0, 1],
        "selecting_alleles": [[1], [0, 1, 3, 4, 5, 7]],
        "k": 4,
        "EC50": {
            "00...": 0.62,
            "01...": 0.85,
            "02...": 0.5,
            "03...": 0.775,
            "04...": 0.62,
            "05...": 0.85,
            "06...": 0.5,
            "07...": 0.775,
            "10...": 0.7,
            "11...": 0.9,
            "12...": 0.65,
            "13...": 0.82,
            "14...": 0.7,
            "15...": 0.9,
            "16...": 0.65,
            "17...": 0.82,
        },
    },
    # Sulfadoxine/pyrimethamine
    2: {
        "name": "SP",
        "half_life": 6.5,
        "maximum_parasite_killing_rate": 0.9,
        "n": 15,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.0,
        "affecting_loci": [],
        "selecting_alleles": [],
        "k": 4,
        "EC50": {".....": 1.08},
    },
    # Chloroquine
    3: {
        "name": "CQ",
        "half_life": 10,
        "maximum_parasite_killing_rate": 0.95,
        "n": 19,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [0, 1],
        "selecting_alleles": [[1], [1, 3, 5, 7]],
        "k": 4,
        "EC50": {
            "00...": 0.72,
            "01...": 0.9,
            "02...": 0.72,
            "03...": 0.9,
            "04...": 0.72,
            "05...": 0.9,
            "06...": 0.72,
            "07...": 0.9,
            "10...": 1.19,
            "11...": 1.35,
            "12...": 1.19,
            "13...": 1.35,
            "14...": 1.19,
            "15...": 1.35,
            "16...": 1.19,
            "17...": 1.35,
        },
    },
    # Lumefantrine
    4: {
        "name": "LUM",
        "half_life": 4.5,
        "maximum_parasite_killing_rate": 0.99,
        "n": 20,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [0, 1],
        "selecting_alleles": [[0], [0, 2, 3, 4, 6, 7]],
        "k": 4,
        "EC50": {
            "00...": 0.8,
            "01...": 0.67,
            "02...": 0.9,
            "03...": 0.8,
            "04...": 1.0,
            "05...": 0.87,
            "06...": 1.1,
            "07...": 1.0,
            "10...": 0.75,
            "11...": 0.6,
            "12...": 0.85,
            "13...": 0.75,
            "14...": 0.95,
            "15...": 0.8,
            "16...": 1.05,
            "17...": 0.95,
        },
    },
    # Piperaquine
    5: {
        "name": "PQ",
        "half_life": 28.0,
        "maximum_parasite_killing_rate": 0.9,
        "n": 15,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [3],
        "selecting_alleles": [[1]],
        "resistant_factor": [[1]],
        "k": 4,
        "EC50": {"...0.": 0.58, "...1.": 1.4},
    },
    # Mefloquine
    6: {
        "name": "MF",
        "half_life": 21.0,
        "maximum_parasite_killing_rate": 0.9,
        "n": 15,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [1],
        "selecting_alleles": [[4, 5, 6, 7]],
        "k": 4,
        "EC50": {
            ".0...": 0.45,
            ".1...": 0.45,
            ".2...": 0.45,
            ".3...": 0.45,
            ".4...": 1.1,
            ".5...": 1.1,
            ".6...": 1.1,
            ".7...": 1.1,
        },
    },
    # Quinine
    7: {
        "name": "QUIN",
        "half_life": 18,
        "maximum_parasite_killing_rate": 0.9,
        "n": 3,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.0,
        "affecting_loci": [],
        "selecting_alleles": [],
        "k": 4,
        "EC50": {"0....": 1.41, "1....": 1.41},
    },
}
THERAPY_DB = {
    # ACT - artesunate–amodiaquine (ASAQ)
    0: {"drug_id": [0, 1], "dosing_days": [3]},
    # ACT - artemether–lumefantrine (AL)
    1: {"drug_id": [0, 4], "dosing_days": [3]},
    # ACT - artesunate-mefloquine (ASMQ)
    2: {"drug_id": [0, 6], "dosing_days": [3]},
    # ACT - Dihydroartemisinin-piperaquine (DP)
    3: {"drug_id": [0, 5], "dosing_days": [3]},
    # MONO - Amodiaquine (ADQ)
    4: {"drug_id": [1], "dosing_days": [3]},
    # MONO - Artesunate (AS)
    5: {"drug_id": [0], "dosing_days": [3]},
    # MONO - Chloroquine (CQ)
    6: {"drug_id": [3], "dosing_days": [3]},
    # MONO - Quinine (QUIN)
    7: {"drug_id": [7], "dosing_days": [7]},
    # COMBINATION - Sulfadoxine/pyrimethamine (SP)
    8: {"drug_id": [2], "dosing_days": [3]},
}
RELATIVE_INFECTIVITY = {
    "sigma": 3.91,
    "ro": 0.00031,
    # on average 1 mosquito take 3 microliters of blood per bloodeal
    "blood_meal_volume": 3,
}
STRATEGY_DB = {
    0: {
        "name": "baseline",
        "type": "MFT",
        "therapy_ids": [0],
        "distribution": [1],
    }
}


@dataclass
class CountryParams:
    country_code: str
    country_name: str
    age_distribution: list[float]
    birth_rate: float
    calibration_year: int
    death_rate: list[float]
    initial_age_structure: list[int]
    target_population: int
    starting_date: date
    ending_date: date
    start_of_comparison_period: date
    target_case_count: int
    lower_bound_case_count: int
    upper_bound_case_count: int
    calibration_betas: list[float]
    calibration_population_bins: list[int]

    def to_dict(self):
        out = asdict(self)
        out["starting_date"] = self.starting_date.strftime("%Y/%m/%d")
        out["ending_date"] = self.ending_date.strftime("%Y/%m/%d")
        out["start_of_comparison_period"] = self.start_of_comparison_period.strftime("%Y/%m/%d")
        return out

    @staticmethod
    def from_dict(data: dict):
        return CountryParams(
            country_code=data["country_code"],
            country_name=data["country_name"],
            age_distribution=data["age_distribution"],
            birth_rate=data["birth_rate"],
            calibration_year=data["calibration_year"],
            death_rate=data["death_rate"],
            initial_age_structure=data["initial_age_structure"],
            target_population=data["target_population"],
            starting_date=datetime.strptime(data["starting_date"], "%Y/%m/%d"),
            ending_date=datetime.strptime(data["ending_date"], "%Y/%m/%d"),
            start_of_comparison_period=datetime.strptime(data["start_of_comparison_period"], "%Y/%m/%d"),
            target_case_count=data["target_case_count"],
            lower_bound_case_count=data["lower_bound_case_count"],
            upper_bound_case_count=data["upper_bound_case_count"],
            calibration_betas=data["calibration_betas"],
            calibration_population_bins=data["calibration_population_bins"],
        )

    @staticmethod
    def load(name: str | None = None, file_path: str | Path | None = None) -> "CountryParams":
        """
        A simple static class method for loading country parameters from a JSON file.
        """
        assert name or file_path, "Either name or file_path must be provided."

        if file_path:
            with open(file_path, "r") as f:
                data = json.load(f)
            return CountryParams.from_dict(data)

        # If only name is provided, construct the file path
        file_path = Path("conf") / name / "test" / f"{name}_country_params.json"  # type: ignore
        return CountryParams.load(file_path=file_path)


@dataclass
class ConfigureParams:
    # country_code: str ###
    days_between_notifications: int = 30
    initial_seed_number: int = 0
    connection_string: str = "host=masimdb.vmhost.psu.edu dbname=rwanda user=sim password=sim connect_timeout=60"
    record_genome_db: bool = True
    report_frequency: int = 30
    starting_date: str = date(2005, 1, 1).strftime("%Y/%m/%d")
    start_of_comparison_period: str = date(2020, 1, 1).strftime("%Y/%m/%d")
    ending_date: str = date(2024, 1, 1).strftime("%Y/%m/%d")
    start_collect_data_day: int = 1826
    number_of_tracking_days: int = 11
    transmission_parameter: float = 0.55  # beta value
    age_structure: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 60, 100])
    initial_age_structure: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 25, 35, 45, 55, 65, 100]
    )
    artificial_rescaling_of_population_size: float = 0.25
    birth_rate: float = 0.0412
    death_rate_by_age_class: List[float] = field(
        default_factory=lambda: [
            0.02641,
            0.00202,
            0.00202,
            0.00202,
            0.00198,
            0.00247,
            0.00247,
            0.00247,
            0.00247,
            0.00247,
            0.00247,
            0.00247,
            0.00455,
            0.00455,
            0.05348,
        ]
    )
    mortality_when_treatment_fail_by_age_class: List[float] = field(
        default_factory=lambda: [
            0.040,
            0.020,
            0.020,
            0.020,
            0.020,
            0.004,
            0.004,
            0.004,
            0.004,
            0.004,
            0.004,
            0.001,
            0.001,
            0.001,
            0.001,
        ]
    )
    initial_strategy_id: int = 0
    days_to_clinical_under_five: int = 4
    days_to_clinical_over_five: int = 6
    days_mature_gametocyte_under_five: int = 4
    days_mature_gametocyte_over_five: int = 6
    gametocyte_level_under_artemisinin_action: float = 1.0
    gametocyte_level_full: float = 1.0
    p_relapse: float = 0.01
    relapse_duration: int = 30
    relapseRate: float = 4.4721
    update_frequency: int = 7
    allow_new_coinfection_to_cause_symtoms: bool = True
    using_free_recombination: bool = True
    tf_window_size: int = 60
    tf_testing_day: int = 28
    fraction_mosquitoes_interrupted_feeding: float = 0.0
    inflation_factor: float = 0.01
    # These features are disabled currently
    using_age_dependent_bitting_level: bool = False
    using_variable_probability_infectious_bites_cause_infection: bool = False
    mda_therapy_id: int = 8
    age_bracket_prob_individual_present_at_mda: List[int] = field(default_factory=lambda: [10, 40])
    mean_prob_individual_present_at_mda: List[float] = field(default_factory=lambda: [0.85, 0.75, 0.85])
    sd_prob_individual_present_at_mda: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3])

    def __post_init__(self):
        self.number_of_age_classes = len(self.age_structure)
        if len(self.death_rate_by_age_class) != len(self.age_structure):
            raise ValueError("The length of death_rate_by_age_class must match the length of age_structure.")
        if len(self.mortality_when_treatment_fail_by_age_class) != len(self.age_structure):
            raise ValueError(
                "The length of mortality_when_treatment_fail_by_age_class must match the length of age_structure."
            )
        # Validate that all non-defaulted values are greater than or equal to zero
        if self.days_between_notifications < 0:
            raise ValueError("days_between_notifications must be greater than or equal to zero.")
        if self.initial_seed_number < 0:
            raise ValueError("initial_seed_number must be greater than or equal to zero.")
        if self.record_genome_db not in [True, False]:
            raise ValueError("record_genome_db must be a boolean value.")
        if self.report_frequency < 0:
            raise ValueError("report_frequency must be greater than or equal to zero.")
        if self.start_collect_data_day < 0:
            raise ValueError("start_collect_data_day must be greater than or equal to zero.")
        if self.number_of_tracking_days < 0:
            raise ValueError("number_of_tracking_days must be greater than or equal to zero.")
        if self.transmission_parameter < 0:
            raise ValueError("transmission_parameter must be greater than or equal to zero.")
        if self.artificial_rescaling_of_population_size < 0:
            raise ValueError("artificial_rescaling_of_population_size must be greater than or equal to zero.")
        if self.birth_rate < 0:
            raise ValueError("birth_rate must be greater than or equal to zero.")
        if any(rate < 0 for rate in self.death_rate_by_age_class):
            raise ValueError("All values in death_rate_by_age_class must be greater than or equal to zero.")
        if any(rate < 0 for rate in self.mortality_when_treatment_fail_by_age_class):
            raise ValueError(
                "All values in mortality_when_treatment_fail_by_age_class must be greater than or equal to zero."
            )
        if self.days_to_clinical_under_five < 0:
            raise ValueError("days_to_clinical_under_five must be greater than or equal to zero.")
        if self.days_to_clinical_over_five < 0:
            raise ValueError("days_to_clinical_over_five must be greater than or equal to zero.")
        if self.days_mature_gametocyte_under_five < 0:
            raise ValueError("days_mature_gametocyte_under_five must be greater than or equal to zero.")
        if self.days_mature_gametocyte_over_five < 0:
            raise ValueError("days_mature_gametocyte_over_five must be greater than or equal to zero.")
        if self.gametocyte_level_under_artemisinin_action < 0:
            raise ValueError("gametocyte_level_under_artemisinin_action must be greater than or equal to zero.")
        if self.gametocyte_level_full < 0:
            raise ValueError("gametocyte_level_full must be greater than or equal to zero.")
        if self.p_relapse < 0:
            raise ValueError("p_relapse must be greater than or equal to zero.")
        if self.relapse_duration < 0:
            raise ValueError("relapse_duration must be greater than or equal to zero.")
        if self.relapseRate < 0:
            raise ValueError("relapseRate must be greater than or equal to zero.")
        if self.update_frequency < 0:
            raise ValueError("update_frequency must be greater than or equal to zero.")
        if self.tf_window_size < 0:
            raise ValueError("tf_window_size must be greater than or equal to zero.")
        if self.tf_testing_day < 0:
            raise ValueError("tf_testing_day must be greater than or equal to zero.")
        if self.fraction_mosquitoes_interrupted_feeding < 0:
            raise ValueError("fraction_mosquitoes_interrupted_feeding must be greater than or equal to zero.")
        if self.inflation_factor < 0:
            raise ValueError("inflation_factor must be greater than or equal to zero.")
        if self.mda_therapy_id < 0:
            raise ValueError("mda_therapy_id must be greater than or equal to zero.")
        if any(prob < 0 for prob in self.mean_prob_individual_present_at_mda):
            raise ValueError("All values in mean_prob_individual_present_at_mda must be greater than or equal to zero.")
        if any(sd < 0 for sd in self.sd_prob_individual_present_at_mda):
            raise ValueError("All values in sd_prob_individual_present_at_mda must be greater than or equal to zero.")

        self.number_of_age_classes = len(self.age_structure)


@dataclass
class ParasiteDensityLevel:
    log_parasite_density_cured: float = -4.699
    log_parasite_density_from_liver: float = -2.000
    log_parasite_density_asymptomatic: float = 3
    log_parasite_density_clinical: float = 4.301
    log_parasite_density_clinical_from: float = 3.301
    log_parasite_density_clinical_to: float = 5.301
    log_parasite_density_detectable: float = 1.000
    log_parasite_density_detectable_pfpr: float = 1.699
    log_parasite_density_pyrogenic: float = 3.398


@dataclass
class ImmuneSystemInformation:
    b1: float = 0.00125
    b2: float = 0.0025
    duration_for_naive: int = 300
    duration_for_fully_immune: int = 60
    mean_initial_condition: float = 0.1
    sd_initial_condition: float = 0.1
    immune_inflation_rate: float = 0.01
    max_clinical_probability: float = 0.99
    immune_effect_on_progression_to_clinical: float = 4
    age_mature_immunity: int = 10
    factor_effect_age_mature_immunity: int = 1
    midpoint: float = 0.4


@dataclass
class CirculationInfo:
    max_relative_moving_value: int = 35
    number_of_moving_levels: int = 100
    moving_level_distribution: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "Gamma",
            "Exponential": {"scale": 0.17},
            "Gamma": {"mean": 5, "sd": 10},
        }
    )
    circulation_percent: float = 0.00336
    length_of_stay: Dict[str, Any] = field(default_factory=lambda: {"mean": 5, "sd": 10})


@dataclass
class ParasiteInfo:
    parasite_type_id: int
    prevalence: float


@dataclass
class InitialParasiteInfoEntry:
    location_id: int
    parasite_info: list[ParasiteInfo]

    def to_dict(self):
        return {"location_id": self.location_id, "parasite_info": [asdict(p) for p in self.parasite_info]}


@dataclass
class RelativeBittingInfo:
    max_relative_biting_value: int = 35
    number_of_biting_levels: int = 100
    biting_level_distribution: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "Gamma",
            "Exponential": {"scale": 0.17},
            "Gamma": {"mean": 5, "sd": 10},
        }
    )


# Create the actual database entries
parasite_density_level = ParasiteDensityLevel()
immune_system_information = ImmuneSystemInformation()
circulation_info = CirculationInfo()
initial_parasite_info = [
    InitialParasiteInfoEntry(
        location_id=-1,
        parasite_info=[
            ParasiteInfo(parasite_type_id=32, prevalence=0.05),
            ParasiteInfo(parasite_type_id=36, prevalence=0.05),
        ],
    )
]
relative_bitting_info = RelativeBittingInfo()


def create_spatial_model(calibration_mode: bool = False) -> dict:
    """Return the spatial coupling model used by MaSim.

    The repository uses a simple 'Wesolowski' gravity-style model. When
    ``calibration_mode`` is True the returned parameters are chosen to be
    suitable for calibration runs (smaller domain / simplified mobility).

    Parameters
    ----------
    calibration_mode
        If True, return parameters optimized for calibration (deterministic
        or reduced mobility). If False, return the full production parameters.

    Returns
    -------
    dict
        Mapping with top-level key "name" and a nested parameter dictionary
        used by the MaSim runtime (example shape shown in caller code).
    """
    if calibration_mode:
        kappa = 0
        alpha = 0
        beta = 0
        gamma = 0
    else:
        kappa = 0.01093251
        alpha = 0.22268982
        beta = 0.14319618
        gamma = 0.83741484
    return {
        "name": "Wesolowski",
        "Wesolowski": {
            "kappa": kappa,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        },
    }


def create_seasonal_model_rainfall(enable: bool, country_code: str, period: int = 365) -> dict:
    """Create a seasonality configuration block for MaSim.

    The function returns a dictionary describing how rainfall/seasonality data
    should be read by MaSim. The CSV file referenced is expected at
    ``data/<country_code>/<country_code>_seasonality.csv``.

    Parameters
    ----------
    enable
        Whether seasonality should be enabled.
    country_code
        Country code used to build the seasonality filename under ``data/``.
    period
        Period (in days) describing the seasonal cycle (default 365).

    Returns
    -------
    dict
        Seasonality configuration dictionary suitable for inclusion in the
        full MaSim execution control configuration.
    """
    return {
        "enable": enable,
        "mode": "rainfall",
        "rainfall": {
            "filename": os.path.join("data", country_code, f"{country_code}_seasonality.csv"),
            "period": period,
        },
    }
<<<<<<< HEAD

=======
>>>>>>> 024dab2 (Update to use qsub to run more calibrations on clusters)


def create_seasonal_model_pattern(enable: bool, country_code: str, period: int = 365, is_single_location: bool = False) -> dict:
    """Create a seasonality configuration block for MaSim.

    The function returns a dictionary describing how rainfall/seasonality data
    should be read by MaSim. The CSV file referenced is expected at
    ``data/<country_code>/<country_code>_seasonality.csv``.

    Parameters
    ----------
    enable
        Whether seasonality should be enabled.
    country_code
        Country code used to build the seasonality filename under ``data/``.
    period
        Period (in days) describing the seasonal cycle (default 365).

    Returns
    -------
    dict
        Seasonality configuration dictionary suitable for inclusion in the
        full MaSim execution control configuration.
    """
    return {
        "enable": enable,
        "mode": "pattern",
        "pattern": {
            "admin_level": "district",
            "filename": os.path.join("data", country_code, f"{country_code}_seasonality_1_location.csv") if is_single_location else os.path.join("data", country_code, f"{country_code}_seasonality.csv"),
            "period": period,
        },
    }


def load_yaml(file_path: str) -> dict:
    """Load a YAML file and return its contents as a native Python dict.

    This utility wraps ``ruamel.yaml`` to read YAML files used for country
    configuration (strategy dbs and test parameter files under ``conf/``).

    Parameters
    ----------
    file_path
        Path to the YAML file to read.

    Returns
    -------
    dict
        Parsed YAML data.
    """
    yaml = YAML()
    with open(file_path, "r") as file:
        return yaml.load(file)


def create_raster_db(
    name: str,
    calibration: bool = False,
    calibration_string: str = "",
    access_rate: float = -1.0,
    age_distribution: list[float] = [
        0.0378,
        0.0378,
        0.0378,
        0.0378,
        0.0282,
        0.0282,
        0.0282,
        0.0282,
        0.0282,
        0.029,
        0.029,
        0.029,
        0.029,
        0.029,
        0.169,
        0.134,
        0.106,
        0.066,
        0.053,
        0.035,
        0.0,
    ],
    beta: float = -1.0,
) -> dict:
    """Validate input raster files and construct the `raster_db` mapping.

    The returned mapping contains the standard keys expected by MaSim's
    execution control YAML such as:

    - ``population_raster``: path to population ASCII raster
    - ``district_raster``: path to administrative district ASCII raster
    - ``pf_treatment_under5`` / ``pf_treatment_over5``: treatment-seeking rasters
    - ``age_distribution_by_location``: per-location age structure

    When ``calibration`` is True the function may write calibration-specific
    versions of population/district rasters (appending ``calibration_string``
    to filenames) so downstream calibration code can use multiple variants.

    Parameters
    ----------
    name
        Country or project name used to locate `conf/` and `data/` directories.
    calibration
        If True, create and return paths for calibration-specific raster files.
    calibration_string
        String appended to output raster filenames when calibration is enabled.
    access_rate
        Access rate override written into the raster_db entries.
    age_distribution
        Default per-location age distribution used when building the mapping.
    beta
        Optional beta override (used only for calibration outputs).

    Returns
    -------
    dict
        `raster_db` mapping consumed by `configure()` when composing the
        execution control dictionary.
    """
    conf_root = os.path.join("conf", name)
    data_root = os.path.join("data", name)
    if calibration:
        data_root = os.path.join(data_root, "calibration")
        conf_root = os.path.join(conf_root, "calibration")
        calibration_string = f"_{calibration_string}"
    try:
        os.makedirs(data_root)
    except FileExistsError:
        pass
    try:
        os.makedirs(conf_root)
    except FileExistsError:
        pass

    administrative_boundaries = {
        "name": "district",
        "raster": os.path.join(data_root, f"{name}{calibration_string}_districts.asc"),
    }

    raster_db = {
        "population_raster": os.path.join(data_root, f"{name}{calibration_string}_initialpopulation.asc"),
        # "district_raster": os.path.join(data_root, f"{name}{calibration_string}_districts.asc"),asdasd
        "administrative_boundaries": [administrative_boundaries],
        "cell_size": 5,
<<<<<<< HEAD
        "pr_treatment_under5": os.path.join(data_root, f"{name}_treatmentseeking.asc"),
        "pr_treatment_over5": os.path.join(data_root, f"{name}_treatmentseeking.asc"),
=======
        "pr_treatment_under5": os.path.join(data_root, f"{name}_treatmentseeking_{access_rate}.asc") if access_rate >= 0.0 else os.path.join(data_root, f"{name}_treatmentseeking.asc")  ,
        "pr_treatment_over5": os.path.join(data_root, f"{name}_treatmentseeking_{access_rate}.asc") if access_rate >= 0.0 else os.path.join(data_root, f"{name}_treatmentseeking.asc"),
>>>>>>> 024dab2 (Update to use qsub to run more calibrations on clusters)
        "age_distribution_by_location": [age_distribution],
        "p_treatment_for_less_than_5_by_location": [access_rate],
        "p_treatment_for_more_than_5_by_location": [access_rate],
    }
    if calibration:
        if beta < 0.0:
            raise ValueError("Beta value must be greater than zero for calibration.")

        raster_db["beta_by_location"] = [beta]
    else:
        raster_db["beta_by_location"] = [-1.0]
        raster_db["beta_raster"] = os.path.join(data_root, f"{name}_beta.asc")

    return raster_db


# TODO: #12 Refactor configure.configure to take only input parameters and then use the dataclass to create the configuration
def configure(
    country_code: str,
    birth_rate: float,
    initial_age_structure: list[int],
    age_distribution: list[float],  # pass to validate_raster_files
    death_rates: list[float],
    starting_date: date,
    start_of_comparison_period: date,
    ending_date: date,
    strategy_db: dict = STRATEGY_DB,
    calibration_str: str = "",  # pass to validate raster files
    beta_override: float = -1.0,  # pass to validate
    population_scalar: float = 0.25,  # pass to validate
    access_rate_override: float = -1.0,  # pass to validate
    calibration: bool = False,
    is_single_location: bool = False,
) -> dict:
    """Assemble the full MaSim execution control dictionary.

    The returned dictionary contains all keys required by the MaSim runtime
    execution control YAML: demographic parameters, ``raster_db``, spatial and
    seasonal models, parasite/drug databases, strategies and events.

    Parameters
    ----------
    country_code
        Country code used to build file references under ``data/`` and ``conf/``.
    birth_rate
        Annual birth rate used for demographic scaling.
    initial_age_structure
        Vector of initial per-age-class population counts.
    age_distribution
        Vector of per-age-class age distribution fractions.
    death_rates
        Per-age-class death rates.
    starting_date, start_of_comparison_period, ending_date
        `datetime.date` objects describing the simulation time window.
    strategy_db
        Intervention definitions placed into the configuration under
        ``strategy_db`` (defaults to module-level ``STRATEGY_DB``).
    calibration_str, beta_override, population_scalar, access_rate_override
        Calibration related overrides that affect `raster_db` contents.
    calibration
        When True, generate a configuration optimized for calibration runs.

    Returns
    -------
    dict
        Fully populated execution control mapping ready to be serialized to
        YAML and passed to `MaSim`.
    """
    if calibration:
        assert calibration_str is not None, "Calibration string must be provided for calibration mode."
        assert beta_override >= 0.0, "Beta override must be greater than or equal to zero for calibration mode."
    assert population_scalar > 0.0 and population_scalar <= 1.0, (
        "Population scalar must be greater than 0 and less than or equal to 1."
    )
    params = ConfigureParams(
        birth_rate=birth_rate,
        initial_age_structure=initial_age_structure,
        death_rate_by_age_class=death_rates,
        starting_date=starting_date.strftime("%Y/%m/%d"),
        start_of_comparison_period=start_of_comparison_period.strftime("%Y/%m/%d"),
        ending_date=ending_date.strftime("%Y/%m/%d"),
        artificial_rescaling_of_population_size=population_scalar,
    )
    execution_control = asdict(params)
    execution_control["raster_db"] = create_raster_db(
        country_code, calibration, calibration_str, access_rate_override, age_distribution, beta_override
    )
    execution_control["spatial_model"] = create_spatial_model(calibration)
    execution_control["seasonal_info"] = create_seasonal_model_pattern(True, country_code,365,is_single_location)
    execution_control["parasite_density_level"] = asdict(parasite_density_level)
    execution_control["immune_system_information"] = asdict(immune_system_information)
    execution_control["circulation_info"] = asdict(circulation_info)
    execution_control["initial_parasite_info"] = [entry.to_dict() for entry in initial_parasite_info]
    execution_control["relative_bitting_info"] = asdict(relative_bitting_info)
    execution_control["relative_infectivity"] = RELATIVE_INFECTIVITY
    execution_control["genotype_info"] = GENOTYPE_INFO
    execution_control["drug_db"] = DRUG_DB
    execution_control["therapy_db"] = THERAPY_DB
    execution_control["strategy_db"] = strategy_db
    execution_control["events"] = [{"name": "turn_off_mutation", "info": [{"day": params.starting_date}]}]

    return execution_control
