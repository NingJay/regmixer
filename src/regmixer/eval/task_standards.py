from __future__ import annotations

from typing import Dict

import pandas as pd

from regmixer.eval.constants import ALL_CORE_TASKS

# round1a writes a few task names differently than the canonical eval constants.
ROUND1A_TASK_ALIASES = {
    "commonsense_qa": "csqa",
    "social_iqa": "socialiqa",
}

MMLU_GROUP_WEIGHTS = {
    "mmlu_stem": {
        "mmlu_abstract_algebra": 0.03313452617627568,
        "mmlu_astronomy": 0.05036447978793903,
        "mmlu_college_biology": 0.04771371769383698,
        "mmlu_college_chemistry": 0.03313452617627568,
        "mmlu_college_computer_science": 0.03313452617627568,
        "mmlu_college_mathematics": 0.03313452617627568,
        "mmlu_college_physics": 0.033797216699801194,
        "mmlu_computer_security": 0.03313452617627568,
        "mmlu_conceptual_physics": 0.07786613651424784,
        "mmlu_electrical_engineering": 0.04804506295559974,
        "mmlu_elementary_mathematics": 0.12524850894632206,
        "mmlu_high_school_biology": 0.10271703114645461,
        "mmlu_high_school_chemistry": 0.06726308813783963,
        "mmlu_high_school_computer_science": 0.03313452617627568,
        "mmlu_high_school_mathematics": 0.08946322067594434,
        "mmlu_high_school_physics": 0.050033134526176276,
        "mmlu_high_school_statistics": 0.07157057654075547,
        "mmlu_machine_learning": 0.03711066931742876,
    },
    "mmlu_other": {
        "mmlu_anatomy": 0.04164096236890808,
        "mmlu_business_ethics": 0.030845157310302282,
        "mmlu_clinical_knowledge": 0.08173966687230105,
        "mmlu_college_medicine": 0.05336212214682295,
        "mmlu_global_facts": 0.030845157310302282,
        "mmlu_human_aging": 0.06878470080197409,
        "mmlu_management": 0.03177051202961135,
        "mmlu_marketing": 0.07217766810610735,
        "mmlu_medical_genetics": 0.030845157310302282,
        "mmlu_miscellaneous": 0.24151758173966686,
        "mmlu_nutrition": 0.09438618136952498,
        "mmlu_professional_accounting": 0.08698334361505243,
        "mmlu_professional_medicine": 0.08389882788402221,
        "mmlu_virology": 0.05120296113510179,
    },
    "mmlu_social_sciences": {
        "mmlu_econometrics": 0.03704907377315567,
        "mmlu_high_school_geography": 0.06434839129021774,
        "mmlu_high_school_government_and_politics": 0.06272343191420214,
        "mmlu_high_school_macroeconomics": 0.12674683132921677,
        "mmlu_high_school_microeconomics": 0.07734806629834254,
        "mmlu_high_school_psychology": 0.17712057198570036,
        "mmlu_human_sexuality": 0.04257393565160871,
        "mmlu_professional_psychology": 0.19889502762430938,
        "mmlu_public_relations": 0.03574910627234319,
        "mmlu_security_studies": 0.07962300942476438,
        "mmlu_sociology": 0.0653233669158271,
        "mmlu_us_foreign_policy": 0.032499187520311994,
    },
    "mmlu_humanities": {
        "mmlu_formal_logic": 0.026780021253985122,
        "mmlu_high_school_european_history": 0.03506907545164718,
        "mmlu_high_school_us_history": 0.04335812964930925,
        "mmlu_high_school_world_history": 0.050371944739638685,
        "mmlu_international_law": 0.0257173219978746,
        "mmlu_jurisprudence": 0.022954303931987247,
        "mmlu_logical_fallacies": 0.034643995749202974,
        "mmlu_moral_disputes": 0.07353878852284804,
        "mmlu_moral_scenarios": 0.1902231668437832,
        "mmlu_philosophy": 0.06609989373007438,
        "mmlu_prehistory": 0.06886291179596174,
        "mmlu_professional_law": 0.32603613177470775,
        "mmlu_world_religions": 0.03634431455897981,
    },
}

ROUND1A_STANDARD_METRICS = (*ALL_CORE_TASKS, *MMLU_GROUP_WEIGHTS.keys())


def canonicalize_round1a_task_name(task_name: str) -> str:
    return ROUND1A_TASK_ALIASES.get(task_name, task_name)


def canonicalize_round1a_task_frame(task_scores: pd.DataFrame) -> pd.DataFrame:
    canonical = task_scores.copy()
    canonical.columns = [canonicalize_round1a_task_name(column) for column in canonical.columns]
    if canonical.columns.duplicated().any():
        duplicates = canonical.columns[canonical.columns.duplicated()].tolist()
        raise ValueError(f"duplicate canonical task columns after alias normalization: {duplicates}")
    return canonical


def get_mmlu_group_weights(metric_suffix: str = "") -> Dict[str, Dict[str, float]]:
    return {
        group: {f"{task_name}{metric_suffix}": weight for task_name, weight in weights.items()}
        for group, weights in MMLU_GROUP_WEIGHTS.items()
    }


def aggregate_mmlu_task_scores(task_scores: pd.DataFrame) -> pd.DataFrame:
    canonical = canonicalize_round1a_task_frame(task_scores)
    aggregated = pd.DataFrame(index=canonical.index)

    for group_name, weights in MMLU_GROUP_WEIGHTS.items():
        missing = [column for column in weights if column not in canonical.columns]
        if missing:
            raise KeyError(f"missing required {group_name} task columns: {missing}")
        weight_series = pd.Series(weights, dtype="float64")
        aggregated[group_name] = canonical[weight_series.index].dot(weight_series)

    return aggregated


def build_round1a_standard_metrics(task_scores: pd.DataFrame) -> pd.DataFrame:
    canonical = canonicalize_round1a_task_frame(task_scores)
    missing_core = [task_name for task_name in ALL_CORE_TASKS if task_name not in canonical.columns]
    if missing_core:
        raise KeyError(f"missing required core task columns: {missing_core}")

    standard_metrics = pd.DataFrame(index=canonical.index)
    for task_name in ALL_CORE_TASKS:
        standard_metrics[task_name] = canonical[task_name]

    mmlu_metrics = aggregate_mmlu_task_scores(canonical)
    for group_name in MMLU_GROUP_WEIGHTS:
        standard_metrics[group_name] = mmlu_metrics[group_name]

    return standard_metrics


def compute_round1a_standard_score(task_scores: pd.DataFrame) -> pd.Series:
    return build_round1a_standard_metrics(task_scores).mean(axis=1)
