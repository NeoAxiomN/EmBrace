from .head_for_bciciv2a import head_for_bciciv2a
from .head_for_chb import head_for_chb
from .head_for_faced import head_for_faced
from .head_for_isruc import head_for_isruc
from .head_for_mumtaz import head_for_mumtaz
from .head_for_physio import head_for_physio
from .head_for_seedv import head_for_seedv
from .head_for_shu import head_for_shu
from .head_for_speech import head_for_speech
from .head_for_stress import head_for_stress
from .head_for_attention import head_for_attention


def create_head(
    dataset, 
    num_classes, 
    dropout
):
    args = {"num_classes": num_classes, "dropout": dropout}

    HEAD_MAP = {
        "bciciv2a": head_for_bciciv2a,
        "chb-mit": head_for_chb,
        "faced": head_for_faced,
        "isruc-s1": head_for_isruc,
        "isruc-s3": head_for_isruc,
        "mumtaz": head_for_mumtaz,
        "physio": head_for_physio,
        "seedv": head_for_seedv,
        "shu": head_for_shu,
        "speech": head_for_speech,
        "stress": head_for_stress,
        "attention": head_for_attention,
    }

    if dataset in HEAD_MAP:
        return HEAD_MAP[dataset](**args)
    else:
        raise NotImplementedError(
            f"Task '{dataset}' has no defined head in head/ directory."
        )


__all__ = [
    "create_head",
    "head_for_bciciv2a",
    "head_for_chb",
    "head_for_faced",
    "head_for_isruc",
    "head_for_mumtaz",
    "head_for_physio",
    "head_for_seedv",
    "head_for_shu",
    "head_for_speech",
    "head_for_stress",
    "head_for_attention",
]
