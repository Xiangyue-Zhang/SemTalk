"""Text-free SMPL-X joint selections used by the formal SHOW Base pipeline."""

from __future__ import annotations

import numpy as np


SMPLX_55 = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
)
COMPONENT_JOINTS = {
    "beat_smplx_full": SMPLX_55,
    "beat_smplx_upper": (
        "spine1",
        "spine2",
        "spine3",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ),
    "beat_smplx_hands": SMPLX_55[25:],
    "beat_smplx_lower": (
        "pelvis",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_foot",
        "right_foot",
    ),
    "beat_smplx_face": ("jaw",),
}


def formal_joint_context(ori_joints_name: str) -> dict[str, object]:
    if ori_joints_name != "beat_smplx_joints":
        raise RuntimeError(
            "formal SHOW training requires ori_joints=beat_smplx_joints"
        )
    source = {
        name: [3, (index + 1) * 3]
        for index, name in enumerate(SMPLX_55)
    }
    target_joint_sets = {
        stage: {name: 3 for name in COMPONENT_JOINTS[target]}
        for stage, target in {
            "face": "beat_smplx_face",
            "upper": "beat_smplx_upper",
            "hands": "beat_smplx_hands",
            "lower": "beat_smplx_lower",
        }.items()
    }
    masks: dict[str, np.ndarray] = {}
    for stage, joints in target_joint_sets.items():
        mask = np.zeros(len(SMPLX_55) * 3)
        for name in joints:
            width, end = source[name]
            mask[end - width:end] = 1
        masks[stage] = mask
    return {
        "ori_joint_list": source,
        "target_joint_sets": target_joint_sets,
        "masks": masks,
        "joints": len(SMPLX_55),
    }
