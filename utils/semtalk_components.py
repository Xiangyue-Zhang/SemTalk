import numpy as np

from dataloaders.data_tools import joints_list
from utils import other_tools
from utils.project_paths import pretrained_vq_path


def build_semtalk_joint_context(ori_joints_name: str):
    ori_joint_list = joints_list[ori_joints_name]
    target_joint_sets = {
        "face": joints_list["beat_smplx_face"],
        "upper": joints_list["beat_smplx_upper"],
        "hands": joints_list["beat_smplx_hands"],
        "lower": joints_list["beat_smplx_lower"],
    }

    masks = {}
    for key, joint_names in target_joint_sets.items():
        mask = np.zeros(len(list(ori_joint_list.keys())) * 3)
        for joint_name in joint_names:
            start = ori_joint_list[joint_name][1] - ori_joint_list[joint_name][0]
            end = ori_joint_list[joint_name][1]
            mask[start:end] = 1
        masks[key] = mask

    return {
        "ori_joint_list": ori_joint_list,
        "target_joint_sets": target_joint_sets,
        "masks": masks,
        "joints": 55,
    }


def load_pretrained_vq_suite(args, device, checkpoint_tag, include_global_motion=False):
    rvq_model_module = __import__("models.rvq", fromlist=["something"])
    motion_rep_module = __import__("models.motion_representation", fromlist=["something"]) if include_global_motion else None

    original_state = {
        "vae_layer": getattr(args, "vae_layer", None),
        "vae_length": getattr(args, "vae_length", None),
        "vae_test_dim": getattr(args, "vae_test_dim", None),
    }

    def _restore_args():
        for key, value in original_state.items():
            if value is not None:
                setattr(args, key, value)

    def _tag(name: str):
        if isinstance(checkpoint_tag, dict):
            return checkpoint_tag[name]
        return checkpoint_tag

    try:
        args.vae_layer = 2
        args.vae_length = 256

        args.vae_test_dim = 106
        vq_model_face = getattr(rvq_model_module, "RVQVAE")(args).to(device)
        other_tools.load_checkpoints(vq_model_face, str(pretrained_vq_path("face")), _tag("face"))

        args.vae_test_dim = 78
        vq_model_upper = getattr(rvq_model_module, "RVQVAE")(args).to(device)
        other_tools.load_checkpoints(vq_model_upper, str(pretrained_vq_path("upper")), _tag("upper"))

        args.vae_test_dim = 180
        vq_model_hands = getattr(rvq_model_module, "RVQVAE")(args).to(device)
        other_tools.load_checkpoints(vq_model_hands, str(pretrained_vq_path("hands")), _tag("hands"))

        args.vae_test_dim = 61
        args.vae_layer = 4
        vq_model_lower = getattr(rvq_model_module, "RVQVAE")(args).to(device)
        other_tools.load_checkpoints(vq_model_lower, str(pretrained_vq_path("lower")), _tag("lower"))

        suite = {
            "face": vq_model_face.eval(),
            "upper": vq_model_upper.eval(),
            "hands": vq_model_hands.eval(),
            "lower": vq_model_lower.eval(),
        }

        if include_global_motion:
            global_motion = getattr(motion_rep_module, "VAEConvZero")(args).to(device)
            other_tools.load_checkpoints(global_motion, str(pretrained_vq_path("global_motion")), _tag("global_motion"))
            suite["global_motion"] = global_motion.eval()

        return suite
    finally:
        _restore_args()
