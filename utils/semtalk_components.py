import numpy as np
from scipy import linalg

from utils import other_tools
from utils.project_paths import pretrained_vq_path
from utils.show_base_joints import formal_joint_context


def frechet_distance(samples_a, samples_b, eps=1e-6):
    """Legacy SemTalk latent FID without importing the text-heavy data_tools."""
    mu_a = np.mean(samples_a, axis=0)
    sigma_a = np.cov(samples_a, rowvar=False)
    mu_b = np.mean(samples_b, axis=0)
    sigma_b = np.cov(samples_b, rowvar=False)
    mu_a = np.atleast_1d(mu_a)
    mu_b = np.atleast_1d(mu_b)
    sigma_a = np.atleast_2d(sigma_a)
    sigma_b = np.atleast_2d(sigma_b)
    assert mu_a.shape == mu_b.shape, (
        "Training and test mean vectors have different lengths"
    )
    assert sigma_a.shape == sigma_b.shape, (
        "Training and test covariances have different dimensions"
    )
    try:
        difference = mu_a - mu_b
        covariance_mean, _ = linalg.sqrtm(
            sigma_a.dot(sigma_b),
            disp=False,
        )
        if not np.isfinite(covariance_mean).all():
            offset = np.eye(sigma_a.shape[0]) * eps
            covariance_mean = linalg.sqrtm(
                (sigma_a + offset).dot(sigma_b + offset)
            )
        if np.iscomplexobj(covariance_mean):
            if not np.allclose(
                np.diagonal(covariance_mean).imag,
                0,
                atol=1e-3,
            ):
                raise ValueError(
                    f"imaginary Fréchet component "
                    f"{np.max(np.abs(covariance_mean.imag))}"
                )
            covariance_mean = covariance_mean.real
        return (
            difference.dot(difference)
            + np.trace(sigma_a)
            + np.trace(sigma_b)
            - 2 * np.trace(covariance_mean)
        )
    except ValueError:
        return 1e10


def build_semtalk_joint_context(ori_joints_name: str):
    return formal_joint_context(ori_joints_name)


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
