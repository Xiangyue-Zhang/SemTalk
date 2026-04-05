#!/usr/bin/env python3

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def check_exists(relpath: str, failures: list[str]):
    path = REPO_ROOT / relpath
    if path.exists():
        print(f"OK hidden prerequisite: {relpath}")
    else:
        failures.append(f"Missing hidden prerequisite: {relpath}")


def main() -> int:
    failures: list[str] = []

    # Current inference path still relies on the bundled reference NPZ for betas metadata.
    check_exists("demo/2_scott_0_1_1.npz", failures)
    check_exists("demo/2_scott_0_1_1_test.wav", failures)
    check_exists("weights", failures)
    check_exists("weights/pretrained_vq", failures)
    check_exists("weights/smplx_models", failures)

    if failures:
        print("\nHidden prerequisite check failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nHidden prerequisite check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
