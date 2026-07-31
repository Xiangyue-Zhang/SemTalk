#!/usr/bin/env bash
# Shared fail-closed Python identity contract for formal SHOW launchers.
#
# The caller-supplied executable path is authority: launchers must keep that
# exact absolute argv[0], including the normal ``venv/bin/python`` symlink.
# Resolving the leaf and then executing its system target bypasses pyvenv.cfg
# and silently changes sys.prefix, sys.path, and the available dependencies.

semtalk_require_formal_venv_python() {
    local python_path=${1-}
    local profile=${2-}
    local python_parent python_name venv_root pyvenv_cfg resolved_target

    if [[ -z $python_path || -z $profile || $python_path != /* || \
          $python_path == *$'\n'* || $python_path == *$'\r'* ]]; then
        printf '%s\n' \
            'formal Python must be one exact absolute executable path' >&2
        return 1
    fi
    case $profile in
        minimal|semtalk) ;;
        *)
            printf 'unknown formal Python runtime profile: %s\n' \
                "$profile" >&2
            return 1
            ;;
    esac

    python_name=${python_path##*/}
    [[ $python_name =~ ^python([0-9]+([.][0-9]+)*)?$ ]] || {
        printf 'formal Python leaf name is invalid: %s\n' "$python_name" >&2
        return 1
    }
    python_parent=$(CDPATH= cd -- "$(dirname -- "$python_path")" 2>/dev/null \
        && pwd -P) || {
        printf '%s\n' 'formal Python parent directory is unavailable' >&2
        return 1
    }
    if [[ $python_path != "$python_parent/$python_name" || \
          ${python_parent##*/} != bin || -L $python_parent ]]; then
        printf '%s\n' \
            'formal Python parent must be one canonical non-symlink venv/bin' >&2
        return 1
    fi
    venv_root=${python_parent%/bin}
    if [[ -z $venv_root || $venv_root == "$python_parent" || \
          ! -d $venv_root || -L $venv_root || \
          $(CDPATH= cd -- "$venv_root" 2>/dev/null && pwd -P) != "$venv_root" ]]; then
        printf '%s\n' \
            'formal Python virtual-environment root is unavailable or unsafe' >&2
        return 1
    fi
    if [[ ! -f $python_path || ! -x $python_path ]]; then
        printf '%s\n' 'formal Python executable is unavailable' >&2
        return 1
    fi
    resolved_target=$(realpath -- "$python_path") || {
        printf '%s\n' 'formal Python executable target is unavailable' >&2
        return 1
    }
    if [[ ! -f $resolved_target || ! -x $resolved_target || -L $resolved_target ]]; then
        printf '%s\n' 'formal Python executable target is unsafe' >&2
        return 1
    fi
    pyvenv_cfg=$venv_root/pyvenv.cfg
    if [[ ! -f $pyvenv_cfg || -L $pyvenv_cfg || \
          $(realpath -- "$pyvenv_cfg") != "$pyvenv_cfg" ]]; then
        printf '%s\n' 'formal Python pyvenv.cfg is unavailable or unsafe' >&2
        return 1
    fi

    # -I makes the validation independent of caller cwd, PYTHONPATH, and the
    # user site.  It does not change sys.executable: the exact venv symlink is
    # retained and verified below.  The formal profile also proves the modules
    # consumed by inference/evaluation are importable from this environment.
    "$python_path" -I -c '
import contextlib
import importlib
import io
import os
import platform
import site
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"formal Python runtime contract: {message}")


expected = sys.argv[1]
venv_root = Path(sys.argv[2])
cfg_path = Path(sys.argv[3])
profile = sys.argv[4]
if os.fsencode(sys.executable) != os.fsencode(expected):
    fail("sys.executable did not retain the exact supplied argv[0]")
if Path(sys.prefix) != venv_root or Path(sys.exec_prefix) != venv_root:
    fail("sys.prefix/sys.exec_prefix do not identify the supplied venv")
if Path(sys.prefix).resolve(strict=True) != venv_root:
    fail("virtual-environment prefix is noncanonical")
if sys.prefix == sys.base_prefix or sys.exec_prefix == sys.base_exec_prefix:
    fail("interpreter is not running in a virtual environment")
if sys.flags.isolated != 1 or sys.flags.no_user_site != 1:
    fail("isolated runtime flags are not active")

configuration = {}
try:
    for raw_line in cfg_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        if "=" not in raw_line:
            fail("invalid pyvenv.cfg record")
        key, value = raw_line.split("=", 1)
        key = key.strip().casefold()
        if not key or key in configuration:
            fail("duplicate or empty pyvenv.cfg key")
        configuration[key] = value.strip()
except OSError as exc:
    fail(f"cannot read pyvenv.cfg: {exc}")
required_keys = {"home", "include-system-site-packages", "version"}
if not required_keys.issubset(configuration):
    fail("pyvenv.cfg is missing required runtime identity fields")
if configuration["include-system-site-packages"].casefold() not in {"true", "false"}:
    fail("pyvenv.cfg has an invalid include-system-site-packages value")
if configuration["version"] != platform.python_version():
    fail("pyvenv.cfg Python version does not match the running interpreter")

configured_home = Path(configuration["home"])
if not Path(expected).is_absolute() or not configured_home.is_absolute():
    fail("base executable identity is not absolute")
try:
    resolved_base_executable = Path(expected).resolve(strict=True)
    resolved_home = configured_home.resolve(strict=True)
    resolved_home_executable = (
        resolved_home / resolved_base_executable.name
    ).resolve(strict=True)
    if resolved_home_executable != resolved_base_executable:
        fail("pyvenv.cfg home does not match the supplied executable target")
    # Python 3.9 creates legitimate pyvenv.cfg files without ``executable``.
    # Newer versions include it; when present it is an additional exact bind.
    configured_executable_value = configuration.get("executable")
    if configured_executable_value is not None:
        configured_executable = Path(configured_executable_value)
        if not configured_executable.is_absolute():
            fail("pyvenv.cfg executable is not absolute")
        if configured_executable.resolve(strict=True) != resolved_base_executable:
            fail("pyvenv.cfg executable does not match the supplied executable target")
except OSError as exc:
    fail(f"base executable identity is unavailable: {exc}")

try:
    configured_sites = [Path(value) for value in site.getsitepackages()]
except AttributeError as exc:
    fail(f"virtual-environment site-packages is unavailable: {exc}")
# Debian/Ubuntu may report optional dist-packages locations which do not
# exist.  They are inert.  Require the actual versioned venv site-packages,
# while ignoring only those optional absent entries.
expected_site = (
    venv_root
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
try:
    expected_site = expected_site.resolve(strict=True)
except OSError as exc:
    fail(f"versioned venv site-packages is unavailable: {exc}")
site_roots = []
for path in configured_sites:
    try:
        site_roots.append(path.resolve(strict=True))
    except OSError:
        continue
if expected_site not in site_roots:
    fail("venv site-packages is absent from the isolated runtime")
resolved_sys_path = []
for value in sys.path:
    if not value:
        continue
    try:
        resolved_sys_path.append(Path(value).resolve(strict=True))
    except OSError:
        continue
if expected_site not in resolved_sys_path:
    fail("venv site-packages is absent from sys.path")

required_modules = ()
if profile == "semtalk":
    required_modules = (
        "numpy", "torch", "scipy", "einops", "smplx", "transformers", "lmdb"
    )
elif profile != "minimal":
    fail("unknown runtime profile")
with contextlib.redirect_stdout(io.StringIO()):
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            fail(f"required module {module_name!r} is unavailable: {exc}")
' "$python_path" "$venv_root" "$pyvenv_cfg" "$profile"
}
