"""Dependency-free test runner (pytest not installed in the radiacuda2 env).

Discovers and runs every ``test_*`` function in the ``test_*.py`` modules of
this directory. Test functions are plain asserts, so the same files run under
pytest unchanged if it is installed later.

Usage:
    C:/Users/Daniel/anaconda3/envs/radiacuda2/python.exe test/run_tests.py [module-substring]
"""

import _testenv  # noqa: F401  (must be first: env + sys.path)

import importlib
import os
import sys
import traceback

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def discover_modules():
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(TEST_DIR)
        if f.startswith("test_") and f.endswith(".py")
    )


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.path.insert(0, TEST_DIR)

    passed, failed, errors = [], [], []

    for mod_name in discover_modules():
        if pattern and pattern not in mod_name:
            continue
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            print(f"\n[IMPORT ERROR] {mod_name}")
            traceback.print_exc()
            errors.append(mod_name)
            continue

        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            func = getattr(module, attr)
            if not callable(func):
                continue
            test_id = f"{mod_name}.{attr}"
            try:
                func()
            except AssertionError:
                print(f"[FAIL] {test_id}")
                traceback.print_exc()
                failed.append(test_id)
            except Exception:
                print(f"[ERROR] {test_id}")
                traceback.print_exc()
                failed.append(test_id)
            else:
                print(f"[ok]   {test_id}")
                passed.append(test_id)

    print("\n" + "=" * 60)
    print(f"{len(passed)} passed, {len(failed)} failed, {len(errors)} import errors")
    print("=" * 60)
    return 1 if (failed or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
