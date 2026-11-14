"""Run the kaggle CLI as one configured account; the token is read here and never printed.

usage: python ops/kg.py <ac_1|ac_2|ac_3> <kaggle args...>
"""
import os
import shutil
import subprocess
import sys

TOKENS = os.environ.get("S4_TOKEN_FILE", r"C:\Thesis new\Kaggle_Token.txt")


def token(label):
    for line in open(TOKENS, encoding="utf-8"):
        key, _, val = line.partition(":")
        if key.strip() == label:
            return val.strip()
    sys.exit(f"no token labelled {label} in {TOKENS}")


if __name__ == "__main__":
    label, args = sys.argv[1], sys.argv[2:]
    fix = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dnsfix")  # resolves the Kaggle host names through 8.8.8.8 (see dnsfix/sitecustomize.py)
    env = dict(os.environ, KAGGLE_API_TOKEN=token(label), PYTHONPATH=os.pathsep.join(filter(None, [fix, os.environ.get("PYTHONPATH")])))
    sys.exit(subprocess.call([shutil.which("kaggle"), "-W", *args], env=env))
