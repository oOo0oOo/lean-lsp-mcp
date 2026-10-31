"""Run inside the built image: docker run --rm -i --entrypoint python IMAGE < this.py."""

import ensurepip
import hashlib
import json
import os
import re
import ssl
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

import certifi

# SHA256 of the DER certificate, independent of PEM comments or formatting.
DISTRUSTED_ROOT = "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"


def check_bundle(bundle):
    certificates = re.findall(
        r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", bundle, re.S
    )
    assert certificates, "Empty certificate bundle"
    assert all(
        hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest() != DISTRUSTED_ROOT
        for pem in certificates
    ), "Bundle still trusts emSign ECC Root CA - G3"


def check_installer(python):
    info = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import json, pip; from pip._vendor import certifi; "
                "print(json.dumps([pip.__version__, open(certifi.where()).read()]))",
            ],
            text=True,
        )
    )
    assert info[0] == "26.2.2", info[0]
    check_bundle(info[1])


assert os.getuid() == 10001, "Image must retain its unprivileged runtime user"
assert os.environ["LEAN_MCP_DISABLED_TOOLS"] == "lean_run_code"
assert sys.prefix == "/app/.venv", sys.prefix
check_bundle(Path(certifi.where()).read_text())
base_python = Path(sys.base_prefix) / "bin/python"
check_installer(base_python)

assert ensurepip.version() == "26.2.2"
wheels = list((Path(ensurepip.__file__).parent / "_bundled").glob("pip-*.whl"))
assert len(wheels) == 1, wheels
with zipfile.ZipFile(wheels[0]) as archive:
    check_bundle(archive.read("pip/_vendor/certifi/cacert.pem").decode())

with tempfile.TemporaryDirectory() as directory:
    venv.EnvBuilder(with_pip=True).create(directory)
    python = Path(directory) / "bin/python"
    check_installer(python)
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(wheels[0]),
        ],
        check=True,
    )
    check_installer(python)

subprocess.run(["uv", "--version"], check=True)
subprocess.run(["elan", "--version"], check=True)
subprocess.run(["lean-lsp-mcp", "--version"], check=True)
print("Container installer, CA bundle, venv, and CLI checks passed")
