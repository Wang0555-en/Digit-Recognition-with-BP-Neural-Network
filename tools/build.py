"""Build the existing Visual Studio solution with a normalized Windows environment."""
import os
import subprocess
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
installation = subprocess.check_output([str(vswhere), "-latest", "-property", "installationPath"], text=True).strip()
environment = {key.upper(): value for key, value in os.environ.items()}
raise SystemExit(subprocess.call([str(Path(installation) / "MSBuild/Current/Bin/MSBuild.exe"), str(root / "BP.sln"), "/p:Configuration=" + (sys.argv[1] if len(sys.argv)>1 else "Release"), "/p:Platform=x64", "/m", "/v:minimal"], cwd=root, env=environment))
