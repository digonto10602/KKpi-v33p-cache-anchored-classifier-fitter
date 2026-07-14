#!/usr/bin/env python3
import subprocess, sys
cmd=[sys.executable, 'scripts/plot_v31zd_F3inv_5panel.py', '--outdir', sys.argv[1] if len(sys.argv)>1 else 'output', '--tag', sys.argv[2] if len(sys.argv)>2 else 'debug_v31z_projF3inv_zero_110A2_L24']
raise SystemExit(subprocess.call(cmd))
