"""Strict dependency check, with a tested, explicit Decord 0.6.0 tag exception.

Some Linux Decord wheels carry incompatible installed WHEEL metadata despite
providing a working shared library. Never suppress other pip check failures;
only allow this exact legacy warning after a real CPU video decode succeeds.
No metadata/package/source files are modified.
"""
from __future__ import annotations
import importlib.metadata as md
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def main() -> int:
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], text=True, capture_output=True)
    output = (result.stdout + result.stderr).strip()
    print(output)
    if result.returncode == 0:
        return 0
    known = 'decord 0.6.0 is not supported on this platform'
    lines = [row.strip() for row in output.splitlines() if row.strip()]
    if lines != [known] or platform.system() != 'Linux' or platform.machine() != 'x86_64':
        return result.returncode or 1
    if md.version('decord') != '0.6.0':
        return 1
    import av
    import decord
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='reroute-decord-') as tmp:
        path = Path(tmp) / 'check.avi'
        with av.open(str(path), 'w') as writer:
            stream = writer.add_stream('mpeg4', rate=2)
            stream.width = stream.height = 64
            stream.pix_fmt = 'yuv420p'
            for value in (40, 180):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), value, dtype=np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    writer.mux(packet)
            for packet in stream.encode():
                writer.mux(packet)
        reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
        frames = reader.get_batch([0, 1]).asnumpy()
        if len(reader) != 2 or frames.shape != (2, 64, 64, 3):
            raise RuntimeError('Decord binary decode smoke failed')
        if float(frames[1].mean()) <= float(frames[0].mean()):
            raise RuntimeError('Decord decoded frames are not ordered correctly')
        del reader
    print(json.dumps({'dependency_conflicts': False, 'warning': known,
                      'warning_is_not_hidden': True,
                      'decord_cpu_binary_decode': '2 frames verified',
                      'wheel_metadata': md.distribution('decord').read_text('WHEEL'),
                      'scope': 'Linux x86_64 only; NOT video benchmark validation'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
