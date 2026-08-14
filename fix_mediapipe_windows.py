"""One-time workaround for a MediaPipe bug on Windows.

Recent MediaPipe Windows wheels ship a ``libmediapipe.dll`` that does not export
the C ``free`` symbol, but ``mediapipe/tasks/python/core/mediapipe_c_bindings.py``
assumes it does. This crashes on the first FaceLandmarker call with:

    AttributeError: function 'free' not found

This script patches the binding to fall back to the Windows C runtime
(``ucrtbase.dll``), which the DLL actually links against. Run it once after
installing dependencies:

    python fix_mediapipe_windows.py

It is idempotent and safe to re-run.
"""

import pathlib
import sys

OLD_LINES = [
    '  # Register "free()"',
    '  _shared_lib.free.argtypes = [ctypes.c_void_p]',
    '  _shared_lib.free.restype = None',
]
NEW_LINES = [
    '  # Register "free()"',
    '  try:',
    '      free_func = _shared_lib.free',
    '  except AttributeError:',
    '      # Some libmediapipe.dll builds do not export free(); fall back to the',
    '      # Windows C runtime (ucrtbase.dll), which the DLL links against.',
    "      free_func = ctypes.CDLL('ucrtbase.dll').free",
    '  free_func.argtypes = [ctypes.c_void_p]',
    '  free_func.restype = None',
    '  _shared_lib.free = free_func',
]


def main():
    import mediapipe  # locate whichever environment this runs in

    target = (
        pathlib.Path(mediapipe.__file__).parent
        / 'tasks'
        / 'python'
        / 'core'
        / 'mediapipe_c_bindings.py'
    )
    if not target.exists():
        print(f'ERROR: could not locate {target}', file=sys.stderr)
        return 1

    data = target.read_bytes()
    nl = b'\r\n' if b'\r\n' in data else b'\n'
    old = nl.join(x.encode() for x in OLD_LINES) + nl
    new = nl.join(x.encode() for x in NEW_LINES) + nl

    if new in data:
        print('Already patched — nothing to do.')
        return 0
    if old not in data:
        print(f'ERROR: unexpected content in {target}; refusing to patch.', file=sys.stderr)
        return 1

    target.write_bytes(data.replace(old, new))
    print(f'Patched {target}')
    print('MediaPipe Windows "free" workaround applied.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
