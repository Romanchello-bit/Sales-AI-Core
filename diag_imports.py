import sys, importlib, importlib.util, traceback, subprocess, os
print('EXE:', sys.executable)
print('PYTHONVERSION:', sys.version)
print('\n--- sys.path ---')
for p in sys.path:
    print(p)

print('\n--- pip show google-generativeai ---')
try:
    import pkgutil
    import gzip
    import json
    # run pip show via subprocess
    r = subprocess.run([sys.executable, '-m', 'pip', 'show', 'google-generativeai'], capture_output=True, text=True)
    print(r.stdout or r.stderr)
except Exception as e:
    print('pip show error', e)

print('\n--- pip show graphviz ---')
r = subprocess.run([sys.executable, '-m', 'pip', 'show', 'graphviz'], capture_output=True, text=True)
print(r.stdout or r.stderr)

print('\n--- importlib specs ---')
print('spec google ->', importlib.util.find_spec('google'))
print('spec google.generativeai ->', importlib.util.find_spec('google.generativeai'))
print('spec graphviz ->', importlib.util.find_spec('graphviz'))

print('\n--- try importing graphviz ---')
try:
    import graphviz
    print('graphviz imported, file ->', getattr(graphviz, '__file__', None))
    print('graphviz version ->', getattr(graphviz, '__version__', None))
except Exception:
    print('graphviz import traceback:')
    traceback.print_exc()

print('\n--- try importing google.generativeai ---')
try:
    import google.generativeai as gen
    print('google.generativeai imported, file ->', getattr(gen, '__file__', None))
    print('google.generativeai attrs ->', [a for a in dir(gen) if not a.startswith('_')][:20])
except Exception:
    print('google.generativeai import traceback:')
    traceback.print_exc()

print('\n--- check dot binary in PATH ---')
print('PATH:', os.environ.get('PATH'))
for cmd in ('dot', 'dot.exe'):
    try:
        r = subprocess.run(['where', cmd], capture_output=True, text=True, shell=True)
        print(f"where {cmd} ->", r.stdout or r.stderr)
    except Exception as e:
        print('where error', e)

print('\nDone')

