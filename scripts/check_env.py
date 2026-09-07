import sys

def check_pkg(name):
    try:
        mod = __import__(name)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"[OK] {name} == {ver}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False

print(f"Python: {sys.version}")
pkgs = ['torch', 'torchvision', 'PIL', 'numpy', 'xarray', 'cfgrib', 'eccodes', 's3fs', 'satpy', 'pyresample', 'fastapi']
for p in pkgs:
    check_pkg(p)
