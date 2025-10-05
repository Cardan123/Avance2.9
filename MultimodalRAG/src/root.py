import os
import sys

# Obtener el directorio raíz del proyecto (un nivel arriba de src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Project root directory:", PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("After modification, sys.path includes:", PROJECT_ROOT)