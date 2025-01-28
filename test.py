import os
import shutil
import graphviz

print(f"Graphviz version: {graphviz.__version__}")

# Check if the dot executable is found
dot_path = shutil.which('dot')
if dot_path:
    print(f"Graphviz executable found at: {dot_path}")
else:
    print("Graphviz executable not found in the system PATH.")
