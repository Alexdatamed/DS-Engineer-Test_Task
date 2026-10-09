"""Execute EDA using this Python interpreter, not a global Jupyter kernel."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
path = root / "task2_regression/notebooks/eda.ipynb"
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(notebook, km=manager, timeout=180, resources={"metadata": {"path": str(root)}})
client.execute()
nbformat.write(notebook, path)
print("Executed all EDA cells successfully")
