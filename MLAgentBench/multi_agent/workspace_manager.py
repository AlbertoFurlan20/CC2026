import os
import shutil
from typing import Dict


class WorkspaceManager:
    """Creates and cleans up isolated workspace directories per worker."""

    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        self._workspaces: Dict[str, str] = {}

    def create(self, worker_id: str, template_dir: str) -> str:
        """Copy template_dir into an isolated workspace for worker_id."""
        dest = os.path.join(self._base_dir, f"worker_{worker_id}")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(template_dir, dest)
        self._workspaces[worker_id] = dest
        return dest

    def cleanup(self, worker_id: str) -> None:
        """Remove the workspace for worker_id if it exists."""
        path = self._workspaces.pop(worker_id, None)
        if path and os.path.exists(path):
            shutil.rmtree(path)

    def cleanup_all(self) -> None:
        for wid in list(self._workspaces):
            self.cleanup(wid)

    def get_path(self, worker_id: str) -> str:
        return self._workspaces[worker_id]
