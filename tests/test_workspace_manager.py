import os
import shutil
import tempfile
import pytest
from MLAgentBench.multi_agent.workspace_manager import WorkspaceManager


@pytest.fixture
def tmp_base(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "train.py").write_text("print('training')")
    (template / "data").mkdir()
    return tmp_path, template


def test_create_worker_workspace(tmp_base):
    base, template = tmp_base
    mgr = WorkspaceManager(base_dir=str(base))
    ws_path = mgr.create(worker_id="w0", template_dir=str(template))
    assert os.path.isdir(ws_path)
    assert os.path.isfile(os.path.join(ws_path, "train.py"))


def test_cleanup_removes_workspace(tmp_base):
    base, template = tmp_base
    mgr = WorkspaceManager(base_dir=str(base))
    ws_path = mgr.create(worker_id="w0", template_dir=str(template))
    mgr.cleanup("w0")
    assert not os.path.exists(ws_path)


def test_cleanup_nonexistent_is_noop(tmp_base):
    base, _ = tmp_base
    mgr = WorkspaceManager(base_dir=str(base))
    mgr.cleanup("w_missing")  # must not raise
