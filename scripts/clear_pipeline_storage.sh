#!/bin/bash
set -euo pipefail

docker system prune -af
sudo rm -rf /tmp/containerd-mount*
rm -rf ~/deploy_agent/_work/*