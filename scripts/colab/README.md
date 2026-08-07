# Running Stage-1 HPO (and pretraining) on Google Colab via SSH

## One-time: start an SSH server inside your Colab notebook

1. Open a Colab notebook, set **Runtime → Change runtime type → GPU**.
2. In a cell, run (cloudflared approach — no account needed):

```python
!pip install -q colab-ssh
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="choose-a-password")
```

3. The cell prints a hostname like `xxxx-yyy.trycloudflare.com` plus
   client-setup instructions. On the Mac, install cloudflared once
   (`brew install cloudflared`) and add to `~/.ssh/config`:

```
Host colab
  HostName <the-printed-hostname>
  User root
  ProxyCommand /opt/homebrew/bin/cloudflared access ssh --hostname %h
```

4. Tell Claude the host alias (`colab`) — or paste the hostname and it
   will be filled in. Test: `ssh colab 'nvidia-smi -L'`.

Notes:
- Colab VMs are **ephemeral** (≤12 h, sometimes less). Everything here
  is resumable: the Optuna study lives in a SQLite file that is synced
  back after every session; interrupted trials are re-sampled.
- Keep the notebook tab open (or use Colab Pro background execution);
  the SSH tunnel dies with the notebook.

## Workflow (from the Mac, repo root)

```bash
bash scripts/colab/sync_to_remote.sh colab      # push code + data + study
bash scripts/colab/remote_setup.sh colab        # deps + CUDA check (once per VM)
bash scripts/colab/run_hpo_remote.sh colab 35   # launch study detached
bash scripts/colab/pull_results.sh colab        # sync study/logs/checkpoints back
```

`pull_results.sh` is safe to run any time (rsync); run it before the VM
expires. Re-running `run_hpo_remote.sh` after a new VM + sync resumes
the study from the pulled-back SQLite file.

Final pretrain on the same VM:
```bash
ssh colab 'cd ~/mast && nohup env PYTHONPATH=src python scripts/pretrain.py \
  --run-name final --subset full --epochs 600 <winner flags> \
  > logs/nohup/final.log 2>&1 &'
```
