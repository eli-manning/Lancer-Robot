# Deploying the app to the robot

The app is a normal pip install on the robot, not a checkout, so pushing a
change means copying files into two places and restarting it. Written down
because working this out from scratch cost an hour and a password prompt that
could not be answered.

## Access

```sh
ssh -i ~/.ssh/id_ed25519_lancer pollen@10.5.1.1
```

- User is `pollen`, host is `reachy-mini`, `sudo -i` for root.
- `10.5.1.1` is the USB-Ethernet link (robot `eth1`, NetworkManager profile
  `mac-link`, Mac side `10.5.1.82`). Tailscale is `100.116.136.5` and works too,
  at roughly triple the latency.
- The key is `~/.ssh/id_ed25519_lancer`, robot-only, no passphrase. Revoke by
  deleting its line from `~pollen/.ssh/authorized_keys` on the robot.

**ssh-copy-id must be run from a real terminal.** Through a harness with no TTY,
ssh falls back to a graphical askpass helper that does not exist on macOS, and
reports `Permission denied` without ever having prompted -- which looks exactly
like a wrong password or a wrong username and is neither.

## Paths

| What | Where |
| --- | --- |
| Installed package (what actually runs) | `/venvs/apps_venv/lib/python3.12/site-packages/reachy_mini_lancer_app/` |
| Install source | `/home/pollen/lancer_src/src/reachy_mini_lancer_app/` |

Both get updated. The installed copy is what runs; the source copy is what a
reinstall would restore from, so skipping it means the fix silently reverts the
next time anyone reinstalls.

## Push a change

```sh
K=~/.ssh/id_ed25519_lancer
SITE=/venvs/apps_venv/lib/python3.12/site-packages/reachy_mini_lancer_app
SRC=/home/pollen/lancer_src/src/reachy_mini_lancer_app
cd lancer_app/src/reachy_mini_lancer_app

# back out of a bad deploy with the .bak file this leaves behind
ssh -i $K pollen@10.5.1.1 "cp $SITE/main.py $SITE/main.py.bak-\$(date +%Y%m%d-%H%M%S)"
scp -i $K main.py sounds.py pollen@10.5.1.1:$SITE/
scp -i $K main.py sounds.py pollen@10.5.1.1:$SRC/
ssh -i $K pollen@10.5.1.1 "/venvs/apps_venv/bin/python -c \
  'import ast; ast.parse(open(\"$SITE/main.py\").read())'"
```

Diff the robot's copy against the repo before overwriting. The two drifted
apart at least once, and finding that out afterwards is worse.

## Restart it

```sh
curl -X POST http://10.5.1.1:8000/api/apps/start-app/reachy_mini_lancer_app
curl -s http://10.5.1.1:8000/api/apps/current-app-status
```

`restart-current-app` returns **400 when no app is currently running**, and the
status endpoint then returns a bare `null` rather than an object. Use
`start-app/<name>` when that happens.

## Confirm it came up

```sh
ssh -i $K pollen@10.5.1.1 \
  "journalctl --since '-2 minutes' --no-pager | grep -iE 'lancer|Noise floor|Turned toward|Traceback'"
```

The app's own logging arrives through `reachy_mini.apps.manager.runner` at
WARNING level, so `INFO:__main__:...` lines show up prefixed and are easy to
miss when grepping for the app name alone.

Lines worth seeing:

- `Noise floor calibrated: gate=... barge=...` -- it started and heard the room
- `Lancer ready -- server at http://10.5.1.82:7870` -- it found the Mac
- `Turned toward speech at N rad` -- sound tracking is alive
