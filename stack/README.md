# Lancer stack

Four containers and one native process. The split is not arbitrary: everything
that needs Metal, low-latency audio, or the USB link to the robot stays on the
host, and everything else is in a container.

```
macOS host (M3 Max, 64 GB)
├─ Ollama (native, Metal)        qwen2.5:14b + embeddings
│                                containers reach it at host.docker.internal:11434
├─ Docker: lancer-net 10.251.251.0/24
│   ├─ zeroclaw   :8081   gateway, ws://localhost:8081/ws/chat?agent=lancer
│   ├─ db         :5433   postgres 16, memory + sessions + proposals
│   ├─ chroma     :8001   the CBU document index
│   └─ reflect            reads transcripts, proposes skills, applies nothing
└─ server.py (native)     voice I/O, retrieval, YOLO, robot control
```

## Bringing it up

```sh
cd stack
docker compose up -d
docker compose ps
```

The ZeroClaw image builds from source (`zeroclaw/Dockerfile`) and takes a while
the first time -- it is a Rust workspace. It is vanilla upstream: none of the
four WISS patches apply here, and dropping the WISS runtime stage's
`mssql-tools18` is what makes an arm64 build possible, since Microsoft ships no
arm64 packages for it.

Verified on this machine: builds clean for `linux/arm64`, reports
`zeroclaw 0.8.4`, `aarch64`.

## Checks worth running

```sh
# containers can see host Ollama -- the assumption the whole design rests on
docker run --rm --platform linux/arm64 curlimages/curl \
  -s http://host.docker.internal:11434/api/tags | head -c 200

# postgres is up and holding Lancer's memory
docker exec lancer-db psql -U lancer -d lancer -c '\dt'

# the gateway answers
curl -s localhost:8081/health
```

## Configuration

`zeroclaw/config.toml` is the gateway's whole configuration. The parts that
matter:

- `providers.models.llamacpp.default` points at Ollama's OpenAI-compatible `/v1`
  endpoint on the host. The block is named `llamacpp` because that is
  ZeroClaw's name for "an OpenAI-compatible server"; the server here is Ollama.
- `context_window` must match what Ollama actually serves. Unset, ZeroClaw
  assumes 32,000 and neither side complains.
- `risk_profiles.campus` is genuinely restricted, unlike the WISS profile it is
  modelled on. Lancer is a robot in a hallway that anyone can talk to.
- Swapping the model is one line here plus `ollama pull`. The 14B stays one
  edit away from any larger model, which is the rollback path.

Secrets belong in `.env` (untracked), not in `config.toml`.

## The reflection loop

`reflect` wakes on an interval, reads the week's turns out of Postgres, groups
the questions Lancer could not answer, and asks the model which skills would
have covered them. It writes a proposal to `stack/proposals/` and a row to
`skill_proposals`. It does not create skills -- the skills volume is mounted
read-only and there is no code path to writing one. A human reads the proposal
and decides.

That is the most interesting thing here to staff with ACM members, and the
restraint is what makes it safe to leave running.
