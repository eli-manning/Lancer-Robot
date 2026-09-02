# ZeroClaw gateway, built for arm64 (Apple Silicon).
#
# Deliberately not the WISS image. That one carries four local patches and a
# runtime stage that installs mssql-tools18 from packages.microsoft.com, which
# publishes no arm64 packages -- the single hard reason the WISS image cannot be
# reused here. Lancer needs none of it: no SQL Server, no WISS app command
# relay, no Anthropic thinking stream (the provider is a local Ollama). So this
# builds vanilla upstream and keeps the runtime stage to what a campus robot
# actually needs, which also removes the arm64 blocker.
FROM rust:1-bookworm AS build
ARG ZEROCLAW_CLONE_CACHEBUST=unset
RUN echo "cachebust=$ZEROCLAW_CLONE_CACHEBUST" \
 && git config --global http.postBuffer 524288000 \
 && for i in 1 2 3 4 5 6; do \
      git clone --depth 1 https://github.com/zeroclaw-labs/zeroclaw.git /src && break \
      || { echo "clone attempt $i failed, retrying..."; rm -rf /src; sleep 5; }; \
    done && test -d /src
WORKDIR /src
RUN cargo build --release --locked --features memory-postgres

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl jq python3 python3-requests \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/zeroclaw /usr/local/bin/zeroclaw
RUN useradd -m -u 1000 zeroclaw && mkdir -p /home/zeroclaw/.zeroclaw \
 && chown -R zeroclaw:zeroclaw /home/zeroclaw/.zeroclaw
USER zeroclaw
WORKDIR /home/zeroclaw
EXPOSE 8081
ENTRYPOINT ["zeroclaw", "gateway", "start", "--allow-degraded-security"]
