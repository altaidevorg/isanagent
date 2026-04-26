# ---------- builder ----------
    FROM rust:1.90-bookworm AS builder

    WORKDIR /app
    
    # install deps often needed by crates (adjust if not needed)
    RUN apt-get update && apt-get install -y \
        pkg-config \
        libssl-dev \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*
    
    # cache dependencies
    COPY Cargo.toml Cargo.lock ./
    RUN cargo fetch
    
    # build actual project
    COPY . .
    RUN cargo build --release
    
    # ---------- runtime ----------
    FROM debian:bookworm-slim AS runtime
    
    # minimal runtime deps (openssl + certs commonly needed)
    RUN apt-get update && apt-get install -y \
        ca-certificates \
        libssl3 \
        && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    # copy binary
    COPY --from=builder /app/target/release/isanagent /usr/local/bin/isanagent
    
    # non-root (optional but good)
    # ARG UID=1000
    # RUN useradd -u $UID -m appuser
    
    # RUN mkdir -p /app && chown appuser:appuser /app
    # USER appuser
    
    ENTRYPOINT ["isanagent"]