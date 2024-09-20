FROM lukemathwalker/cargo-chef:latest-rust-1.80.1 AS chef
WORKDIR /app
RUN apt update && apt install lld clang -y

FROM chef AS planner
COPY . .
# Compute a lock-like file for our project
RUN cargo chef prepare  --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build our project dependencies, not our application!
RUN rustup toolchain install nightly
RUN cargo +nightly chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release --bin backend

FROM debian:bookworm-slim AS runtime
WORKDIR /app
COPY --from=builder /app/target/release/backend /app/backend
COPY ./model /app/model
# ENV RUST_LOG="error,backend=info"
EXPOSE 8080
CMD ["/app/backend"]
