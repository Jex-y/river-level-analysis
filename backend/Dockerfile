FROM lukemathwalker/cargo-chef:latest-rust-1.80.1 AS chef
WORKDIR /app
RUN apt update && apt install lld clang -y
RUN rustup toolchain install nightly

FROM chef AS planner
COPY ./src ./src
COPY ./Cargo.* .
# Compute a lock-like file for our project
RUN cargo chef prepare  --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build our project dependencies, not our application!
RUN cargo +nightly chef cook --release --recipe-path recipe.json
COPY ./src ./src
COPY ./Cargo.* .

RUN cargo build --release --bin backend

FROM gcr.io/distroless/cc-debian12 AS runtime

# COPY ./adc.json /gcp/creds.json
# ENV GOOGLE_APPLICATION_CREDENTIALS=/gcp/creds.json

COPY --from=builder /app/target/release/backend /
COPY ./config ./config

ENV RUST_LOG="error,backend=info"
ENV APP_ENV="prod"

EXPOSE 8080
CMD ["./backend"]
