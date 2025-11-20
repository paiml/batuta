# Batuta Makefile
# EXTREME TDD workflow per sovereign-ai-spec.md

.PHONY: help test test-fast test-unit test-integration coverage build clean lint fmt check pre-commit examples tdg wasm wasm-release wasm-test docker docker-dev docker-test docker-clean book book-serve book-watch

# Default target
help:
	@echo "Batuta - Sovereign AI Stack Orchestrator"
	@echo ""
	@echo "EXTREME TDD Targets (time constraints):"
	@echo "  make test-fast     - Fast tests (< 5 min)  [current: ~0.3s]"
	@echo "  make pre-commit    - Pre-commit tests (< 30 sec)  [current: ~0.3s]"
	@echo "  make coverage      - Coverage report (< 10 min)"
	@echo ""
	@echo "Development Targets:"
	@echo "  make test          - Run all tests"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint          - Run clippy lints"
	@echo "  make fmt           - Format code"
	@echo "  make check         - Type check without building"
	@echo "  make build         - Build debug binary"
	@echo "  make release       - Build release binary"
	@echo "  make examples      - Run all examples"
	@echo "  make tdg           - Calculate TDG score"
	@echo "  make clean         - Clean build artifacts"
	@echo ""
	@echo "WASM Targets:"
	@echo "  make wasm          - Build WASM (debug)"
	@echo "  make wasm-release  - Build WASM (optimized)"
	@echo "  make wasm-test     - Test WASM build"
	@echo ""
	@echo "Docker Targets:"
	@echo "  make docker        - Build production Docker image"
	@echo "  make docker-dev    - Build development Docker image"
	@echo "  make docker-test   - Run tests in Docker"
	@echo "  make docker-clean  - Clean Docker images and volumes"
	@echo ""
	@echo "Documentation Targets:"
	@echo "  make book          - Build The Batuta Book (mdBook)"
	@echo "  make book-serve    - Build and serve book locally"
	@echo "  make book-watch    - Watch and rebuild book on changes"
	@echo ""
	@echo "Quality Gates:"
	@echo "  make quality       - Run all quality checks"

# EXTREME TDD: Pre-commit tests (< 30s constraint)
pre-commit: lint test-fast
	@echo "✅ Pre-commit checks passed (< 30s)"

# EXTREME TDD: Fast tests (< 5min constraint)
test-fast:
	@echo "🚀 Running fast test suite..."
	@time cargo test --quiet --all
	@echo "✅ Fast tests completed"

# EXTREME TDD: Coverage (< 10min constraint)
coverage:
	@echo "📊 Generating coverage report..."
	@command -v cargo-llvm-cov >/dev/null 2>&1 || { echo "Installing cargo-llvm-cov..."; cargo install cargo-llvm-cov; }
	@cargo llvm-cov --all-features --html
	@echo "✅ Coverage report: target/llvm-cov/html/index.html"

# Run all tests
test:
	cargo test --all

# Unit tests only
test-unit:
	cargo test --lib

# Integration tests only
test-integration:
	cargo test --test '*'

# Linting
lint:
	cargo clippy --all-targets --all-features -- -D warnings

# Formatting
fmt:
	cargo fmt --all

# Type check
check:
	cargo check --all-targets --all-features

# Build debug
build:
	cargo build

# Build release
release:
	cargo build --release --locked

# Run examples
examples:
	@echo "🎯 Backend Selection Demo"
	@cargo run --example backend_selection --quiet
	@echo ""
	@echo "🔄 Pipeline Demo"
	@cargo run --example pipeline_demo --quiet

# TDG score
tdg:
	@command -v pmat >/dev/null 2>&1 || { echo "Error: pmat not installed"; exit 1; }
	pmat tdg src/

# Quality gate (all checks)
quality: lint test coverage tdg
	@echo "✅ All quality gates passed"

# Clean
clean:
	cargo clean
	rm -rf target/
	rm -f .batuta-state.json

# Install binary
install:
	cargo install --path .

# Development watch mode
watch:
	cargo watch -x check -x test -x run

# Mutation testing (optional - takes longer)
mutants:
	@command -v cargo-mutants >/dev/null 2>&1 || { echo "Installing cargo-mutants..."; cargo install cargo-mutants; }
	cargo mutants --timeout 300

# Benchmark
bench:
	cargo bench

# Documentation
docs:
	cargo doc --no-deps --open

# All checks before PR
pr-ready: fmt lint test coverage
	@echo "✅ Ready for PR submission"

# WASM build (debug)
wasm:
	@echo "🌐 Building Batuta for WebAssembly (debug)..."
	./scripts/build-wasm.sh debug

# WASM build (release, optimized)
wasm-release:
	@echo "🌐 Building Batuta for WebAssembly (release)..."
	./scripts/build-wasm.sh release

# Test WASM build
wasm-test:
	@echo "🧪 Testing WASM build..."
	cargo test --target wasm32-unknown-unknown --no-default-features --features wasm --lib

# Docker build (production)
docker:
	@echo "🐳 Building production Docker image..."
	./scripts/docker-build.sh prod

# Docker build (development)
docker-dev:
	@echo "🐳 Building development Docker image..."
	./scripts/docker-build.sh dev

# Run tests in Docker
docker-test:
	@echo "🧪 Running tests in Docker..."
	docker-compose up --abort-on-container-exit ci

# Clean Docker artifacts
docker-clean:
	@echo "🧹 Cleaning Docker images and volumes..."
	docker-compose down -v
	docker rmi batuta:latest batuta:dev batuta:ci 2>/dev/null || true
	@echo "✅ Docker cleanup complete"

# Build The Batuta Book
book:
	@echo "📚 Building The Batuta Book..."
	@command -v mdbook >/dev/null 2>&1 || { echo "Error: mdbook not installed. Install with: cargo install mdbook"; exit 1; }
	mdbook build book
	@echo "✅ Book built: book/book/index.html"

# Build and serve book locally
book-serve:
	@echo "📖 Serving The Batuta Book..."
	@command -v mdbook >/dev/null 2>&1 || { echo "Error: mdbook not installed. Install with: cargo install mdbook"; exit 1; }
	@echo "Open http://localhost:3000 in your browser"
	mdbook serve book --open

# Watch and rebuild book on changes
book-watch:
	@echo "👀 Watching The Batuta Book..."
	@command -v mdbook >/dev/null 2>&1 || { echo "Error: mdbook not installed. Install with: cargo install mdbook"; exit 1; }
	mdbook watch book
