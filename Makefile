# Batuta Makefile
# EXTREME TDD workflow per sovereign-ai-spec.md

.PHONY: help test test-fast test-unit test-integration coverage build clean lint fmt check pre-commit examples tdg

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
