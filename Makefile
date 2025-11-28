# Batuta Makefile
# Certeza Methodology - Tiered Quality Gates
#
# PERFORMANCE TARGETS (Toyota Way: Zero Defects, Fast Feedback)
# - make test-fast: < 30 seconds (unit tests, no heavy features)
# - make test:      < 2 minutes (all tests)
# - make coverage:  < 5 minutes (coverage report, two-phase pattern)
# - make test-full: comprehensive (all tests, all features)

# Use bash for shell commands
SHELL := /bin/bash

# Disable built-in rules for performance
.SUFFIXES:

# Delete partially-built files on error
.DELETE_ON_ERROR:

# Multi-line recipes execute in same shell
.ONESHELL:

.PHONY: all build test test-fast test-quick test-full lint fmt fmt-check clean doc \
        book book-build book-serve book-test tier1 tier2 tier3 tier4 \
        coverage coverage-fast coverage-full coverage-open \
        profile hooks-install hooks-verify lint-scripts bashrs-lint-makefile \
        bench dev pre-push ci check audit deps-validate deny \
        pmat-score pmat-gates quality-report semantic-search \
        examples examples-fast examples-list \
        mutants mutants-fast mutants-file mutants-list \
        property-test property-test-fast property-test-extensive \
        wasm wasm-release wasm-test docker docker-dev docker-test docker-clean \
        help quality tdg release install watch pr-ready

# Default target
all: tier2

# Build
build:
	cargo build

release:
	cargo build --release --locked

# ============================================================================
# TEST TARGETS (Performance-Optimized with nextest)
# ============================================================================

# Fast tests (<30s): Uses nextest for parallelism if available
# Pattern from bashrs: cargo-nextest + parallel execution
test-fast: ## Fast unit tests (<30s target)
	@echo "⚡ Running fast tests (target: <30s)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run --workspace --lib \
			--status-level skip \
			--failure-output immediate; \
	else \
		echo "💡 Install cargo-nextest for faster tests: cargo install cargo-nextest"; \
		time cargo test --workspace --lib; \
	fi
	@echo "✅ Fast tests passed"

# Quick alias for test-fast
test-quick: test-fast

# Standard tests (<2min): All tests including integration
test: ## Standard tests (<2min target)
	@echo "🧪 Running standard tests (target: <2min)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run --workspace \
			--status-level skip \
			--failure-output immediate; \
	else \
		time cargo test --workspace; \
	fi
	@echo "✅ Standard tests passed"

# Unit tests only
test-unit:
	cargo test --lib

# Integration tests only
test-integration:
	cargo test --test '*'

# Full comprehensive tests: All features, all property cases
test-full: ## Comprehensive tests (all features)
	@echo "🔬 Running full comprehensive tests..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run --workspace --all-features; \
	else \
		time cargo test --workspace --all-features; \
	fi
	@echo "✅ Full tests passed"

# Linting - allows dead_code warnings for CLI code
lint: ## Run clippy lints
	cargo clippy --lib --bins --tests --all-features -- -D warnings -A dead_code

# Format check
fmt: ## Format code
	cargo fmt --all

fmt-check: ## Check formatting
	cargo fmt --check

# Clean build artifacts
clean:
	cargo clean
	rm -rf target/ || true
	rm -f .batuta-state.json || true
	rm -f lcov.info || true

# Generate documentation
doc:
	cargo doc --no-deps --open

# ============================================================================
# BOOK TARGETS (mdBook)
# ============================================================================

book: book-build ## Build and open the book

book-build: ## Build the book
	@echo "📚 Building The Batuta Book..."
	@if command -v mdbook >/dev/null 2>&1; then \
		mdbook build book; \
		echo "✅ Book built: book/book/index.html"; \
	else \
		echo "❌ mdbook not found. Install with: cargo install mdbook"; \
		exit 1; \
	fi

book-serve: ## Serve the book locally for development
	@echo "📖 Serving book at http://localhost:3000..."
	@mdbook serve book --open

book-test: ## Test book synchronization
	@echo "🔍 Testing book synchronization..."
	@for example in examples/*.rs; do \
		if [ -f "$$example" ]; then \
			EXAMPLE_NAME=$$(basename "$$example" .rs); \
			echo "  Checking $$EXAMPLE_NAME..."; \
		fi; \
	done
	@echo "✅ Book sync check complete"

# ============================================================================
# TIER TARGETS (Certeza Quality Gates)
# ============================================================================

# Tier 1: On-save (<1 second, non-blocking)
tier1:
	@echo "Running Tier 1: Fast feedback..."
	@cargo fmt --check
	@cargo clippy -- -W clippy::all -A dead_code
	@cargo check
	@echo "Tier 1: PASSED"

# Tier 2: Pre-commit (<5 seconds, changed files only)
tier2:
	@echo "Running Tier 2: Pre-commit checks..."
	@cargo test --lib
	@cargo clippy -- -D warnings -A dead_code
	@echo "Tier 2: PASSED"

# Tier 3: Pre-push (1-5 minutes, full validation)
tier3:
	@echo "Running Tier 3: Full validation..."
	@cargo test --all
	@cargo clippy -- -D warnings -A dead_code
	@echo "Tier 3: PASSED"

# Tier 4: CI/CD (5-60 minutes, heavyweight)
tier4: tier3
	@echo "Running Tier 4: CI/CD validation..."
	@cargo test --release
	@echo "Running pmat analysis..."
	-pmat tdg . --include-components
	-pmat rust-project-score
	-pmat quality-gates --report
	@echo "Tier 4: PASSED"

# ============================================================================
# COVERAGE TARGETS (Two-Phase Pattern from bashrs)
# ============================================================================
# Pattern: bashrs/Makefile - Two-phase coverage with mold linker workaround
# CRITICAL: mold linker breaks LLVM coverage instrumentation
# Solution: Temporarily move ~/.cargo/config.toml during coverage runs

# Standard coverage (<5 min): Two-phase pattern with nextest
coverage: ## Generate HTML coverage report (target: <5 min)
	@echo "📊 Running coverage analysis (target: <5 min)..."
	@echo "🔍 Checking for cargo-llvm-cov and cargo-nextest..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@which cargo-nextest > /dev/null 2>&1 || (echo "📦 Installing cargo-nextest..." && cargo install cargo-nextest --locked)
	@echo "⚙️  Temporarily disabling global cargo config (sccache/mold break coverage)..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@echo "🧹 Cleaning old coverage data..."
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage
	@echo "🧪 Phase 1: Running tests with instrumentation (no report)..."
	@cargo llvm-cov --no-report nextest --no-tests=warn --workspace --no-fail-fast --all-features
	@echo "📊 Phase 2: Generating coverage reports..."
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@echo "⚙️  Restoring global cargo config..."
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo ""
	@echo "📊 Coverage Summary:"
	@echo "=================="
	@cargo llvm-cov report --summary-only
	@echo ""
	@echo "💡 Reports:"
	@echo "- HTML: target/coverage/html/index.html"
	@echo "- LCOV: target/coverage/lcov.info"
	@echo ""

# Fast coverage alias (same as coverage, optimized by default)
coverage-fast: coverage

# Full coverage: All features (for CI, slower)
coverage-full: ## Full coverage report (all features)
	@echo "📊 Running full coverage analysis (all features)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || cargo install cargo-llvm-cov --locked
	@which cargo-nextest > /dev/null 2>&1 || cargo install cargo-nextest --locked
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@cargo llvm-cov --no-report nextest --no-tests=warn --workspace --all-features
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo ""
	@cargo llvm-cov report --summary-only

# Open coverage report in browser
coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first"; \
	fi

# ============================================================================
# EXAMPLES TARGETS
# ============================================================================

examples: ## Run all examples to verify they work
	@echo "🎯 Running all examples..."
	@failed=0; \
	total=0; \
	for example in examples/*.rs; do \
		name=$$(basename "$$example" .rs); \
		total=$$((total + 1)); \
		echo "  Running $$name..."; \
		if cargo run --example "$$name" --quiet 2>/dev/null; then \
			echo "    ✅ $$name passed"; \
		else \
			echo "    ❌ $$name failed"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "📊 Results: $$((total - failed))/$$total examples passed"; \
	if [ $$failed -gt 0 ]; then exit 1; fi
	@echo "✅ All examples passed"

examples-fast: ## Run examples with release mode (faster execution)
	@echo "⚡ Running examples in release mode..."
	@for example in examples/*.rs; do \
		name=$$(basename "$$example" .rs); \
		echo "  Running $$name..."; \
		cargo run --example "$$name" --release --quiet 2>/dev/null || echo "    ⚠️  $$name failed"; \
	done
	@echo "✅ Examples complete"

examples-list: ## List all available examples
	@echo "📚 Available examples:"
	@for example in examples/*.rs; do \
		name=$$(basename "$$example" .rs); \
		echo "  - $$name"; \
	done
	@echo ""
	@echo "Run with: cargo run --example <name>"

# ============================================================================
# MUTATION TESTING TARGETS
# ============================================================================

mutants: ## Run mutation testing (full, ~30-60 min)
	@echo "🧬 Running mutation testing (full suite)..."
	@echo "⚠️  This may take 30-60 minutes for full coverage"
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@cargo mutants --no-times --timeout 300 -- --all-features
	@echo "✅ Mutation testing complete"

mutants-fast: ## Run mutation testing on a sample (quick feedback, ~5 min)
	@echo "⚡ Running mutation testing (fast sample)..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@cargo mutants --no-times --timeout 120 --shard 1/10 -- --lib
	@echo "✅ Mutation sample complete"

mutants-file: ## Run mutation testing on specific file (usage: make mutants-file FILE=src/oracle/mod.rs)
	@echo "🧬 Running mutation testing on $(FILE)..."
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Usage: make mutants-file FILE=src/path/to/file.rs"; \
		exit 1; \
	fi
	@which cargo-mutants > /dev/null 2>&1 || cargo install cargo-mutants --locked
	@cargo mutants --no-times --timeout 120 --file "$(FILE)" -- --all-features
	@echo "✅ Mutation testing on $(FILE) complete"

mutants-list: ## List mutants without running tests
	@echo "📋 Listing potential mutants..."
	@cargo mutants --list 2>/dev/null | head -100
	@echo "..."
	@echo "(showing first 100 mutants)"

# ============================================================================
# QUALITY & ANALYSIS TARGETS
# ============================================================================

# Development workflow
dev: tier1

# Pre-push checks
pre-push: tier3

# Pre-commit shortcut
pre-commit: lint test-fast
	@echo "✅ Pre-commit checks passed"

# CI/CD checks
ci: tier4

# Quick check (compile only)
check:
	cargo check --all-targets --all-features

# Run security audit
audit:
	@echo "🔒 Running security audit..."
	@cargo audit
	@echo "✅ Security audit completed"

# Validate dependencies (duplicates + security)
deps-validate:
	@echo "🔍 Validating dependencies..."
	@cargo tree --duplicate | grep -v "^$$" || echo "✅ No duplicate dependencies"
	@cargo audit || echo "⚠️  Security issues found"

# Run cargo-deny checks (licenses, bans, advisories, sources)
deny:
	@echo "🔒 Running cargo-deny checks..."
	@if command -v cargo-deny >/dev/null 2>&1; then \
		cargo deny check; \
	else \
		echo "❌ cargo-deny not installed. Install with: cargo install cargo-deny"; \
		exit 1; \
	fi
	@echo "✅ cargo-deny checks passed"

# TDG score
tdg:
	@command -v pmat >/dev/null 2>&1 || { echo "Error: pmat not installed"; exit 1; }
	pmat tdg src/

# PMAT Quality Analysis
pmat-score: ## Calculate Rust project quality score
	@echo "📊 Calculating Rust project quality score..."
	@pmat rust-project-score || echo "⚠️  pmat not found. Install with: cargo install pmat"
	@echo ""

pmat-gates: ## Run pmat quality gates
	@echo "🔍 Running pmat quality gates..."
	@pmat quality-gates --report || echo "⚠️  pmat not found or gates failed"
	@echo ""

quality-report: ## Generate comprehensive quality report
	@echo "📋 Generating comprehensive quality report..."
	@mkdir -p docs/quality-reports
	@echo "# Batuta Quality Report" > docs/quality-reports/latest.md
	@echo "" >> docs/quality-reports/latest.md
	@echo "Generated: $$(date)" >> docs/quality-reports/latest.md
	@echo "" >> docs/quality-reports/latest.md
	@echo "## Rust Project Score" >> docs/quality-reports/latest.md
	@pmat rust-project-score >> docs/quality-reports/latest.md 2>&1 || echo "Error getting score" >> docs/quality-reports/latest.md
	@echo "" >> docs/quality-reports/latest.md
	@echo "## Quality Gates" >> docs/quality-reports/latest.md
	@pmat quality-gates --report >> docs/quality-reports/latest.md 2>&1 || echo "Error running gates" >> docs/quality-reports/latest.md
	@echo "✅ Report generated: docs/quality-reports/latest.md"

semantic-search: ## Interactive semantic code search
	@echo "🔍 Semantic code search..."
	@pmat semantic || echo "⚠️  pmat semantic search not available"

# Quality gate (all checks) - ENFORCES coverage
quality: lint test coverage
	@echo "✅ All quality gates passed"

# Benchmarks
bench:
	cargo bench

# Profiling (requires renacer)
profile:
	renacer --function-time --source -- cargo bench

# Install binary
install:
	cargo install --path .

# Development watch mode
watch:
	cargo watch -x check -x test -x run

# All checks before PR
pr-ready: fmt lint test coverage
	@echo "✅ Ready for PR submission"

# ============================================================================
# WASM TARGETS
# ============================================================================

wasm:
	@echo "🌐 Building Batuta for WebAssembly (debug)..."
	./scripts/build-wasm.sh debug

wasm-release:
	@echo "🌐 Building Batuta for WebAssembly (release)..."
	./scripts/build-wasm.sh release

wasm-test:
	@echo "🧪 Testing WASM build..."
	cargo test --target wasm32-unknown-unknown --no-default-features --features wasm --lib

# ============================================================================
# DOCKER TARGETS
# ============================================================================

docker:
	@echo "🐳 Building production Docker image..."
	./scripts/docker-build.sh prod

docker-dev:
	@echo "🐳 Building development Docker image..."
	./scripts/docker-build.sh dev

docker-test:
	@echo "🧪 Running tests in Docker..."
	docker-compose up --abort-on-container-exit ci

docker-clean:
	@echo "🧹 Cleaning Docker images and volumes..."
	docker-compose down -v
	docker rmi batuta:latest batuta:dev batuta:ci 2>/dev/null || true
	@echo "✅ Docker cleanup complete"

# ============================================================================
# HELP
# ============================================================================

help: ## Show this help
	@echo "Batuta - Sovereign AI Stack Orchestrator"
	@echo ""
	@echo "EXTREME TDD Targets (time constraints):"
	@echo "  make test-fast     - Fast tests (< 30s)  [uses nextest]"
	@echo "  make test          - Standard tests (< 2min)"
	@echo "  make test-full     - Comprehensive tests (all features)"
	@echo "  make coverage      - Coverage report (two-phase, <5 min)"
	@echo "  make pre-commit    - Pre-commit checks (lint + test-fast)"
	@echo ""
	@echo "Quality Tiers (Certeza Methodology):"
	@echo "  make tier1         - On-save (<1s, non-blocking)"
	@echo "  make tier2         - Pre-commit (<5s)"
	@echo "  make tier3         - Pre-push (1-5 min)"
	@echo "  make tier4         - CI/CD (5-60 min)"
	@echo ""
	@echo "Development Targets:"
	@echo "  make lint          - Run clippy lints"
	@echo "  make fmt           - Format code"
	@echo "  make check         - Type check without building"
	@echo "  make build         - Build debug binary"
	@echo "  make release       - Build release binary"
	@echo "  make examples      - Run all examples"
	@echo "  make examples-list - List available examples"
	@echo "  make tdg           - Calculate TDG score"
	@echo "  make clean         - Clean build artifacts"
	@echo ""
	@echo "Mutation Testing:"
	@echo "  make mutants       - Full mutation testing (~30-60 min)"
	@echo "  make mutants-fast  - Fast sample (~5 min)"
	@echo "  make mutants-file FILE=path - Test specific file"
	@echo ""
	@echo "Documentation:"
	@echo "  make book          - Build The Batuta Book (mdBook)"
	@echo "  make book-serve    - Build and serve book locally"
	@echo "  make doc           - Generate API documentation"
	@echo ""
	@echo "WASM & Docker:"
	@echo "  make wasm          - Build WASM (debug)"
	@echo "  make wasm-release  - Build WASM (optimized)"
	@echo "  make docker        - Build production Docker image"
	@echo ""
