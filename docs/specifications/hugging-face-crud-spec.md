# Hugging Face Query Specification

**Version**: 1.0.0
**Status**: Draft
**Created**: 2026-01-12
**Authors**: Pragmatic AI Labs

## Executive Summary

This specification defines the query capabilities for Hugging Face ecosystem integration within Batuta. The primary use case is supporting educational content development for the "Next-Gen AI Development with Hugging Face" Coursera specialization (5 courses, 60 hours, 15 weeks).

The first iteration focuses exclusively on **Read/Query operations** - enabling comprehensive discovery of what's available in the Hugging Face ecosystem to inform course content creation.

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Use Case: Coursera Specialization Support](#2-use-case-coursera-specialization-support)
3. [Hugging Face Ecosystem Taxonomy](#3-hugging-face-ecosystem-taxonomy)
4. [Query Requirements](#4-query-requirements)
5. [Technical Architecture](#5-technical-architecture)
6. [API Design](#6-api-design)
7. [Work Items (pmat Tickets)](#7-work-items-pmat-tickets)
8. [100-Point Falsification Checklist](#8-100-point-falsification-checklist)
9. [Peer-Reviewed Citations](#9-peer-reviewed-citations)
10. [Appendices](#10-appendices)

---

## 1. Purpose and Scope

### 1.1 Problem Statement

Current Batuta HuggingFace integration (`batuta hf tree`) provides only a partial view of the ecosystem (~25 items). Course developers need comprehensive, queryable access to:

- **50+ HuggingFace offerings** (libraries, tools, services, formats)
- **Live Hub metadata** (700K+ models, 100K+ datasets, 300K+ spaces)
- **Course-aligned categorization** (what's needed for each module)

### 1.2 Scope

**In Scope (v1.0 - Query Only)**:
- Static catalog of all HuggingFace ecosystem components
- Live API queries to HuggingFace Hub
- Course module alignment metadata
- Export to JSON/Markdown for course planning

**Out of Scope (Future Iterations)**:
- Create operations (push models/datasets)
- Update operations (modify Hub assets)
- Delete operations (remove Hub assets)
- Authentication/private repository access

### 1.3 Success Criteria

1. Query any of 50+ HuggingFace ecosystem components
2. Retrieve metadata for Hub assets (models, datasets, spaces)
3. Filter by course module requirements
4. Export structured data for course planning
5. 100% falsification checklist compliance

---

## 2. Use Case: Coursera Specialization Support

### 2.1 Specialization Overview

| Attribute | Value |
|-----------|-------|
| Title | Next-Gen AI Development with Hugging Face |
| Courses | 5 |
| Duration | 60 hours / 15 weeks |
| Target Audience | Python developers with ML basics |
| Prerequisites | Python, PyTorch/TensorFlow basics, neural networks |

### 2.2 Course-to-Component Mapping

#### Course 1: Hugging Face Hub and Ecosystem Fundamentals

| Week | Components Required |
|------|---------------------|
| Week 1 | Hub (models, datasets, spaces), Model Cards, Licensing, Safetensors |
| Week 2 | Transformers (pipeline, AutoModel, AutoTokenizer), Tokenizers |
| Week 3 | Vision Transformers (ViT, CLIP, DINOv2), Whisper, LLaVA, BLIP |

**Query Requirements**:
```
batuta hf query --course 1 --week 1
batuta hf query --component "transformers.pipeline"
batuta hf query --task "image-classification"
```

#### Course 2: Fine-Tuning Transformers with Hugging Face

| Week | Components Required |
|------|---------------------|
| Week 1 | Datasets (load_dataset, map, filter), Data formats (CSV, JSON, Parquet) |
| Week 2 | Trainer API, TrainingArguments, Callbacks, TensorBoard, WandB |
| Week 3 | Evaluate (metrics), BLEU, ROUGE, push_to_hub |

**Query Requirements**:
```
batuta hf query --course 2 --week 2
batuta hf query --component "datasets.map"
batuta hf query --integration "wandb"
```

#### Course 3: Large Language Models with Hugging Face

| Week | Components Required |
|------|---------------------|
| Week 1 | Text Generation (sampling, chat templates), Generation Config |
| Week 2 | Sentence-Transformers, FAISS, RAG patterns |
| Week 3 | Structured output, JSON mode, Function calling, Outlines |

**Query Requirements**:
```
batuta hf query --course 3 --topic "rag"
batuta hf query --component "sentence-transformers"
batuta hf query --pattern "structured-output"
```

#### Course 4: Advanced Fine-Tuning: PEFT, RLHF, and Alignment

| Week | Components Required |
|------|---------------------|
| Week 1 | PEFT (LoRA, QLoRA), bitsandbytes, NF4 quantization |
| Week 2 | TRL (SFTTrainer), Instruction datasets (Alpaca, ShareGPT) |
| Week 3 | DPO, RLHF, PPO, Reward models, Preference data |

**Query Requirements**:
```
batuta hf query --course 4 --week 1
batuta hf query --component "peft.LoraConfig"
batuta hf query --technique "dpo"
```

#### Course 5: Production ML with Hugging Face

| Week | Components Required |
|------|---------------------|
| Week 1 | TGI (Text Generation Inference), PagedAttention, Tensor parallelism |
| Week 2 | Inference Endpoints, Gradio, Spaces |
| Week 3 | Optimum (ONNX, quantization), Transformers.js |

**Query Requirements**:
```
batuta hf query --course 5 --deployment
batuta hf query --component "tgi"
batuta hf query --runtime "browser"
```

### 2.3 Asset Count Requirements

The query system must support discovering assets for:

| Asset Type | Count | Query Support |
|------------|-------|---------------|
| Videos | 75 | By topic, duration, course |
| Readings | 25 | By component, depth |
| Labs | 40 | By hands-on component |
| Quizzes | 15 | By assessment topic |
| Discussions | 5 | By topic |

---

## 3. Hugging Face Ecosystem Taxonomy

### 3.1 Complete Component Registry

The following components MUST be queryable:

#### 3.1.1 Hub & Client Libraries

| Component | Description | Status |
|-----------|-------------|--------|
| `hub.models` | 700K+ ML models | Required |
| `hub.datasets` | 100K+ datasets | Required |
| `hub.spaces` | 300K+ demos | Required |
| `hub.papers` | Research papers | Required |
| `huggingface_hub` | Python Hub client | Required |
| `huggingface.js` | JavaScript libraries | Required |
| `tasks` | Task taxonomy | Required |
| `dataset-viewer` | Dataset API | Required |

#### 3.1.2 Deployment & Inference

| Component | Description | Status |
|-----------|-------------|--------|
| `inference-providers` | 10+ inference partners | Required |
| `inference-endpoints` | Dedicated deployment | Required |
| `tgi` | Text Generation Inference | Required |
| `tei` | Text Embeddings Inference | Required |
| `aws-dlcs` | AWS Deep Learning Containers | Required |
| `azure` | Microsoft Azure integration | Required |
| `gcp` | Google Cloud integration | Required |

#### 3.1.3 Core ML Libraries

| Component | Description | Status |
|-----------|-------------|--------|
| `transformers` | Model architectures | Required |
| `diffusers` | Diffusion models | Required |
| `datasets` | Dataset loading | Required |
| `transformers.js` | Browser ML | Required |
| `tokenizers` | Fast tokenization | Required |
| `evaluate` | Metrics library | Required |
| `timm` | Vision models | Required |
| `sentence-transformers` | Embeddings | Required |
| `kernels` | Compute kernels | Required |

#### 3.1.4 Training & Optimization

| Component | Description | Status |
|-----------|-------------|--------|
| `peft` | Parameter-efficient tuning | Required |
| `accelerate` | Distributed training | Required |
| `optimum` | Hardware optimization | Required |
| `aws-trainium` | AWS Trainium/Inferentia | Required |
| `tpu` | Google TPU support | Required |
| `trl` | Reinforcement learning | Required |
| `safetensors` | Safe serialization | Required |
| `bitsandbytes` | Quantization | Required |
| `lighteval` | LLM evaluation | Required |

#### 3.1.5 Collaboration & Extras

| Component | Description | Status |
|-----------|-------------|--------|
| `gradio` | ML demos | Required |
| `trackio` | Experiment tracking | Required |
| `smolagents` | Agent framework | Required |
| `lerobot` | Robotics AI | Required |
| `autotrain` | AutoML | Required |
| `chat-ui` | Chat frontend | Required |
| `leaderboards` | Custom leaderboards | Required |
| `argilla` | Data annotation | Required |
| `distilabel` | Synthetic data | Required |

#### 3.1.6 Community Resources

| Component | Description | Status |
|-----------|-------------|--------|
| `blog` | HuggingFace blog | Required |
| `learn` | Learning resources | Required |
| `discord` | Community Discord | Required |
| `forum` | Discussion forum | Required |

### 3.2 Model Format Support

| Format | Extension | Query Support |
|--------|-----------|---------------|
| Safetensors | `.safetensors` | Required |
| GGUF | `.gguf` | Required |
| ONNX | `.onnx` | Required |
| PyTorch | `.pt`, `.bin` | Required |

### 3.3 Task Taxonomy

| Category | Tasks |
|----------|-------|
| NLP | text-generation, text-classification, question-answering, translation, summarization, fill-mask, token-classification |
| Vision | image-classification, object-detection, image-segmentation, depth-estimation |
| Audio | automatic-speech-recognition, audio-classification, text-to-speech |
| Multimodal | image-to-text, visual-question-answering, document-question-answering |

---

## 4. Query Requirements

### 4.1 Functional Requirements

#### FR-001: Static Ecosystem Query

**Description**: Query the complete HuggingFace ecosystem catalog
**Priority**: P0 (Critical)
**Ticket**: `HF-QUERY-001`

```bash
# List all components
batuta hf catalog

# List by category
batuta hf catalog --category "training"

# Get component details
batuta hf catalog --component "peft"

# Export for course planning
batuta hf catalog --format json > hf-ecosystem.json
```

#### FR-002: Live Hub Search

**Description**: Search HuggingFace Hub for models, datasets, spaces
**Priority**: P0 (Critical)
**Ticket**: `HF-QUERY-002`

```bash
# Search models
batuta hf search models --task "text-generation" --limit 10

# Search datasets
batuta hf search datasets --task "question-answering"

# Search spaces
batuta hf search spaces --sdk "gradio"

# Filter by downloads
batuta hf search models --min-downloads 10000
```

#### FR-003: Asset Metadata Query

**Description**: Get detailed metadata for Hub assets
**Priority**: P0 (Critical)
**Ticket**: `HF-QUERY-003`

```bash
# Model info
batuta hf info model meta-llama/Llama-2-7b-hf

# Dataset info
batuta hf info dataset squad

# Space info
batuta hf info space gradio/chatbot
```

#### FR-004: Course Module Query

**Description**: Query components by course module alignment
**Priority**: P1 (High)
**Ticket**: `HF-QUERY-004`

```bash
# Get all components for Course 1
batuta hf course 1

# Get components for specific week
batuta hf course 1 --week 2

# Get lab requirements
batuta hf course 3 --labs
```

#### FR-005: Dependency Query

**Description**: Query component dependencies and relationships
**Priority**: P1 (High)
**Ticket**: `HF-QUERY-005`

```bash
# What does PEFT depend on?
batuta hf deps peft

# What depends on transformers?
batuta hf rdeps transformers

# Compatibility matrix
batuta hf compat --component trl --with peft
```

#### FR-006: Documentation Query

**Description**: Query documentation links and resources
**Priority**: P2 (Medium)
**Ticket**: `HF-QUERY-006`

```bash
# Get docs link
batuta hf docs transformers

# Get API reference
batuta hf docs transformers --api

# Get tutorials
batuta hf docs peft --tutorials
```

### 4.2 Non-Functional Requirements

#### NFR-001: Response Time

- Static catalog queries: < 100ms
- Live Hub queries: < 2s (with caching)
- Cache TTL: 15 minutes

#### NFR-002: Offline Support

- Static catalog must work offline
- Live queries gracefully degrade with cached data

#### NFR-003: Output Formats

- Text (default, colored terminal)
- JSON (machine-readable)
- Markdown (documentation)

---

## 5. Technical Architecture

### 5.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      batuta CLI                              │
├─────────────────────────────────────────────────────────────┤
│  batuta hf <command>                                         │
│    ├── catalog    (static ecosystem query)                   │
│    ├── search     (live Hub search)                          │
│    ├── info       (asset metadata)                           │
│    ├── course     (course alignment)                         │
│    ├── deps       (dependencies)                             │
│    └── docs       (documentation links)                      │
├─────────────────────────────────────────────────────────────┤
│                    HfQueryEngine                             │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ EcosystemCatalog│  │  HubApiClient   │                   │
│  │  (static data)  │  │  (live queries) │                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           │                    │                             │
│  ┌────────▼────────────────────▼────────┐                   │
│  │           ResponseCache              │                   │
│  │  (15-min TTL, content-addressable)   │                   │
│  └──────────────────────────────────────┘                   │
├─────────────────────────────────────────────────────────────┤
│                    CourseAlignmentIndex                      │
│  Course 1 → [hub, transformers, whisper, ...]               │
│  Course 2 → [datasets, trainer, evaluate, ...]              │
│  Course 3 → [sentence-transformers, outlines, ...]          │
│  Course 4 → [peft, trl, bitsandbytes, ...]                  │
│  Course 5 → [tgi, gradio, optimum, ...]                     │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Data Model

```rust
/// Hugging Face ecosystem component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfComponent {
    /// Unique identifier (e.g., "transformers", "peft")
    pub id: String,
    /// Display name
    pub name: String,
    /// Category (hub, library, deployment, training, collaboration)
    pub category: HfCategory,
    /// Short description
    pub description: String,
    /// Documentation URL
    pub docs_url: String,
    /// GitHub repository
    pub repo_url: Option<String>,
    /// PyPI package name
    pub pypi_name: Option<String>,
    /// npm package name
    pub npm_name: Option<String>,
    /// Dependencies on other components
    pub dependencies: Vec<String>,
    /// Course alignments
    pub courses: Vec<CourseAlignment>,
    /// Related components
    pub related: Vec<String>,
}

/// Course alignment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CourseAlignment {
    /// Course number (1-5)
    pub course: u8,
    /// Week number (1-3)
    pub week: u8,
    /// Lesson numbers
    pub lessons: Vec<String>,
    /// Asset types used (video, lab, reading)
    pub asset_types: Vec<AssetType>,
}

/// Hub asset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubAsset {
    /// Asset ID (e.g., "meta-llama/Llama-2-7b-hf")
    pub id: String,
    /// Asset type (model, dataset, space)
    pub asset_type: HubAssetType,
    /// Author/organization
    pub author: String,
    /// Downloads count
    pub downloads: u64,
    /// Likes count
    pub likes: u64,
    /// Tags
    pub tags: Vec<String>,
    /// Pipeline tag (task)
    pub pipeline_tag: Option<String>,
    /// Library (transformers, diffusers, etc.)
    pub library: Option<String>,
    /// License
    pub license: Option<String>,
    /// Last modified
    pub last_modified: String,
}
```

### 5.3 API Endpoints Used

| Endpoint | Purpose | Rate Limit |
|----------|---------|------------|
| `GET /api/models` | Search models | 1000/hour |
| `GET /api/datasets` | Search datasets | 1000/hour |
| `GET /api/spaces` | Search spaces | 1000/hour |
| `GET /api/models/{id}` | Model metadata | 1000/hour |
| `GET /api/datasets/{id}` | Dataset metadata | 1000/hour |

---

## 6. API Design

### 6.1 CLI Interface

```bash
# Ecosystem catalog commands
batuta hf catalog                           # Full catalog tree
batuta hf catalog --category <cat>          # Filter by category
batuta hf catalog --component <name>        # Component details
batuta hf catalog --search <query>          # Search catalog
batuta hf catalog --format json             # JSON output

# Hub search commands
batuta hf search models [OPTIONS]           # Search models
batuta hf search datasets [OPTIONS]         # Search datasets
batuta hf search spaces [OPTIONS]           # Search spaces

# Search options
  --task <task>                             # Filter by task
  --library <lib>                           # Filter by library
  --author <org>                            # Filter by author
  --license <license>                       # Filter by license
  --min-downloads <n>                       # Minimum downloads
  --min-likes <n>                           # Minimum likes
  --limit <n>                               # Result limit (default: 20)
  --sort <field>                            # Sort by field

# Asset info commands
batuta hf info model <id>                   # Model details
batuta hf info dataset <id>                 # Dataset details
batuta hf info space <id>                   # Space details

# Course alignment commands
batuta hf course <n>                        # Course n components
batuta hf course <n> --week <w>             # Specific week
batuta hf course <n> --labs                 # Lab components only
batuta hf course <n> --readings             # Reading components only

# Dependency commands
batuta hf deps <component>                  # What it depends on
batuta hf rdeps <component>                 # What depends on it
batuta hf compat <c1> --with <c2>           # Compatibility check

# Documentation commands
batuta hf docs <component>                  # Docs URL
batuta hf docs <component> --api            # API reference
batuta hf docs <component> --tutorials      # Tutorials
```

### 6.2 Rust API

```rust
use batuta::hf::{HfQueryEngine, HfCatalog, HubClient};

// Static catalog query
let catalog = HfCatalog::load()?;
let peft = catalog.get("peft")?;
println!("PEFT: {}", peft.description);

// Course alignment query
let course1_components = catalog.by_course(1)?;
let week2_components = catalog.by_course_week(1, 2)?;

// Live Hub search
let client = HubClient::new()?;
let models = client.search_models()
    .task("text-generation")
    .library("transformers")
    .min_downloads(10000)
    .limit(10)
    .execute()
    .await?;

// Asset metadata
let model = client.get_model("meta-llama/Llama-2-7b-hf").await?;
println!("Downloads: {}", model.downloads);
```

---

## 7. Work Items (pmat Tickets)

### 7.1 Epic: HF-QUERY

**Title**: Hugging Face Query Capabilities
**Description**: Implement comprehensive query interface for HuggingFace ecosystem
**Priority**: P0
**Estimate**: 40 story points

### 7.2 Stories

#### HF-QUERY-001: Ecosystem Catalog Implementation

**Title**: Implement static HuggingFace ecosystem catalog
**Priority**: P0
**Estimate**: 8 points
**Acceptance Criteria**:
- [ ] 50+ components registered with full metadata
- [ ] Category filtering (hub, library, deployment, training, collaboration)
- [ ] Component detail view with description, docs, dependencies
- [ ] JSON/Markdown export support
- [ ] Offline operation without network

**Subtasks**:
- HF-QUERY-001a: Define HfComponent data structure
- HF-QUERY-001b: Populate catalog with all components
- HF-QUERY-001c: Implement `batuta hf catalog` CLI
- HF-QUERY-001d: Add category filtering
- HF-QUERY-001e: Add export formats

#### HF-QUERY-002: Hub Search Implementation

**Title**: Implement live HuggingFace Hub search
**Priority**: P0
**Estimate**: 8 points
**Acceptance Criteria**:
- [ ] Model search with task/library/author filters
- [ ] Dataset search with task/size filters
- [ ] Space search with SDK filters
- [ ] Pagination support (limit, offset)
- [ ] Response caching (15-min TTL)

**Subtasks**:
- HF-QUERY-002a: Implement HubApiClient
- HF-QUERY-002b: Add model search endpoint
- HF-QUERY-002c: Add dataset search endpoint
- HF-QUERY-002d: Add space search endpoint
- HF-QUERY-002e: Implement response caching

#### HF-QUERY-003: Asset Metadata Query

**Title**: Implement Hub asset metadata retrieval
**Priority**: P0
**Estimate**: 5 points
**Acceptance Criteria**:
- [ ] Retrieve model metadata (downloads, likes, tags, license)
- [ ] Retrieve dataset metadata (size, features, splits)
- [ ] Retrieve space metadata (SDK, hardware, status)
- [ ] Model card parsing (description, usage, limitations)

**Subtasks**:
- HF-QUERY-003a: Implement model info endpoint
- HF-QUERY-003b: Implement dataset info endpoint
- HF-QUERY-003c: Implement space info endpoint
- HF-QUERY-003d: Parse model cards

#### HF-QUERY-004: Course Alignment Index

**Title**: Implement course module alignment queries
**Priority**: P1
**Estimate**: 5 points
**Acceptance Criteria**:
- [ ] Map all components to courses 1-5
- [ ] Map components to weeks within courses
- [ ] Filter by asset type (lab, video, reading)
- [ ] Generate course planning reports

**Subtasks**:
- HF-QUERY-004a: Define CourseAlignment structure
- HF-QUERY-004b: Populate course-component mapping
- HF-QUERY-004c: Implement `batuta hf course` CLI
- HF-QUERY-004d: Add report generation

#### HF-QUERY-005: Dependency Graph

**Title**: Implement component dependency queries
**Priority**: P1
**Estimate**: 5 points
**Acceptance Criteria**:
- [ ] Query forward dependencies (deps)
- [ ] Query reverse dependencies (rdeps)
- [ ] Compatibility matrix between components
- [ ] Transitive dependency resolution

**Subtasks**:
- HF-QUERY-005a: Build dependency graph
- HF-QUERY-005b: Implement deps command
- HF-QUERY-005c: Implement rdeps command
- HF-QUERY-005d: Add compatibility checks

#### HF-QUERY-006: Documentation Links

**Title**: Implement documentation link queries
**Priority**: P2
**Estimate**: 3 points
**Acceptance Criteria**:
- [ ] Official docs URLs for all components
- [ ] API reference links
- [ ] Tutorial links
- [ ] GitHub repository links

**Subtasks**:
- HF-QUERY-006a: Populate docs URLs
- HF-QUERY-006b: Implement docs command
- HF-QUERY-006c: Add --api and --tutorials flags

#### HF-QUERY-007: Test Suite

**Title**: Comprehensive test coverage for HF query
**Priority**: P0
**Estimate**: 5 points
**Acceptance Criteria**:
- [ ] Unit tests for catalog (95% coverage)
- [ ] Integration tests for Hub API (mocked)
- [ ] E2E tests for CLI commands
- [ ] Falsification checklist validation

**Subtasks**:
- HF-QUERY-007a: Unit tests for HfCatalog
- HF-QUERY-007b: Unit tests for HubClient
- HF-QUERY-007c: Integration tests with mocked API
- HF-QUERY-007d: CLI E2E tests

### 7.3 Ticket Summary

| Ticket | Title | Priority | Points | Dependencies |
|--------|-------|----------|--------|--------------|
| HF-QUERY-001 | Ecosystem Catalog | P0 | 8 | None |
| HF-QUERY-002 | Hub Search | P0 | 8 | None |
| HF-QUERY-003 | Asset Metadata | P0 | 5 | HF-QUERY-002 |
| HF-QUERY-004 | Course Alignment | P1 | 5 | HF-QUERY-001 |
| HF-QUERY-005 | Dependency Graph | P1 | 5 | HF-QUERY-001 |
| HF-QUERY-006 | Documentation Links | P2 | 3 | HF-QUERY-001 |
| HF-QUERY-007 | Test Suite | P0 | 5 | All above |
| **Total** | | | **39** | |

---

## 8. 100-Point Falsification Checklist

Based on the Popperian Falsification methodology and Toyota Production System principles, this checklist provides 100 verification points organized into 10 categories.

### 8.1 Data Completeness (10 points)

| # | Check | Falsification Condition |
|---|-------|------------------------|
| 1 | All 50+ HuggingFace components are registered | Missing any component from Section 3.1 |
| 2 | Each component has a unique identifier | Duplicate IDs exist |
| 3 | Each component has a description | Empty description field |
| 4 | Each component has a documentation URL | Missing or invalid URL |
| 5 | Each component has category assignment | Uncategorized component |
| 6 | Hub asset types are complete (model, dataset, space) | Missing asset type |
| 7 | Task taxonomy covers all HuggingFace tasks | Missing task type |
| 8 | Model formats are complete (safetensors, gguf, onnx, pt) | Missing format |
| 9 | Course alignment covers courses 1-5 | Missing course mapping |
| 10 | Dependencies are bidirectionally consistent | A→B but B doesn't list A as rdep |

### 8.2 API Correctness (10 points)

| # | Check | Falsification Condition |
|---|-------|------------------------|
| 11 | `batuta hf catalog` returns valid output | Command fails or empty output |
| 12 | `batuta hf catalog --category X` filters correctly | Returns components from other categories |
| 13 | `batuta hf catalog --component X` returns details | Returns wrong component or fails |
| 14 | `batuta hf search models` returns models | Returns datasets/spaces or fails |
| 15 | `batuta hf search datasets` returns datasets | Returns models/spaces or fails |
| 16 | `batuta hf search spaces` returns spaces | Returns models/datasets or fails |
| 17 | `batuta hf info model X` returns correct metadata | Wrong model or missing fields |
| 18 | `batuta hf course N` returns course N components | Returns wrong course components |
| 19 | `batuta hf deps X` returns dependencies | Missing or incorrect dependencies |
| 20 | `batuta hf docs X` returns valid URL | Invalid or broken URL |

### 8.3 Query Accuracy (10 points)

| # | Check | Falsification Condition |
|---|-------|------------------------|
| 21 | Task filter returns only matching tasks | Results include non-matching tasks |
| 22 | Library filter returns only matching library | Results include other libraries |
| 23 | Author filter returns only matching author | Results include other authors |
| 24 | License filter returns only matching license | Results include other licenses |
| 25 | min-downloads filter is respected | Results below threshold |
| 26 | min-likes filter is respected | Results below threshold |
| 27 | limit parameter is respected | More results than limit |
| 28 | Sort order is correct | Results not in specified order |
| 29 | Pagination works correctly | Duplicate/missing results across pages |
| 30 | Search query matches relevant results | Irrelevant results returned |

### 8.4 Course Alignment (10 points)

| # | Check | Falsification Condition |
|---|-------|------------------------|
| 31 | Course 1 components match specification | Missing: hub, transformers, whisper |
| 32 | Course 2 components match specification | Missing: datasets, trainer, evaluate |
| 33 | Course 3 components match specification | Missing: sentence-transformers, rag |
| 34 | Course 4 components match specification | Missing: peft, trl, bitsandbytes |
| 35 | Course 5 components match specification | Missing: tgi, gradio, optimum |
| 36 | Week 1 components are foundational | Advanced topics in week 1 |
| 37 | Week 3 components build on earlier weeks | Prerequisites not covered |
| 38 | Lab components are hands-on capable | Non-interactive component for lab |
| 39 | Reading components have documentation | No docs available |
| 40 | Asset type counts match specification | Incorrect counts |

### 8.5 Performance (10 points)

| # | Check | Falsification Condition |
|---|-------|------------------------|
| 41 | Static catalog query < 100ms | Query exceeds 100ms |
| 42 | Cached Hub query < 500ms | Query exceeds 500ms |
| 43 | Uncached Hub query < 2s | Query exceeds 2s |
| 44 | Cache TTL is 15 minutes | Cache expires too early/late |
| 45 | Cache invalidation works | Stale data after TTL |
| 46 | Parallel queries don't block | Sequential execution detected |
| 47 | Memory usage is bounded | Memory leak or excessive usage |
| 48 | Large result sets are paginated | OOM on large queries |
| 49 | Offline mode returns cached data | Fails without network |
| 50 | Rate limiting is respected | 429 errors from API |

### 8.6 Output Formats (10 points)

| # | Check | Falsification Condition |
|---|-------|------------------------|
| 51 | Text output is human-readable | Garbled or truncated output |
| 52 | JSON output is valid JSON | Parse error |
| 53 | Markdown output is valid Markdown | Rendering errors |
| 54 | Text output uses colors appropriately | Missing colors or wrong colors |
| 55 | JSON output includes all fields | Missing fields |
| 56 | Markdown tables are properly formatted | Misaligned columns |
| 57 | Output format flag works | Ignores --format flag |
| 58 | Default format is text | Unexpected default format |
| 59 | Piped output disables colors | ANSI codes in piped output |
| 60 | Unicode characters render correctly | Encoding issues |

### 8.7 Error Handling (10 points)

| # | Check | Falsification Condition |
|---|-------|------------------------|
| 61 | Invalid component ID returns helpful error | Cryptic error or crash |
| 62 | Invalid course number returns helpful error | Crash or silent failure |
| 63 | Network failure returns helpful error | Hang or crash |
| 64 | Rate limit returns helpful error | Cryptic 429 error |
| 65 | Invalid filter value returns helpful error | Crash or silent ignore |
| 66 | Empty search results returns helpful message | Error instead of empty |
| 67 | Malformed Hub response is handled | Crash on bad JSON |
| 68 | Timeout is handled gracefully | Hang on slow response |
| 69 | Partial failure returns partial results | All-or-nothing behavior |
| 70 | Error messages include remediation | Error without guidance |

### 8.8 Security (10 points)

| # | Check | Falsification Condition |
|---|-------|------------------------|
| 71 | No credentials in output | API keys/tokens exposed |
| 72 | No credentials in cache | Credentials persisted |
| 73 | HTTPS used for all API calls | HTTP used |
| 74 | Input is sanitized | Command injection possible |
| 75 | Output is sanitized | XSS in markdown output |
| 76 | Rate limiting prevents abuse | No rate limiting |
| 77 | Cache is not world-readable | Insecure permissions |
| 78 | No PII in logs | Personal data logged |
| 79 | Error messages don't leak internals | Stack traces exposed |
| 80 | Dependencies have no known CVEs | Vulnerable dependencies |

### 8.9 Documentation (10 points)

| # | Check | Falsification Condition |
|---|-------|------------------------|
| 81 | CLI help is accurate | Help doesn't match behavior |
| 82 | All commands are documented | Undocumented command |
| 83 | All flags are documented | Undocumented flag |
| 84 | Examples are provided | No examples |
| 85 | Examples are correct | Examples don't work |
| 86 | Error codes are documented | Undocumented error |
| 87 | API is documented | Missing API docs |
| 88 | Architecture is documented | Missing architecture docs |
| 89 | Course mapping is documented | Missing course docs |
| 90 | Changelog is maintained | No changelog |

### 8.10 Testing (10 points)

| # | Check | Falsification Condition |
|---|-------|------------------------|
| 91 | Unit test coverage ≥ 95% | Coverage below 95% |
| 92 | Integration tests exist | No integration tests |
| 93 | E2E tests exist | No E2E tests |
| 94 | Mocked API tests exist | Tests hit real API |
| 95 | Edge cases are tested | Missing edge case tests |
| 96 | Error paths are tested | Only happy path tested |
| 97 | Performance tests exist | No performance tests |
| 98 | Tests run in CI | Tests not in CI |
| 99 | Tests are deterministic | Flaky tests |
| 100 | Mutation testing score ≥ 80% | Score below 80% |

### 8.11 Falsification Summary

| Category | Points | Weight |
|----------|--------|--------|
| Data Completeness | 10 | 10% |
| API Correctness | 10 | 10% |
| Query Accuracy | 10 | 10% |
| Course Alignment | 10 | 10% |
| Performance | 10 | 10% |
| Output Formats | 10 | 10% |
| Error Handling | 10 | 10% |
| Security | 10 | 10% |
| Documentation | 10 | 10% |
| Testing | 10 | 10% |
| **Total** | **100** | **100%** |

**Passing Threshold**: 90/100 points (Toyota Standard)
**Kaizen Required**: 80-89 points
**Andon Warning**: 70-79 points
**Stop the Line**: < 70 points

---

## 9. Peer-Reviewed Citations

### 9.1 HuggingFace Technical Papers

1. **Wolf, T., et al. (2020)**. "Transformers: State-of-the-Art Natural Language Processing." *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pp. 38-45. ACL. DOI: 10.18653/v1/2020.emnlp-demos.6

   *Foundational paper on the transformers library architecture and design principles.*

2. **Lhoest, Q., et al. (2021)**. "Datasets: A Community Library for Natural Language Processing." *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pp. 175-184. ACL. DOI: 10.18653/v1/2021.emnlp-demo.21

   *Describes the datasets library design for efficient data loading and processing.*

3. **Mangrulkar, S., et al. (2022)**. "PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods." *arXiv preprint arXiv:2304.01933*.

   *Technical overview of parameter-efficient fine-tuning methods including LoRA.*

4. **von Werra, L., et al. (2023)**. "TRL: Transformer Reinforcement Learning." *GitHub Repository*. https://github.com/huggingface/trl

   *Reference implementation for RLHF and DPO training.*

5. **Hugging Face Team (2023)**. "Text Generation Inference." *Technical Documentation*. https://huggingface.co/docs/text-generation-inference

   *Production deployment architecture for LLM serving.*

6. **Reimers, N., & Gurevych, I. (2019)**. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. DOI: 10.18653/v1/D19-1410

   *Foundational paper for sentence-transformers library.*

7. **Raffel, C., et al. (2020)**. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *Journal of Machine Learning Research*, 21(140), 1-67.

   *T5 paper establishing transfer learning paradigms used in HuggingFace ecosystem.*

### 9.2 ML Education Methodology

8. **Ng, A. (2021)**. "Machine Learning Yearning." *deeplearning.ai*.

   *Practical guide for structuring ML education and project-based learning.*

9. **Géron, A. (2022)**. "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow." *O'Reilly Media*, 3rd Edition. ISBN: 978-1098125974.

   *Reference for progressive ML curriculum design from basics to production.*

10. **Chollet, F. (2021)**. "Deep Learning with Python." *Manning Publications*, 2nd Edition. ISBN: 978-1617296864.

    *Pedagogical approach to teaching deep learning concepts progressively.*

11. **Howard, J., & Gugger, S. (2020)**. "Deep Learning for Coders with fastai and PyTorch." *O'Reilly Media*. ISBN: 978-1492045526.

    *Top-down teaching methodology for ML education.*

12. **Bloom, B. S. (1956)**. "Taxonomy of Educational Objectives: The Classification of Educational Goals." *Longmans, Green*.

    *Foundation for learning outcome design in course curriculum.*

### 9.3 Software Engineering & API Design

13. **Fielding, R. T. (2000)**. "Architectural Styles and the Design of Network-based Software Architectures." *Doctoral dissertation*, University of California, Irvine.

    *REST principles for API design used in HuggingFace Hub API.*

14. **Gamma, E., et al. (1994)**. "Design Patterns: Elements of Reusable Object-Oriented Software." *Addison-Wesley*. ISBN: 978-0201633610.

    *Factory and Strategy patterns used in HuggingFace Auto classes.*

15. **Martin, R. C. (2008)**. "Clean Code: A Handbook of Agile Software Craftsmanship." *Prentice Hall*. ISBN: 978-0132350884.

    *Code quality principles applied to query implementation.*

16. **Fowler, M. (2018)**. "Refactoring: Improving the Design of Existing Code." *Addison-Wesley*, 2nd Edition. ISBN: 978-0134757599.

    *Refactoring patterns for evolving query capabilities.*

17. **Newman, S. (2021)**. "Building Microservices." *O'Reilly Media*, 2nd Edition. ISBN: 978-1492034025.

    *Service design principles for Hub API integration.*

### 9.4 Quantization & Optimization

18. **Dettmers, T., et al. (2022)**. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *arXiv preprint arXiv:2208.07339*.

    *Foundation for bitsandbytes INT8 quantization.*

19. **Dettmers, T., et al. (2023)**. "QLoRA: Efficient Finetuning of Quantized LLMs." *arXiv preprint arXiv:2305.14314*.

    *4-bit quantization with LoRA for efficient fine-tuning.*

20. **Frantar, E., et al. (2022)**. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *arXiv preprint arXiv:2210.17323*.

    *GPTQ quantization method supported by HuggingFace.*

### 9.5 Alignment & RLHF

21. **Ouyang, L., et al. (2022)**. "Training language models to follow instructions with human feedback." *arXiv preprint arXiv:2203.02155*.

    *InstructGPT paper establishing RLHF methodology.*

22. **Rafailov, R., et al. (2023)**. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *arXiv preprint arXiv:2305.18290*.

    *DPO paper for preference-based alignment without reward models.*

23. **Schulman, J., et al. (2017)**. "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*.

    *PPO algorithm used in TRL for RLHF training.*

### 9.6 Production Deployment

24. **Kwon, W., et al. (2023)**. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *arXiv preprint arXiv:2309.06180*.

    *vLLM PagedAttention technique used in TGI.*

25. **Pope, R., et al. (2022)**. "Efficiently Scaling Transformer Inference." *arXiv preprint arXiv:2211.05102*.

    *Scaling techniques for production LLM inference.*

---

## 10. Appendices

### Appendix A: Complete Component Catalog

See `src/hf/catalog.json` for the full component registry.

### Appendix B: Course-Component Mapping Matrix

| Component | C1W1 | C1W2 | C1W3 | C2W1 | C2W2 | C2W3 | C3W1 | C3W2 | C3W3 | C4W1 | C4W2 | C4W3 | C5W1 | C5W2 | C5W3 |
|-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| hub | X | | | | | X | | | | | | | | | |
| transformers | | X | X | | X | | X | | | | | | | | |
| datasets | | | | X | | | | | | | X | | | | |
| trainer | | | | | X | | | | | | X | | | | |
| evaluate | | | | | | X | | | | | | | | | |
| whisper | | | X | | | | | | | | | | | | |
| sentence-transformers | | | | | | | | X | | | | | | | |
| peft | | | | | | | | | | X | | | | | |
| trl | | | | | | | | | | | X | X | | | |
| bitsandbytes | | | | | | | | | | X | | | | | |
| tgi | | | | | | | | | | | | | X | | |
| gradio | | | | | | | | | | | | | | X | |
| optimum | | | | | | | | | | | | | | | X |
| transformers.js | | | | | | | | | | | | | | | X |

### Appendix C: API Rate Limits

| Endpoint | Limit | Window | Retry-After |
|----------|-------|--------|-------------|
| /api/models | 1000 | 1 hour | 3600s |
| /api/datasets | 1000 | 1 hour | 3600s |
| /api/spaces | 1000 | 1 hour | 3600s |
| /api/models/{id} | 1000 | 1 hour | 3600s |

### Appendix D: Glossary

| Term | Definition |
|------|------------|
| PEFT | Parameter-Efficient Fine-Tuning |
| LoRA | Low-Rank Adaptation |
| QLoRA | Quantized LoRA |
| DPO | Direct Preference Optimization |
| RLHF | Reinforcement Learning from Human Feedback |
| TGI | Text Generation Inference |
| TEI | Text Embeddings Inference |
| RAG | Retrieval-Augmented Generation |
| SFT | Supervised Fine-Tuning |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-12 | Pragmatic AI Labs | Initial specification |

---

*This specification is part of the Batuta Sovereign AI Stack documentation.*
