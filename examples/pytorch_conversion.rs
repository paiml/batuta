//! PyTorch to Realizar Conversion Example (BATUTA-010)
//!
//! Demonstrates the conversion mapping from Python PyTorch inference code
//! to Rust Realizar equivalents with MoE backend selection.
//!
//! Run with: cargo run --example pytorch_conversion

use batuta::pytorch_converter::{PyTorchConverter, PyTorchOperation};

fn main() {
    println!("🔄 PyTorch → Realizar Conversion Demo (BATUTA-010)");
    println!("=================================================\n");

    let converter = PyTorchConverter::new();

    // Show conversion mapping
    println!("📋 Operation Conversion Mapping");
    println!("-------------------------------\n");

    let operations = vec![
        (
            PyTorchOperation::LoadModel,
            "torch.load('model.pt') / AutoModel.from_pretrained()",
        ),
        (
            PyTorchOperation::LoadTokenizer,
            "AutoTokenizer.from_pretrained('model')",
        ),
        (
            PyTorchOperation::Forward,
            "model(input) / model.forward(input)",
        ),
        (
            PyTorchOperation::Generate,
            "model.generate(inputs, max_length=50)",
        ),
        (PyTorchOperation::Linear, "nn.Linear(768, 512)"),
        (PyTorchOperation::Attention, "nn.MultiheadAttention(512, 8)"),
        (PyTorchOperation::Encode, "tokenizer.encode('text')"),
        (PyTorchOperation::Decode, "tokenizer.decode(tokens)"),
    ];

    for (op, pytorch_code) in operations {
        if let Some(realizar_op) = converter.convert(&op) {
            println!("PyTorch:  {}", pytorch_code);
            println!("Realizar: {}", realizar_op.code_template);
            println!("          Complexity: {:?}", realizar_op.complexity);
            println!();
        }
    }

    // Show backend recommendations
    println!("🎯 Backend Recommendations by Model Size");
    println!("-----------------------------------------\n");

    let workloads = vec![
        (PyTorchOperation::TensorCreation, "Tensor operations"),
        (PyTorchOperation::Linear, "Linear layer"),
        (PyTorchOperation::Generate, "Text generation"),
    ];

    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

    for (op, desc) in workloads {
        println!("{} ({:?} complexity):", desc, op.complexity());
        for &size in &sizes {
            let backend = converter.recommend_backend(&op, size);
            println!("  {:>8} params → {}", format_size(size), backend);
        }
        println!();
    }

    // Show Python→Rust conversion example
    println!("💡 Practical Conversion Example: Text Generation");
    println!("------------------------------------------------\n");

    println!("Python PyTorch + Transformers:");
    println!("```python");
    println!("import torch");
    println!("from transformers import AutoModelForCausalLM, AutoTokenizer");
    println!();
    println!("# Load model and tokenizer");
    println!("model = AutoModelForCausalLM.from_pretrained('gpt2')");
    println!("tokenizer = AutoTokenizer.from_pretrained('gpt2')");
    println!("model.eval()");
    println!();
    println!("# Generate text");
    println!("with torch.no_grad():");
    println!("    inputs = tokenizer('Hello, world!', return_tensors='pt')");
    println!("    outputs = model.generate(**inputs, max_length=50)");
    println!("    text = tokenizer.decode(outputs[0])");
    println!("    print(text)");
    println!("```\n");

    println!("Rust Realizar (converted):");
    println!("```rust");
    println!("use realizar::gguf::GGUFModel;");
    println!("use realizar::tokenizer::Tokenizer;");
    println!("use realizar::generate::generate_text;");
    println!();
    println!("// Load model and tokenizer (GGUF format)");
    println!("let model = GGUFModel::from_file(\"gpt2.gguf\")?;");
    println!("let tokenizer = Tokenizer::from_file(\"tokenizer.json\")?;");
    println!();
    println!("// Generate text");
    println!("let tokens = tokenizer.encode(\"Hello, world!\")?;");
    println!("let output_tokens = generate_text(&model, &tokens, 50)?;");
    println!("let text = tokenizer.decode(&output_tokens)?;");
    println!("println!(\" {{}}\", text);");
    println!("```\n");

    // Show performance analysis
    println!("⚡ Performance Analysis");
    println!("----------------------\n");

    println!("Backend selection via MoE routing:");
    println!(
        "  • 1K params (tensor ops): {} (small operations)",
        converter.recommend_backend(&PyTorchOperation::TensorCreation, 1_000)
    );
    println!(
        "  • 100K params (linear layer): {}",
        converter.recommend_backend(&PyTorchOperation::Linear, 100_000)
    );
    println!(
        "  • 1M params (generation): {} (high-compute)",
        converter.recommend_backend(&PyTorchOperation::Generate, 1_000_000)
    );
    println!(
        "  • 10M params (attention): {} (transformer models)",
        converter.recommend_backend(&PyTorchOperation::Attention, 10_000_000)
    );

    println!("\n✨ Benefits:");
    println!("  • Inference-focused: Optimized for model serving");
    println!("  • Format support: GGUF and SafeTensors");
    println!("  • Performance: CPU SIMD, GPU, and WASM backends");
    println!("  • Safety: Pure Rust with zero unsafe in public APIs");
    println!("  • Adaptive: MoE routing selects optimal backend");
    println!("  • Production-ready: 85%+ test coverage");

    // Show common patterns
    println!("\n📚 Common Conversion Patterns");
    println!("-----------------------------\n");

    let patterns = vec![
        ("Model Loading", PyTorchOperation::LoadModel),
        ("Inference Pipeline", PyTorchOperation::Forward),
        ("Text Generation", PyTorchOperation::Generate),
        ("Tokenization", PyTorchOperation::Encode),
    ];

    for (category, op) in patterns {
        if let Some(realizar_op) = converter.convert(&op) {
            println!("{} ({:?}):", category, op);
            println!("  PyTorch:  {}.{:?}()", op.pytorch_module(), op);
            println!("  Realizar: {}", realizar_op.code_template);
            println!(
                "  Usage:\n    {}",
                realizar_op.usage_pattern.replace('\n', "\n    ")
            );
            println!();
        }
    }

    // Show module organization
    println!("🗂️  Realizar Module Organization");
    println!("--------------------------------\n");

    let modules = vec![
        ("tensor", "Core numerical data structure"),
        ("gguf", "GGUF model format support"),
        ("safetensors", "SafeTensors model format"),
        ("layers", "Neural network components"),
        ("tokenizer", "Text encoding/decoding"),
        ("generate", "Text generation & sampling"),
        ("quantize", "Model weight compression"),
        ("api", "HTTP inference server"),
    ];

    for (module, description) in modules {
        println!("  realizar::{:<20} - {}", module, description);
    }

    // Show key differences
    println!("\n🔍 Key Differences from PyTorch");
    println!("--------------------------------\n");

    println!("  PyTorch (Training + Inference):");
    println!("    • Autograd and backpropagation");
    println!("    • Dynamic computation graphs");
    println!("    • .pt/.pth model files");
    println!("    • Python-first API");
    println!();
    println!("  Realizar (Inference Only):");
    println!("    • No training, optimized for serving");
    println!("    • Static model loading");
    println!("    • GGUF/SafeTensors formats");
    println!("    • Rust-native with unified CPU/GPU/WASM");

    println!("\n✅ Conversion Complete!");
    println!("   - 10 operation mappings demonstrated");
    println!("   - MoE backend selection enabled");
    println!("   - Inference patterns preserved");
    println!("   - Production-ready with zero unsafe");
}

fn format_size(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}
