# SynapseDSL: Event-Driven Neuromorphic Graph Architecture

**SynapseDSL** is an experimental hardware-aware domain-specific language (DSL) and sparse graph execution engine designed to replace brute-force dense matrix multiplication ($O(N^2)$) with asynchronous, biologically inspired event-driven computation.

---

## 1. The Core Problem

Current deep learning architectures (Transformers, dense MLPs) are trapped in a hardware-software local optimum:
- **Compute/Memory Bottleneck:** Dense tensor operations move massive blocks of inactive weights between DRAM and compute units on every clock cycle.
- **Zero Sparsity Exploitation:** High-level frameworks (PyTorch/CUDA) optimize for uniform dense algebra rather than dynamic, conditional signal routing.
- **Energy Scaling Wall:** While the human brain operates at ~20W using asynchronous spike/event routing, modern LLMs require megawatts of continuous power.

## 2. Our Solution: Sparse Graph + Hardware-Aware DSL

SynapseDSL treats intelligence not as static layers of matrices, but as a **dynamic directed graph of stateful micro-nodes** operating directly near memory:

```text
[Input Pulse] ──► (Node A: V_m >= Threshold) ──► Fires Sparse Event ──► (Node B: Updates State)
                         │
                  (Node C: V_m < Threshold)  ──► 0 FLOPs (Remains Dormant in SRAM)
```

### Key Architectural Primitives
1. **Asynchronous Threshold Activation:** Nodes consume zero compute cycles unless incoming weighted pulses exceed their membrane threshold ($\theta$).
2. **Near-Memory Graph Layout:** The DSL compiles graph topologies into compact Compressed Sparse Row (CSR) and custom ring-buffer layouts to maximize CPU/SRAM cache locality.
3. **Symbolic & Plastic Routing:** Synaptic weights decay, strengthen (Hebbian-like local plasticity), or reroute dynamically during inference without global backpropagation overhead.

---

## 3. Language Syntax Preview (`example.syn`)

SynapseDSL provides low-level control over memory alignment, membrane decay, and event propagation:

```rust
// Allocate graph cluster directly in fast local cache line
@memory(layout = "csr_aligned", target = "l1_cache")
cluster VisualCortex {
    state potential: f16 = 0.0;
    const threshold: f16 = 0.75;
    const decay_rate: f16 = 0.92;

    // Event-driven trigger: executes ONLY when threshold is crossed
    on_pulse(signal: f16) -> event {
        potential = (potential * decay_rate) + signal;
        if potential >= threshold {
            potential = 0.0; // Refractory reset
            emit_sparse( current_node_id, strength = 1.0 );
        }
    }
}
```

---

## 4. Proof-of-Concept Benchmark (`sim_core.py`)

We include a lightweight mathematical simulation (`sim_core.py`) comparing standard dense layer execution against our sparse event-driven graph routing.

To run the benchmark locally:
```bash
python3 sim_core.py
```

**Typical Output (1,024-node graph across 100 time steps):**
- **Dense Matrix Execution:** `104,857,600` total multiply-accumulate (MAC) operations.
- **SynapseDSL Sparse Graph:** `8,412,320` active event operations (**~91.9% compute reduction** with zero accuracy degradation on sparse temporal patterns).

---

## 5. 90-Day Engineering Roadmap

- [x] Mathematical specification of sparse threshold state transitions & local decay rules.
- [x] Baseline Python simulation demonstrating >85% MAC operation reduction (`sim_core.py`).
- [ ] **Phase 1 (Month 1):** Complete the custom Lexer and AST Parser for `.syn` files in C/Rust.
- [ ] **Phase 2 (Month 2):** Implement the CSR memory allocator and zero-copy event queue engine.
- [ ] **Phase 3 (Month 3):** Benchmark inference latency and power draw on edge ARM/RISC-V hardware against quantized ONNX/GGUF baselines.

---

## License
MIT License — Open research for decentralized, low-power intelligence.


