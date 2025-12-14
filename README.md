# AgentFlow

AgentFlow is a high-performance, actor-based framework for building complex, distinct AI agent workflows in Rust. It leverages the **Actor Model** to provide inherent concurrency, decentralized control flow, and robust error handling.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Advanced Patterns](#advanced-patterns)
  - [Batching](#batching)
  - [Supervision](#supervision)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🔭 Overview

AgentFlow shifts away from centralized "flow" execution to a distributed network of independent actors. Each actor is an isolated unit of logic that communicates via asynchronous message passing.

**Why Actor Model?**
- **Concurrency**: Actors run in parallel by default (on Tokio).
- **Simplicity**: No complex shared locks (`Arc<Mutex>`) required for basic flows.
- **Resilience**: Errors are isolated; supervisors can restart failed actors without crashing the system.
- **Flexibility**: Define complex routing logic dynamically.

## 🌟 Key Features

- **Async Native**: Built on `tokio` for high-throughput asynchronous processing.
- **Declarative Wiring**: Connect actors using readable syntax: `&node1 - "action" >> &node2`.
- **Smart Defaults**: Minimal boilerplate. `prep`, `post`, and `name` have sensible default implementations.
- **Robustness**:
  - **Typed Errors**: Explicit `ActorError` handling.
  - **Automatic Retries**: Configure retries and backoff per actor.
  - **Supervision**: Hierarchical fault tolerance with `Supervisor` actors.
- **Batching**: Generic `Batcher` actor to group high-throughput messages.

## 📦 Installation

Add AgentFlow to your `Cargo.toml`:

```toml
[dependencies]
agentflow = "0.1.0"
async-trait = "0.1.68"
tokio = { version = "1.28.0", features = ["full"] }
```

## 🚀 Basic Usage

### 1. Define your Message and Actor

```rust
use agent_rs::{ActorLogic, NodeHandle, ActorError};
use async_trait::async_trait;

#[derive(Clone, Debug)]
struct MyMessage(String);

struct EchoActor;

#[async_trait]
impl ActorLogic<MyMessage> for EchoActor {
    // optional: fn name(), fn prep(), fn post() have defaults!
    
    async fn process(&mut self, msg: MyMessage) -> Result<Option<(String, MyMessage)>, ActorError> {
        println!("Received: {}", msg.0);
        Ok(Some(("next".to_string(), msg)))
    }
}
```

### 2. Wire and Run

```rust
#[tokio::main]
async fn main() {
    // Create Nodes
    // Logic, Buffer Size, Max Retries, Retry Wait
    let node1 = NodeHandle::new(EchoActor, 10, 3, std::time::Duration::from_millis(100));
    let node2 = NodeHandle::new(EchoActor, 10, 3, std::time::Duration::from_millis(100));

    // Connect them
    let _ = &node1 - "next" >> &node2;

    // Send Message
    node1.send_packet(MyMessage("Hello Actor World!".into())).await.unwrap();

    // Prevent main text from exiting immediately
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
}
```

## 🔄 Advanced Patterns

### Batching

Efficiently process high-volume streams by grouping messages.

```rust
use agent_rs::Batcher;

// Create a batcher that flushes every 10 items OR every 1 second
let batcher_logic = Batcher::new(
    10,                                // Batch Size
    Duration::from_secs(1),            // Timeout
    "flush".to_string(),               // Action to emit
    |items: Vec<MyMsg>| MyMsg::Batch(items) // Wrap items
);

let batcher_node = NodeHandle::new(batcher_logic, 100, 3, Duration::ZERO);
```

### Supervision

Automatically recover from failures using a `Supervisor`.

```rust
use agent_rs::{Supervisor, SupervisorPolicy};

// Factory to create a fresh instance of your actor
let factory = || Box::new(MyFlakyActor::new());

// Create Supervisor with Restart Policy
let supervised_logic = Supervisor::new(SupervisorPolicy::Restart, factory);
let node = NodeHandle::new(supervised_logic, 10, 3, Duration::from_millis(100));

// If MyFlakyActor crashes, Supervisor will restart it and retry the message.
```

## 📚 API Reference

### `ActorLogic<T>` Trait
The core interface for your business logic.
- `prep(msg)`: Prepare/Validate input (Default: pass-through).
- `process(msg)`: **Required**. Core logic execution. Returns `Option<(Action, T)>`.
- `post(result)`: Post-process output (Default: pass-through).
- `name()`: Actor name for logging (Default: Struct name via reflection).
- `tick_interval()` / `on_tick()`: Optional periodic background tasks.

### `NodeHandle<T>`
The handle allows you to control the actor.
- `send_packet(msg)`: Send a message to the actor.
- `clone()`: Cheaply cloneable for usage in multiple places.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
