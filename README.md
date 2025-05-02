# AgentFlow

AgentFlow is a flexible, Rust-based execution framework for building AI agent workflows as directed graphs. It allows you to compose complex agent behaviors by connecting nodes conditionally based on their execution outcomes.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)

## 🔭 Overview

AgentFlow enables AI agent developers to:

- Define both synchronous and asynchronous processing nodes
- Build flexible execution flows using a declarative, operator-based syntax
- Conditionally route execution based on node results
- Share state between nodes in a type-safe manner
- Handle errors with automatic retries and fallbacks

The framework is particularly well-suited for AI systems that require complex, conditional processing logic with dependency management.

## 🧩 Key Concepts

### Nodes and Flows

- **Nodes**: Encapsulate individual processing steps with prep, exec, post lifecycle
- **Flows**: Organize nodes into directed graphs with conditional transitions
- **Actions**: String values returned by nodes that determine the next execution path

### Execution Lifecycle

Each node goes through a standard lifecycle:

1. **Prep**: Prepare resources and extract data from shared state
2. **Exec**: Perform main computation (with automatic retries)
3. **Fallback**: Execute if all exec retries fail
4. **Post**: Update shared state and determine next action

### State Management

- **SharedState**: User-defined type shared across all nodes in a flow
- **Params**: Configuration values passed to nodes

## 📦 Installation

Add AgentFlow to your Cargo.toml:

```toml
[dependencies]
agentflow = "0.1.0"  # Replace with actual version
async-trait = "0.1.68"
tokio = { version = "1.28.0", features = ["full"] }
```

## 🚀 Basic Usage

### 1. Define your shared state

```rust
use agent_rs::*;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
struct MyState {
    context: String,
    memory: HashMap<String, String>,
    conversation_history: Vec<String>,
}
```

### 2. Create node logic

Here's an example creating a guardrail node that checks if user questions are weather-related:

```rust
#[derive(Debug, Default)]
struct GuardrailNode;

impl SyncLogic<MyState, EmptyParams> for GuardrailNode {
    fn name(&self) -> String {
        "Guardrail".to_string()
    }

    fn prep(&mut self, shared: &mut MyState, _params: &EmptyParams) -> PrepResult {
        shared.conversation_history.push(shared.context.clone());
        // Generate a prompt based on state
        let prompt = format!(
            "User question: {}\nIs this related to weather? yes or no",
            shared.context
        );
        println!("{}: Prepared the prompt", self.name());
        Ok(Box::new(prompt))
    }

    fn exec(&mut self, prep_res: AnySendSync, _params: &EmptyParams) -> ExecResult {
        let prompt = prep_res.downcast_ref::<String>().unwrap();
        println!("{}: Executing with prompt: {:?}", self.name(), prompt);

        // In a real implementation, you would send this to an LLM
        let response = "yes".to_string();
        Ok(Box::new(response))
    }

    fn post(
        &mut self,
        shared: &mut MyState,
        _prep_res: AnySendSync,
        exec_res: ExecResult,
        _params: &EmptyParams,
    ) -> PostResult {
        match exec_res {
            Ok(response) => {
                let response_str = response.downcast_ref::<String>().unwrap();
                println!("{}: {:?}", self.name(), response_str);
                shared
                    .memory
                    .insert("relevant".to_string(), response_str.clone());

                // Determine next action based on response content
                let action = if response_str.contains("yes") {
                    "assist"
                } else {
                    "do_not_assist"
                };
                println!("{}: Next action: {:?}", self.name(), action);
                Ok(action.to_string())
            }
            Err(_) => Ok("error".to_string()),
        }
    }

    fn exec_fallback(
        &mut self,
        _prep_res: AnySendSync,
        error: FlowError,
        _params: &EmptyParams,
    ) -> ExecResult {
        println!("Fallback: {:?}", error);
        Ok(Box::new("Sorry, I couldn't process that.".to_string()))
    }
}
```

And a response node that generates answers to weather questions:

```rust
#[derive(Debug, Default)]
struct ResponseNode;

impl SyncLogic<MyState, EmptyParams> for ResponseNode {
    fn name(&self) -> String {
        "ResponseGenerator".to_string()
    }

    fn prep(&mut self, shared: &mut MyState, _params: &EmptyParams) -> PrepResult {
        // Generate a response based on state
        let prompt = format!(
            "You are a helpful assistant. Answer the following question wisely:\n{:?}",
            shared.context
        );
        println!("{}: Prepared the prompt", self.name());
        Ok(Box::new(prompt))
    }

    fn exec(&mut self, prep_res: AnySendSync, _params: &EmptyParams) -> ExecResult {
        let prompt = prep_res.downcast_ref::<String>().unwrap();
        println!("{}: Executing with prompt: {:?}", self.name(), prompt);

        // In a real implementation, you would send this to an LLM
        Ok(Box::new(
            "Here's a summary of your options: ...".to_string(),
        ))
    }

    fn post(
        &mut self,
        shared: &mut MyState,
        _prep_res: AnySendSync,
        exec_res: ExecResult,
        _params: &EmptyParams,
    ) -> PostResult {
        match exec_res {
            Ok(response) => {
                let response_str = response.downcast_ref::<String>().unwrap();
                shared.conversation_history.push(response_str.clone());
                Ok("finish".to_string())
            }
            Err(_) => Ok("error".to_string()),
        }
    }

    fn exec_fallback(
        &mut self,
        _prep_res: AnySendSync,
        error: FlowError,
        _params: &EmptyParams,
    ) -> ExecResult {
        println!("Fallback: {:?}", error);
        Ok(Box::new("Sorry, I could not process that.".to_string()))
    }
}
```

### 3. Create and connect nodes in a flow

```rust
fn main() {
    // Create nodes
    let guardrail_node = SyncNodeHandle::new(GuardrailNode, 2, 1).into_nodetype();
    let response_node = SyncNodeHandle::new(ResponseNode, 1, 0).into_nodetype();

    // Create flow
    let flow = Flow::<MyState, EmptyParams>::new("AgentConversationFlow");

    // Build the graph with conditional routing
    flow.start(guardrail_node.clone());

    // Route based on action strings
    let _ = guardrail_node.clone() - "assist" >> response_node.clone();
    let _ = guardrail_node.clone() - "do_not_assist" >> guardrail_node.clone();

    // Initialize state and run
    let mut state = MyState::default();
    state.context =
        "It's raining outside, but it'll be sunny in the afternoon. What should I wear today?"
            .to_string();

    // Execute the flow
    let result = flow.run(&mut state);
    println!("Flow completed with action: {:?}", result.unwrap());
    println!(
        "Final message: {:?}",
        state.conversation_history.last().unwrap()
    );
}
```

In this example:

1. We define a shared state (`MyState`) that holds conversation context, memory, and history.

2. We implement two nodes:
   - `GuardrailNode`: Checks if the user's question is weather-related
   - `ResponseNode`: Generates responses to weather-related questions

3. The flow logic:
   - We start with the `GuardrailNode` which analyzes the input
   - If the content is weather-related (action="assist"), it routes to the `ResponseNode`
   - Otherwise, it routes back to itself as a default action
   - The `ResponseNode` generates the final response and signals completion

4. We initialize the state with a weather-related question, run the flow, and display the results.

## 🔄 Advanced Usage

### Async Nodes

```rust
use agentflow::*;
use async_trait::async_trait;

#[derive(Debug, Default)]
struct ApiCallNode;

#[async_trait]
impl AsyncLogic<MyState, EmptyParams> for ApiCallNode {
    fn name(&self) -> String {
        "ApiCall".to_string()
    }
    
    async fn prep_async(&mut self, shared: &mut MyState, _params: &EmptyParams) -> PrepResult {
        // Prepare API request
        Ok(Box::new("https://api.example.com/data".to_string()))
    }
    
    async fn exec_async(&mut self, prep_res: AnySendSync, _params: &EmptyParams) -> ExecResult {
        let url = prep_res.downcast_ref::<String>().unwrap();
        
        // In a real implementation, make an actual API call
        // For example: let response = reqwest::get(url).await?;
        
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        Ok(Box::new("API response data".to_string()))
    }
    
    async fn post_async(
        &mut self,
        shared: &mut MyState,
        _prep_res: AnySendSync,
        exec_res: ExecResult,
        _params: &EmptyParams,
    ) -> PostResult {
        if let Ok(data) = exec_res {
            let response = data.downcast_ref::<String>().unwrap();
            shared.memory.insert("api_result".to_string(), response.clone());
            Ok("success".to_string())
        } else {
            Ok("failure".to_string())
        }
    }
    
    async fn exec_fallback_async(
        &mut self,
        _prep_res: AnySendSync,
        _error: FlowError,
        _params: &EmptyParams,
    ) -> ExecResult {
        Ok(Box::new("Fallback response".to_string()))
    }
}
```

### Using Async Flows

```rust
#[tokio::main]
async fn main() {
    // Create nodes
    let prompt_node = SyncNodeHandle::new(PromptNode, 2, 1).into_nodetype();
    let api_call = AsyncNodeHandle::new(ApiCallNode, 3, 2).into_nodetype();
    
    // Create async flow
    let flow = AsyncFlow::<MyState, EmptyParams>::new("AsyncAgentFlow");
    
    // Build graph
    flow.start(prompt_node.clone());
    let _ = prompt_node.clone() - "default" >> api_call.clone();
    
    // Initialize state
    let mut state = MyState::default();
    
    // Run async flow
    let result = flow.run_async(&mut state).await;
    println!("Async flow completed with action: {:?}", result);
}
```

### Using Custom Parameters

```rust
#[derive(Clone, Debug, Default)]
struct AgentParams {
    temperature: f32,
    max_tokens: usize,
    model: String,
}

// Then use it with nodes:
let llm_node = SyncNodeHandle::<MyState, AgentParams>::new(LlmNode, 2, 1).into_nodetype();

// Set parameters for specific node
let params = AgentParams {
    temperature: 0.7,
    max_tokens: 512,
    model: "gpt-4".to_string(),
};

// Apply params before running
let flow = Flow::<MyState, AgentParams>::new("ParameterizedFlow");
flow.start(llm_node.clone());
flow.set_params(params);

// Run flow
let result = flow.run(&mut state);
```

### Nesting Flows

```rust
// Create a sub-flow
let sub_flow = Flow::<MyState, EmptyParams>::new("SubFlow");
sub_flow.start(some_node.clone());

// Add to parent flow
let parent_flow = Flow::<MyState, EmptyParams>::new("ParentFlow");
parent_flow.start(entry_node.clone());

// Connect from parent to sub-flow
let _ = entry_node.clone() - "process" >> sub_flow.into_nodetype();
```

## 📚 API Reference

### Core Types

- `AnySendSync`: Box-wrapped type for flexible data passing
- `FlowError`: Error types for different failure modes
- `SharedState` and `Params`: Traits for user-defined state and parameters
- `SyncLogic` and `AsyncLogic`: Traits for implementing node logic
- `Flow` and `AsyncFlow`: Containers for organizing nodes

### Node Methods

- `prep`/`prep_async`: Prepare for execution
- `exec`/`exec_async`: Execute main logic
- `post`/`post_async`: Process results and determine next action
- `exec_fallback`/`exec_fallback_async`: Handle failures

### Flow Methods

- `new`: Create a new flow
- `start`: Set the starting node
- `run`/`run_async`: Execute the flow

### Operators

- `-`: Create a conditional transition with a specific action
- `>>`: Connect nodes together

## 🌟 Examples

### Build a Weather Guidance Agent

```rust
// Define state for our weather guidance agent
#[derive(Clone, Debug, Default)]
struct WeatherState {
    user_question: String,
    is_weather_related: bool,
    conversation_history: Vec<String>,
}

// Define node for checking if query is weather-related
#[derive(Debug, Default)]
struct GuardrailNode;

impl SyncLogic<WeatherState, EmptyParams> for GuardrailNode {
    fn name(&self) -> String {
        "WeatherGuardrail".to_string()
    }

    fn prep(&mut self, shared: &mut WeatherState, _params: &EmptyParams) -> PrepResult {
        // Save user query to history
        shared.conversation_history.push(shared.user_question.clone());
        
        // Prepare prompt that checks if query is weather-related
        let prompt = format!(
            "User question: {}\nIs this related to weather? yes or no",
            shared.user_question
        );
        Ok(Box::new(prompt))
    }

    fn exec(&mut self, prep_res: AnySendSync, _params: &EmptyParams) -> ExecResult {
        let prompt = prep_res.downcast_ref::<String>().unwrap();
        
        // In a real implementation, send to LLM to check weather relevance
        // For example: openai.chat.completions.create({ messages: [{ role: "user", content: prompt }] })
        
        // Simulated response for demo
        let is_weather = true;
        Ok(Box::new(if is_weather { "yes" } else { "no" }.to_string()))
    }

    fn post(
        &mut self,
        shared: &mut WeatherState,
        _prep_res: AnySendSync,
        exec_res: ExecResult,
        _params: &EmptyParams,
    ) -> PostResult {
        match exec_res {
            Ok(response) => {
                let response_str = response.downcast_ref::<String>().unwrap();
                
                // Update state based on result
                shared.is_weather_related = response_str.contains("yes");
                
                // Return appropriate action for routing
                if shared.is_weather_related {
                    Ok("provide_weather_guidance".to_string())
                } else {
                    Ok("not_weather_related".to_string())
                }
            }
            Err(_) => Ok("error".to_string()),
        }
    }
}

// Define node that provides weather guidance
#[derive(Debug, Default)]
struct WeatherResponseNode;

impl SyncLogic<WeatherState, EmptyParams> for WeatherResponseNode {
    fn name(&self) -> String {
        "WeatherResponse".to_string()
    }

    fn prep(&mut self, shared: &mut WeatherState, _params: &EmptyParams) -> PrepResult {
        // Generate prompt for weather-specific response
        let prompt = format!(
            "As a weather expert, answer this question: {}",
            shared.user_question
        );
        Ok(Box::new(prompt))
    }

    fn exec(&mut self, prep_res: AnySendSync, _params: &EmptyParams) -> ExecResult {
        let prompt = prep_res.downcast_ref::<String>().unwrap();
        
        // In a real implementation, send to LLM to generate weather advice
        // Simulate response for demo
        let response = "For rainy morning and sunny afternoon, layered clothing works best. \
                       Start with a waterproof jacket in the morning and remove it later. \
                       Don't forget an umbrella!".to_string();
        
        Ok(Box::new(response))
    }

    fn post(
        &mut self,
        shared: &mut WeatherState,
        _prep_res: AnySendSync,
        exec_res: ExecResult,
        _params: &EmptyParams,
    ) -> PostResult {
        match exec_res {
            Ok(response) => {
                let response_str = response.downcast_ref::<String>().unwrap();
                // Store response in conversation history
                shared.conversation_history.push(response_str.clone());
                Ok("complete".to_string())
            }
            Err(_) => Ok("error".to_string()),
        }
    }
}

// Create and run the weather guidance flow
fn main() {
    // Create nodes
    let guardrail = SyncNodeHandle::new(GuardrailNode, 2, 1).into_nodetype();
    let weather_response = SyncNodeHandle::new(WeatherResponseNode, 1, 0).into_nodetype();
    let general_response = SyncNodeHandle::new(GeneralResponseNode, 1, 0).into_nodetype();
    
    // Create flow
    let flow = Flow::<WeatherState, EmptyParams>::new("WeatherGuidanceFlow");
    
    // Build graph with conditional routing
    flow.start(guardrail.clone());
    let _ = guardrail.clone() - "provide_weather_guidance" >> weather_response.clone();
    let _ = guardrail.clone() - "not_weather_related" >> general_response.clone();
    
    // Initialize state with user question
    let mut state = WeatherState::default();
    state.user_question = "It's raining now but will be sunny later. What should I wear today?".to_string();
    
    // Run the flow
    let result = flow.run(&mut state);
    println!("Flow completed with final action: {:?}", result.unwrap());
    println!("Final response: {:?}", state.conversation_history.last().unwrap());
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with ❤️ for AI agent developers.
