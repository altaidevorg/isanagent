use agent_rs::{ActorLogic, NodeHandle, ActorError};
use async_trait::async_trait;
use tokio::time::{sleep, Duration};
use std::sync::{Arc, Mutex};

// Define the Data Packet
#[derive(Debug, Clone)]
struct ConversationState {
    context: String,
    history: Vec<String>,
}

// --- Guardrail Actor ---
struct Guardrail;

#[async_trait]
impl ActorLogic<ConversationState> for Guardrail {
    async fn process(&mut self, mut state: ConversationState) -> Result<Option<(String, ConversationState)>, ActorError> {
        println!("[Guardrail] Processing: {}", state.context);
        state.history.push(format!("User: {}", state.context));
        
        // Simulate LLM check
        sleep(Duration::from_millis(100)).await;
        
        let is_safe = true; // Hardcoded for meaningful flow
        if is_safe {
            println!("[Guardrail] Approved.");
            Ok(Some(("assist".to_string(), state)))
        } else {
            println!("[Guardrail] Blocked.");
            Ok(Some(("block".to_string(), state)))
        }
    }
}

// --- Response Actor ---
struct ResponseGenerator;

#[async_trait]
impl ActorLogic<ConversationState> for ResponseGenerator {
    async fn process(&mut self, mut state: ConversationState) -> Result<Option<(String, ConversationState)>, ActorError> {
        println!("[ResponseGenerator] Generating response for: {}", state.context);
        
        // Simulate Generation
        sleep(Duration::from_millis(200)).await;
        let response = "This is a generated response.".to_string();
        
        state.history.push(format!("AI: {}", response));
        println!("[ResponseGenerator] Done: {}", response);
        
        Ok(Some(("finish".to_string(), state)))
    }
}

// --- Main ---

#[tokio::main]
async fn main() {
    // Setup logging (optional)
    // env_logger::init();

    println!("Starting Actor System...");

    // Create Nodes
    let guardrail = NodeHandle::new(Guardrail, 10, 3, Duration::from_millis(100));
    
    let response = NodeHandle::new(ResponseGenerator, 10, 3, Duration::from_millis(100));
    
    // Wire them up
    // Guardrail - "assist" >> Response
    let _ = &guardrail - "assist" >> &response;
    // We may need to wire loopback or other paths in real-world applications.
    // Let's just do a simple one-pass for this demo
    
    // Verify wiring (wait a bit for async wiring tasks to complete)
    sleep(Duration::from_millis(50)).await;

    // Send initial packet
    let initial_state = ConversationState {
        context: "Hello Agent".to_string(),
        history: Vec::new(),
    };

    println!("Sending initial packet...");
    guardrail.send_packet(initial_state).await.expect("Failed to send");

    // Wait for processing (In a real app, we might wait for a specific signal or just keep running)
    sleep(Duration::from_secs(1)).await;
    
    println!("Exiting.");
}
