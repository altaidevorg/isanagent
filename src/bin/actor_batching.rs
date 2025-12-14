use agent_rs::{ActorLogic, NodeHandle, ActorError};
use async_trait::async_trait;
use tokio::time::{sleep, Duration};

// --- Message Types ---
// In a real batched flow, we often mix single items and batches.
// We use an Enum to represent both.
#[derive(Debug, Clone)]
enum AppMsg {
    Request(String),
    Batch(Vec<String>),
    Result(String),
}

// --- Processor Actor ---
// Processes a Batch and emits results (or done).
struct Processor;

#[async_trait]
impl ActorLogic<AppMsg> for Processor {
    async fn process(&mut self, packet: AppMsg) -> Result<Option<(String, AppMsg)>, ActorError> {
        match packet {
            AppMsg::Batch(items) => {
                println!("[Processor] received batch of {} items.", items.len());
                sleep(Duration::from_millis(500)).await; // Simulate heavy batched work
                
                let result_str = format!("Processed: {:?}", items);
                println!("[Processor] {}", result_str);
                
                Ok(Some(("done".to_string(), AppMsg::Result(result_str))))
            }
            _ => Err(ActorError::from("Processor received non-batch message")),
        }
    }
}

// --- Sink Actor ---
// Just prints the result
struct Sink;

#[async_trait]
impl ActorLogic<AppMsg> for Sink {
    async fn process(&mut self, packet: AppMsg) -> Result<Option<(String, AppMsg)>, ActorError> {
        match packet {
            AppMsg::Result(res) => {
                println!("[Sink] Final Result: {}", res);
                Ok(None) // End of line
            }
            _ => Err(ActorError::from("Sink received unexpected message")),
        }
    }
}

// --- Main ---

#[tokio::main]
async fn main() {
    println!("Starting Batching Demo...");

    // Create Nodes
    // Use the generic Batcher from the library
    // batch_size: 3, timeout: 2 seconds
    let batcher_logic = agent_rs::Batcher::new(
        3, 
        Duration::from_secs(2), 
        "process_batch".to_string(), 
        |items| {
            // We need to extract the strings from AppMsg::Request if possible,
            // OR the Batcher accumulates AppMsg directly. 
            // The GenericBatcher accumulates T (AppMsg).
            // So items is Vec<AppMsg>. We need to produce AppMsg::Batch(Vec<String>).
            
            // Extract strings from requests
            let strings: Vec<String> = items
                .into_iter()
                .filter_map(|msg| match msg {
                    AppMsg::Request(s) => Some(s),
                    _ => None,
                })
                .collect();
            println!("[GenericBatcher] Flushed {} items.", strings.len());
            AppMsg::Batch(strings)
        }
    );
    let batcher = NodeHandle::new(batcher_logic, 100, 3, Duration::from_millis(100));
    let processor = NodeHandle::new(Processor, 100, 3, Duration::from_millis(100));
    let sink = NodeHandle::new(Sink, 100, 3, Duration::from_millis(100));

    // Wire: Batcher - "process_batch" >> Processor - "done" >> Sink
    let _ = &batcher - "process_batch" >> &processor;
    let _ = &processor - "done" >> &sink;

    // Wait for wiring
    sleep(Duration::from_millis(50)).await;

    // Send 5 requests
    // Expectation: 
    // - Requests 1, 2, 3 -> Trigger Batch 1
    // - Requests 4, 5 -> Buffer (wait for 6th)
    
    for i in 1..=5 {
        println!("Sending Request {}...", i);
        batcher.send_packet(AppMsg::Request(format!("req-{}", i))).await.unwrap();
        sleep(Duration::from_millis(100)).await;
    }

    // Wait for processing to complete
    sleep(Duration::from_secs(2)).await;
    
    // Explicitly send the 6th to trigger second batch?
    println!("Sending Request 6...");
    batcher.send_packet(AppMsg::Request("req-6".to_string())).await.unwrap();
    
    sleep(Duration::from_secs(1)).await;
    println!("Exiting.");
}
