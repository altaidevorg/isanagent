use agent_rs::{ActorLogic, NodeHandle, ActorError, Supervisor, SupervisorPolicy};
use async_trait::async_trait;
use tokio::time::{sleep, Duration};
use std::sync::{Arc, Mutex};

// --- Flaky Actor ---
#[derive(Clone)]
struct FlakyLogic {
    fail_countdown: Arc<Mutex<i32>>,
}

impl FlakyLogic {
    fn new(fails_in: i32) -> Self {
        Self {
            fail_countdown: Arc::new(Mutex::new(fails_in)),
        }
    }
}

#[async_trait]
impl ActorLogic<String> for FlakyLogic {
    fn name(&self) -> String {
        "FlakyActor".to_string()
    }

    async fn process(&mut self, packet: String) -> Result<Option<(String, String)>, ActorError> {
        let mut count = self.fail_countdown.lock().unwrap();
        println!("[FlakyActor] Processing: '{}', countdown: {}", packet, *count);
        
        if *count <= 0 {
            println!("[FlakyActor] CRASHING!");
            return Err(ActorError::from("Simulated Crash"));
        }
        
        *count -= 1;
        Ok(Some(("next".to_string(), format!("Processed: {}", packet))))
    }
}

// --- Main ---

#[tokio::main]
async fn main() {
    println!("Starting Supervisor Demo...");

    // Factory: Creates a fresh FlakyActor that survives 2 calls then crashes
    // Note: To demonstrate restart, the factory should return a *fresh* state.
    // Ideally, we want the actor to be reset.
    let factory = || {
        // New instance each time
        Box::new(FlakyLogic::new(2)) as Box<dyn ActorLogic<String>>
    };

    // Create Supervisor with Restart Policy
    let supervisor_logic = Supervisor::new(SupervisorPolicy::Restart, factory);
    
    // NodeHandle
    let node = NodeHandle::new(supervisor_logic, 10, 1, Duration::from_millis(100));

    // Send messages
    // 1. Ok
    // 2. Ok
    // 3. Crash -> Restart (fresh state, countdown=2) -> Retry -> Ok
    
    for i in 1..=5 {
        println!("Sending Msg {}...", i);
        match node.send_packet(format!("Msg-{}", i)).await {
            Ok(_) => println!("Sent Msg {}", i),
            Err(e) => println!("Failed to send Msg {}: {}", i, e),
        }
        sleep(Duration::from_millis(200)).await;
    }

    sleep(Duration::from_secs(1)).await;
    println!("Exiting.");
}
