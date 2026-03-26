use isanagent::{ActorLogic, NodeHandle, ActorError};
use async_trait::async_trait;
use tokio::time::{sleep, Duration};

// --- Basic Message Type ---
#[derive(Debug, Clone)]
enum Message {
    Ping(usize),
    Pong(usize),
}

// --- Ping Actor ---
struct PingActor;

#[async_trait]
impl ActorLogic<Message> for PingActor {
    async fn process(&mut self, packet: Message) -> Result<Option<(String, Message)>, ActorError> {
        match packet {
            Message::Ping(n) => {
                println!("[PingActor] Received Ping({})", n);
                sleep(Duration::from_millis(100)).await;
                Ok(Some(("pong".to_string(), Message::Pong(n + 1))))
            }
            _ => Err(ActorError::from("Unexpected message")),
        }
    }
}

// --- Pong Actor ---
struct PongActor;

#[async_trait]
impl ActorLogic<Message> for PongActor {
    async fn process(&mut self, packet: Message) -> Result<Option<(String, Message)>, ActorError> {
        match packet {
            Message::Pong(n) => {
                println!("[PongActor] Received Pong({})", n);
                if n < 5 {
                    sleep(Duration::from_millis(100)).await;
                    Ok(Some(("ping".to_string(), Message::Ping(n + 1))))
                } else {
                    println!("[PongActor] Done.");
                    Ok(None)
                }
            }
            _ => Err(ActorError::from("Unexpected message")),
        }
    }
}

// --- Main ---

#[tokio::main]
async fn main() {
    println!("Starting isanagent Actor System...");

    // Create Nodes
    let pinger = NodeHandle::new(PingActor, 10, 3, Duration::from_millis(100));
    let ponger = NodeHandle::new(PongActor, 10, 3, Duration::from_millis(100));

    // Wire them: Ping - "pong" >> Pong
    let _ = &pinger - "pong" >> &ponger;
    // Wire: Pong - "ping" >> Ping
    let _ = &ponger - "ping" >> &pinger;

    // Wait for wiring
    sleep(Duration::from_millis(50)).await;

    // Start
    println!("Sending initial Ping(0)...");
    pinger.send_packet(Message::Ping(0)).await.unwrap();

    // Run for a bit
    sleep(Duration::from_secs(2)).await;
    println!("Exiting.");
}
