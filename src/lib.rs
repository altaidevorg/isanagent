use async_trait::async_trait;
use log::{error, info, warn};
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Shr, Sub};
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

pub mod agent;
pub mod bus;
pub mod channels;
pub mod checkpoint;
pub mod clarification;
pub mod config;
pub mod execution;
pub mod hooks;
pub mod host;
pub mod log_rotation;
pub mod logging;
pub mod memory;
pub mod ml_engineer;
pub mod multi_tenant_edge;
pub mod onboarding;
pub mod onboarding_interactive;
pub mod provider;
pub mod provider_registry;
pub mod redact;
pub mod reflection;
pub mod scheduler;
pub mod session;
pub mod skills;
pub mod tool_activity;
pub mod tool_runtime;
pub mod tools;
pub mod traits;
pub mod utils;
pub mod workspace;

// --- Message Protocol ---

// Define the message protocol and actor logic.

/// Message Protocol for inter-actor communication.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Message<T: Debug + Send + Sync + Clone> {
    /// Data payload passed between nodes.
    Packet(T),
    /// Signal to stop/terminate the actor.
    Terminate,
    /// Configuration message to dynamically add a successor.
    AddSuccessor {
        /// The action string that triggers this route.
        action: String,
        /// The channel sender for the destination actor.
        sender: mpsc::Sender<Message<T>>,
    },
}

// --- Errors ---

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum ActorError {
    #[error("Logic error in actor '{actor}': {source}")]
    LogicError {
        actor: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("Actor '{actor}' failed after {max_retries} attempts: {last_error}")]
    MaxRetriesReached {
        actor: String,
        max_retries: u32,
        last_error: String, // Simplified for now
    },
    #[error("Generic error: {0}")]
    Generic(String),
}

// Helper for user logic to return generic errors easily
impl From<String> for ActorError {
    fn from(s: String) -> Self {
        ActorError::Generic(s)
    }
}
impl From<&str> for ActorError {
    fn from(s: &str) -> Self {
        ActorError::Generic(s.to_string())
    }
}

// --- Supervisor ---

/// Policy for handling child actor failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupervisorPolicy {
    /// Stop the actor (propagate error).
    Stop,
    /// Restart the actor and retry the message.
    Restart,
}

/// A Supervisor actor that wraps a child logic and manages recovery.
pub struct Supervisor<T, F>
where
    T: Debug + Send + Sync + Clone + 'static,
    F: Fn() -> Box<dyn ActorLogic<T>> + Send + Sync + 'static,
{
    factory: F,
    child: Box<dyn ActorLogic<T>>,
    policy: SupervisorPolicy,
}

impl<T, F> Supervisor<T, F>
where
    T: Debug + Send + Sync + Clone + 'static,
    F: Fn() -> Box<dyn ActorLogic<T>> + Send + Sync + 'static,
{
    pub fn new(policy: SupervisorPolicy, factory: F) -> Self {
        let child = factory();
        Self {
            factory,
            child,
            policy,
        }
    }
}

#[async_trait]
impl<T, F> ActorLogic<T> for Supervisor<T, F>
where
    T: Debug + Send + Sync + Clone + 'static,
    F: Fn() -> Box<dyn ActorLogic<T>> + Send + Sync + 'static,
{
    fn name(&self) -> String {
        format!("Supervisor({})", self.child.name())
    }

    async fn process(&mut self, packet: T) -> Result<Option<(String, T)>, ActorError> {
        // Try processing with current child
        match self.child.process(packet.clone()).await {
            Ok(result) => Ok(result),
            Err(e) => {
                match self.policy {
                    SupervisorPolicy::Stop => Err(e),
                    SupervisorPolicy::Restart => {
                        error!(
                            "Supervisor caught error in '{}': {}. Restarting...",
                            self.child.name(),
                            e
                        );
                        // Restart
                        self.child = (self.factory)();
                        // Retry once
                        info!("Supervisor retrying message for '{}'...", self.child.name());
                        self.child.process(packet).await
                    }
                }
            }
        }
    }

    async fn on_tick(&mut self) -> Result<Option<(String, T)>, ActorError> {
        match self.child.on_tick().await {
            Ok(result) => Ok(result),
            Err(e) => match self.policy {
                SupervisorPolicy::Stop => Err(e),
                SupervisorPolicy::Restart => {
                    error!(
                        "Supervisor caught tick error in '{}': {}. Restarting...",
                        self.child.name(),
                        e
                    );
                    self.child = (self.factory)();
                    Ok(None)
                }
            },
        }
    }

    fn tick_interval(&self) -> Option<Duration> {
        self.child.tick_interval()
    }
}

// --- Logic Trait ---

/// Trait for Actor Logic.
///
/// Actors consume messages of type `T` and produce results which determine the next route.
/// This logic runs inside the actor's event loop.
#[async_trait]
pub trait ActorLogic<T>: Send + 'static
where
    T: Debug + Send + Sync + Clone + 'static,
{
    /// Returns the name of the actor (for logging/debugging).
    /// Default implementation uses the struct name via reflection.
    fn name(&self) -> String {
        let full_name = std::any::type_name::<Self>();
        full_name
            .split("::")
            .last()
            .unwrap_or(full_name)
            .to_string()
    }

    /// Optional: Prepare the data before processing.
    /// Default: returns data as-is.
    async fn prep(&mut self, packet: T) -> Result<T, ActorError> {
        Ok(packet)
    }

    /// Process a message packet and return an optional Action string and the (possibly modified) packet.
    /// This corresponds to `exec` in the old model.
    ///
    /// # Returns
    /// - `Ok(Some((action, packet)))`: The actor processed the input and chose to transition.
    ///   The system will route `packet` to the successor registered for `action`.
    /// - `Ok(None)`: The actor absorbed the input (e.g., buffering/batching) and no transition occurs yet.
    /// - `Err(ActorError)`: Logic error occurred.
    async fn process(&mut self, packet: T) -> Result<Option<(String, T)>, ActorError>;

    /// Optional: Post-process the result.
    /// Default: returns result as-is.
    async fn post(
        &mut self,
        result: Option<(String, T)>,
    ) -> Result<Option<(String, T)>, ActorError> {
        Ok(result)
    }

    /// Optional: Return a duration for periodic ticking.
    fn tick_interval(&self) -> Option<Duration> {
        None
    }

    /// Optional: Called periodically if `tick_interval` returns `Some`.
    async fn on_tick(&mut self) -> Result<Option<(String, T)>, ActorError> {
        Ok(None)
    }
}

// --- Actor Node ---

/// Internal Actor structure that runs the event loop.
pub struct ActorNode<T>
where
    T: Debug + Send + Sync + Clone + 'static,
{
    receiver: mpsc::Receiver<Message<T>>,
    successors: HashMap<String, mpsc::Sender<Message<T>>>,
    logic: Box<dyn ActorLogic<T>>,
    name: String,
    max_retries: u32,
    retry_wait: Duration,
}

impl<T> ActorNode<T>
where
    T: Debug + Send + Sync + Clone + 'static,
{
    /// Create a new ActorNode with the given logic and receiver channel.
    pub fn new(
        logic: Box<dyn ActorLogic<T>>,
        receiver: mpsc::Receiver<Message<T>>,
        max_retries: u32,
        retry_wait: Duration,
    ) -> Self {
        Self {
            receiver,
            successors: HashMap::new(),
            name: logic.name(),
            logic,
            max_retries: max_retries.max(1), // At least 1 attempt
            retry_wait,
        }
    }

    /// Run the actor's main event loop. This blocks until the channel is closed or Terminate is received.
    pub async fn run(mut self) {
        info!("Actor '{}' started.", self.name);

        let tick_dur = self.logic.tick_interval();
        let mut interval = tick_dur.map(tokio::time::interval);

        loop {
            // We use tokio::select! to listen to both channel and ticker (if present)
            let msg_opt = if let Some(ref mut ticker) = interval {
                tokio::select! {
                    msg = self.receiver.recv() => msg,
                    _ = ticker.tick() => {
                        // Handle Tick with Retry
                        let mut attempt = 0;
                        loop {
                            attempt += 1;
                            match self.logic.on_tick().await {
                                Ok(Some((action, new_data))) => {
                                    let successor = self.get_successor(&action);
                                    if let Some(sender) = successor {
                                        info!("Actor '{}' transitioning with action '{}'.", self.name, action);
                                        let _ = sender.send(Message::Packet(new_data)).await;
                                    } else if !self.successors.is_empty() {
                                        warn!("Actor '{}' has no successor for action '{}'. Dropping packet.", self.name, action);
                                    } else {
                                        info!("Actor '{}' finished chain (no successors).", self.name);
                                    }
                                    break;
                                }
                                Ok(None) => {
                                    break;
                                }
                                Err(e) => {
                                    if attempt >= self.max_retries {
                                        error!("Actor '{}' tick failed after {} attempts: {:?}", self.name, attempt, e);
                                        break;
                                    } else {
                                        warn!("Actor '{}' tick attempt {} failed: {:?}. Retrying...", self.name, attempt, e);
                                        if self.retry_wait > Duration::ZERO {
                                            sleep(self.retry_wait).await;
                                        }
                                    }
                                }
                            }
                        }
                        None // Loop back to select
                    }
                }
            } else {
                self.receiver.recv().await
            };

            if let Some(msg) = msg_opt {
                match msg {
                    Message::Packet(data) => {
                        info!(
                            "Actor '{}' received packet of type '{}'.",
                            self.name,
                            std::any::type_name::<T>()
                        );

                        // Retry loop
                        let mut attempt = 0;
                        loop {
                            attempt += 1;
                            // Clone data for processing if retries needed (T is Clone)
                            let data_clone = data.clone();

                            // Lifecycle: Prep -> Process -> Post
                            // We need to handle intermediate failures

                            let run_lifecycle = async {
                                let prepped_data = self.logic.prep(data_clone).await?;
                                let result = self.logic.process(prepped_data).await?;
                                let posted_result = self.logic.post(result).await?;
                                Ok::<Option<(String, T)>, ActorError>(posted_result)
                            };

                            match run_lifecycle.await {
                                Ok(Some((action, new_data))) => {
                                    let successor = self.get_successor(&action);
                                    if let Some(sender) = successor {
                                        info!(
                                            "Actor '{}' transitioning with action '{}'.",
                                            self.name, action
                                        );
                                        let _ = sender.send(Message::Packet(new_data)).await;
                                    } else if !self.successors.is_empty() {
                                        warn!("Actor '{}' has no successor for action '{}'. Dropping packet.", self.name, action);
                                    } else {
                                        info!(
                                            "Actor '{}' finished chain (no successors).",
                                            self.name
                                        );
                                    }
                                    break; // Success
                                }
                                Ok(None) => {
                                    info!(
                                        "Actor '{}' consumed packet (no further action needed).",
                                        self.name
                                    );
                                    break; // Success (absorbed)
                                }
                                Err(e) => {
                                    if attempt >= self.max_retries {
                                        error!(
                                            "Actor '{}' failed final attempt {}: {:?}",
                                            self.name, attempt, e
                                        );
                                        break; // Give up
                                    } else {
                                        warn!(
                                            "Actor '{}' attempt {} failed: {:?}. Retrying...",
                                            self.name, attempt, e
                                        );
                                        if self.retry_wait > Duration::ZERO {
                                            sleep(self.retry_wait).await;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Message::Terminate => {
                        info!("Actor '{}' received terminate signal.", self.name);
                        break;
                    }
                    Message::AddSuccessor { action, sender } => {
                        info!(
                            "Actor '{}' wiring successor for action '{}'.",
                            self.name, action
                        );
                        self.successors.insert(action, sender);
                    }
                }
            } else {
                // Channel closed (and no tick or tick loop hit None check)
                // If interval is None, msg_opt is None means channel closed.
                // If interval is Some, we only reach here if recv() returned None inside select.
                // In select!, if recv returns None, we get None.
                if interval.is_none() || self.receiver.is_closed() {
                    break;
                }
            }
        }
        info!("Actor '{}' run loop finished.", self.name);
    }

    fn get_successor(&self, action: &str) -> Option<mpsc::Sender<Message<T>>> {
        self.successors
            .get(action)
            .or_else(|| self.successors.get("default"))
            .cloned()
    }
}

// --- Generic Actors ---

/// A generic batching actor.
///
/// Accumulates items of type `T` into a buffer.
/// Emits a batch when:
/// 1. Buffer size reaches `batch_size`.
/// 2. `timeout` elapses since the last flush (if buffer is not empty).
///
/// Requires a `wrapper` function to convert `Vec<T>` back into `T` (e.g. wrapping in an Enum variant).
pub struct Batcher<T, F>
where
    T: Debug + Send + Sync + Clone + 'static,
    F: Fn(Vec<T>) -> T + Send + Sync + 'static,
{
    buffer: Vec<T>,
    batch_size: usize,
    timeout: Duration,
    wrapper: F,
    action: String,
}

impl<T, F> Batcher<T, F>
where
    T: Debug + Send + Sync + Clone + 'static,
    F: Fn(Vec<T>) -> T + Send + Sync + 'static,
{
    pub fn new(batch_size: usize, timeout: Duration, action: String, wrapper: F) -> Self {
        Self {
            buffer: Vec::new(),
            batch_size,
            timeout,
            wrapper,
            action,
        }
    }

    fn flush(&mut self) -> (String, T) {
        let items = std::mem::take(&mut self.buffer);
        let payload = (self.wrapper)(items);
        (self.action.clone(), payload)
    }
}

#[async_trait]
impl<T, F> ActorLogic<T> for Batcher<T, F>
where
    T: Debug + Send + Sync + Clone + 'static,
    F: Fn(Vec<T>) -> T + Send + Sync + 'static,
{
    fn name(&self) -> String {
        "GenericBatcher".to_string()
    }

    fn tick_interval(&self) -> Option<Duration> {
        Some(self.timeout)
    }

    async fn on_tick(&mut self) -> Result<Option<(String, T)>, ActorError> {
        if !self.buffer.is_empty() {
            info!("Batcher: Timeout flush ({} items)", self.buffer.len());
            Ok(Some(self.flush()))
        } else {
            Ok(None)
        }
    }

    async fn process(&mut self, packet: T) -> Result<Option<(String, T)>, ActorError> {
        self.buffer.push(packet);
        if self.buffer.len() >= self.batch_size {
            info!("Batcher: Size flush ({} items)", self.buffer.len());
            Ok(Some(self.flush()))
        } else {
            Ok(None)
        }
    }
}

// --- Node Handle ---

/// External handle to an Actor. Check `src/bin/actor_example.rs` for usage.
///
/// This handle is cheap to clone (it wraps a channel sender) and is used to wire up the graph
/// and send initial messages.
#[derive(Clone, Debug)]
pub struct NodeHandle<T>
where
    T: Debug + Send + Sync + Clone + 'static,
{
    pub sender: mpsc::Sender<Message<T>>,
    pub name: String,
}

impl<T> NodeHandle<T>
where
    T: Debug + Send + Sync + Clone + 'static,
{
    /// Create a new Actor and return its handle.
    ///
    /// The actor is immediately spawned onto the Tokio runtime.
    /// `buffer` specifies the channel capacity.
    /// `max_retries` and `retry_wait` configure basic error resilience.
    pub fn new<L>(logic: L, buffer: usize, max_retries: u32, retry_wait: Duration) -> Self
    where
        L: ActorLogic<T> + 'static,
    {
        let (tx, rx) = mpsc::channel(buffer);
        let name = logic.name();
        let actor = ActorNode::new(Box::new(logic), rx, max_retries, retry_wait);

        // Spawn the actor immediately
        tokio::spawn(async move {
            actor.run().await;
        });

        Self { sender: tx, name }
    }

    /// Send a data packet to this actor.
    pub async fn send_packet(&self, packet: T) -> Result<(), String> {
        self.sender
            .send(Message::Packet(packet))
            .await
            .map_err(|e| e.to_string())
    }

    /// Asynchronously wire a successor to this node for a given action.
    /// This ensures the wiring message is sent before returning.
    pub async fn wire(&self, action: &str, target: &NodeHandle<T>) {
        let msg = Message::AddSuccessor {
            action: action.to_string(),
            sender: target.sender.clone(),
        };
        if let Err(e) = self.sender.send(msg).await {
            log::error!("Failed to wire successor: {}", e);
        }
    }

    /// Create a "Listener" handle and a receiver.
    /// This handle acts as a valid destination for actors, but sends messages directly to the returned receiver
    /// instead of another actor. Use this to consume outputs in the main thread.
    pub fn create_listener(name: &str, buffer: usize) -> (Self, mpsc::Receiver<Message<T>>) {
        let (tx, rx) = mpsc::channel(buffer);
        let handle = Self {
            sender: tx,
            name: name.to_string(),
        };
        (handle, rx)
    }
}

// --- Syntactic Sugar (Operator Overloading) ---

// We want: &node1 - "action" >> &node2
// Step 1: &node1 - "action" -> Returns a temporary "Connector"
// Step 2: Connector >> &node2 -> Sends Wire message to node1, returns node2 (cloned)

/// Temporary structure created by the `-` operator.
pub struct Connector<T>
where
    T: Debug + Send + Sync + Clone + 'static,
{
    source: NodeHandle<T>,
    action: String,
}

// Implement Sub for &NodeHandle ( &node - "action" )
// The output Connector owns a CLONE of the handle, so we don't need lifetimes in Connector itself.
impl<T, S> Sub<S> for &NodeHandle<T>
where
    T: Debug + Send + Sync + Clone + 'static,
    S: Into<String>,
{
    type Output = Connector<T>;

    fn sub(self, action: S) -> Self::Output {
        Connector {
            source: self.clone(),
            action: action.into(),
        }
    }
}

// Implement Shr for Connector ( connector >> &node )
// Accepts a reference to the target node, removing the need for explicit clones by the user.
impl<T> Shr<&NodeHandle<T>> for Connector<T>
where
    T: Debug + Send + Sync + Clone + 'static,
{
    type Output = NodeHandle<T>;

    fn shr(self, rhs: &NodeHandle<T>) -> Self::Output {
        // We need to send a message to self.source to add rhs as successor
        let action = self.action;
        let source_sender = self.source.sender.clone(); // Clone sender for the task
        let target_sender = rhs.sender.clone(); // Clone target sender for the message

        // Wiring happens asynchronously.
        tokio::spawn(async move {
            let msg = Message::AddSuccessor {
                action,
                sender: target_sender,
            };
            if let Err(e) = source_sender.send(msg).await {
                error!("Failed to wire successor: {}", e);
            }
        });

        // Return a clone of the RHS handle to allow chaining (A >> B >> C) if desired
        rhs.clone()
    }
}
