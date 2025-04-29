use async_trait::async_trait;
use downcast_rs::{impl_downcast, DowncastSync};
use dyn_clone::DynClone;
use futures::executor::block_on;
use log::{info, warn};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ops::{Shr, Sub};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::Mutex as TokioMutex;

// --- Basic Types ---

// Use Box<dyn Any + Send + Sync> for flexibility similar to Python's dynamic types
// Requires downcasting in user code.
// pub type AnySendSync = Box<dyn Any + Send + Sync>; // Remove this original definition

// Define a trait that includes Any, Send, Sync, DowncastSync, and DynClone
pub trait AnySendSyncExt: Any + DowncastSync + DynClone + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
impl_downcast!(AnySendSyncExt);

// Implement the trait for any type T that meets the bounds
impl<T: Any + Clone + Send + Sync + 'static> AnySendSyncExt for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
dyn_clone::clone_trait_object!(AnySendSyncExt);

// Define the type alias using the trait object
pub type AnySendSync = Box<dyn AnySendSyncExt>;

// Define a common Error type
#[derive(Error, Debug)]
pub enum FlowError {
    #[error("Execution error in node '{node_name}': {source}")]
    ExecutionError {
        node_name: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("Fallback error in node '{node_name}': {source}")]
    FallbackError {
        node_name: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("Prep error in node '{node_name}': {source}")]
    PrepError {
        node_name: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("Post error in node '{node_name}': {source}")]
    PostError {
        node_name: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("Orchestration error: {0}")]
    OrchestrationError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Type downcast error: expected {expected}, got value")]
    DowncastError { expected: String },
    #[error("Action '{action}' not found in successors for node '{node_name}'. Available: {available:?}")]
    ActionNotFound {
        node_name: String,
        action: String,
        available: Vec<String>,
    },
    #[error("Error during batch processing: {0}")]
    BatchError(String),
    #[error("Cannot run async node synchronously")]
    CannotRunAsyncSynchronously,
    #[error("Generic error: {0}")]
    Generic(String), // For simpler errors
}

// Result types using AnySendSync for Prep/Exec results
pub type PrepResult = Result<AnySendSync, FlowError>;
pub type ExecResult = Result<AnySendSync, FlowError>;
// Post result is the action string or an error
pub type PostResult = Result<String, FlowError>;

// Marker trait for user-defined Shared State and Params (must be Send + Sync + Clone + 'static)
pub trait SharedState: Send + Sync + Clone + Debug + 'static {}
impl<T: Send + Sync + Clone + Debug + 'static> SharedState for T {}

pub trait Params: Send + Sync + Clone + Debug + 'static {}
impl<T: Send + Sync + Clone + Debug + 'static> Params for T {}

// Default empty state/params if needed
#[derive(Clone, Debug, Default)]
pub struct EmptyState;

#[derive(Clone, Debug, Default)]
pub struct EmptyParams;

// --- Forward Declarations ---
#[derive(Clone)]
pub struct Flow<S: SharedState, P: Params + Default> {
    node_impl: Arc<Mutex<FlowImpl<S, P>>>,
}
#[derive(Clone)]
pub struct AsyncFlow<S: SharedState, P: Params + Default> {
    node_impl: Arc<TokioMutex<AsyncFlowImpl<S, P>>>,
}

// --- Node Type Enum ---
// Enum to hold either sync or async node references within a potentially mixed graph
#[derive(Clone)]
pub enum NodeType<S: SharedState, P: Params + Default> {
    Sync(SyncNodeHandle<S, P>),
    Async(AsyncNodeHandle<S, P>),
    SyncFlow(Flow<S, P>),       // Embed Flows as Nodes
    AsyncFlow(AsyncFlow<S, P>), // Embed AsyncFlows as Nodes
}

// Manual Debug impl for NodeType to avoid issues with dyn NodeLike and cycles
impl<S: SharedState, P: Params + Default> Debug for NodeType<S, P>
where
    P: Default,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeType::Sync(h) => write!(f, "NodeType::Sync({})", h.name()),
            NodeType::Async(h) => write!(f, "NodeType::Async({})", h.name()),
            NodeType::SyncFlow(h) => write!(f, "NodeType::SyncFlow({})", h.name()),
            NodeType::AsyncFlow(h) => write!(f, "NodeType::AsyncFlow({})", h.name()),
        }
    }
}

impl<S: SharedState, P: Params + Default> NodeType<S, P> {
    // Helper to get successors mutably (requires locking)
    // Be careful with locking order if modifying multiple nodes.
    fn with_successors_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut HashMap<String, NodeType<S, P>>) -> R,
        P: Params + Default,
    {
        match self {
            NodeType::Sync(h) => f(h.node_impl.lock().unwrap().successors_mut()),
            NodeType::Async(h) => {
                // Use a simple executor that works in any context
                futures::executor::block_on(async {
                    let mut guard = h.node_impl.lock().await;
                    f(guard.successors_mut())
                })
            }
            NodeType::SyncFlow(h) => f(h.node_impl.lock().unwrap().successors_mut()),
            NodeType::AsyncFlow(h) => {
                // Use a simple executor that works in any context
                futures::executor::block_on(async {
                    let mut guard = h.node_impl.lock().await;
                    f(guard.successors_mut())
                })
            }
        }
    }

    // Helper to set params (requires locking)
    fn set_params(&self, params: P)
    where
        P: Params + Default + Clone,
    {
        match self {
            NodeType::Sync(h) => h.node_impl.lock().unwrap().set_params(params.clone()),
            NodeType::Async(h) => {
                // Use a simple executor that works in any context
                futures::executor::block_on(async {
                    h.node_impl.lock().await.set_params(params.clone());
                });
            }
            NodeType::SyncFlow(h) => h.node_impl.lock().unwrap().set_params(params.clone()),
            NodeType::AsyncFlow(h) => {
                // Use a simple executor that works in any context
                futures::executor::block_on(async {
                    h.node_impl.lock().await.set_params(params);
                });
            }
        }
    }

    // Helper to get name
    fn name(&self) -> String
    where
        P: Params + Default,
    {
        match self {
            NodeType::Sync(h) => h.node_impl.lock().unwrap().name(),
            NodeType::Async(h) => {
                // Use a simple executor that works in any context
                futures::executor::block_on(async { h.node_impl.lock().await.name() })
            }
            NodeType::SyncFlow(h) => h.node_impl.lock().unwrap().name(),
            NodeType::AsyncFlow(h) => {
                // Use a simple executor that works in any context
                futures::executor::block_on(async { h.node_impl.lock().await.name() })
            }
        }
    }
}

// --- Node Traits ---

/// Common methods for all node-like structures (Nodes and Flows)
#[async_trait]
pub trait NodeLike<S: SharedState, P: Params + Default>: Send + Sync {
    fn name(&self) -> String;
    fn set_params(&mut self, params: P);
    fn successors(&self) -> &HashMap<String, NodeType<S, P>>;
    fn successors_mut(&mut self) -> &mut HashMap<String, NodeType<S, P>>;

    fn add_successor(&mut self, action: String, successor: NodeType<S, P>) -> NodeType<S, P> {
        if self.successors().contains_key(&action) {
            warn!(
                "Node '{}': Overwriting successor for action '{}'",
                self.name(),
                action
            );
        }
        self.successors_mut().insert(action, successor.clone());
        successor
    }

    // Internal prep/exec/post methods specific to node/flow types
    fn _run(&mut self, shared: &mut S) -> PostResult;
    async fn _run_async(&mut self, shared: &mut S) -> PostResult;
}

/// Trait for synchronous node logic
pub trait SyncLogic<S: SharedState, P: Params + Default>: Send + Sync {
    fn name(&self) -> String;
    fn prep(&mut self, shared: &mut S, params: &P) -> PrepResult;
    fn exec(&mut self, prep_res: AnySendSync, params: &P) -> ExecResult;
    fn post(
        &mut self,
        shared: &mut S,
        prep_res: AnySendSync,
        exec_res: ExecResult,
        params: &P,
    ) -> PostResult;
    fn exec_fallback(&mut self, prep_res: AnySendSync, error: FlowError, params: &P) -> ExecResult;
}

/// Trait for asynchronous node logic
#[async_trait]
pub trait AsyncLogic<S: SharedState, P: Params + Default>: Send + Sync {
    fn name(&self) -> String;
    async fn prep_async(&mut self, shared: &mut S, params: &P) -> PrepResult;
    async fn exec_async(&mut self, prep_res: AnySendSync, params: &P) -> ExecResult;
    async fn post_async(
        &mut self,
        shared: &mut S,
        prep_res: AnySendSync,
        exec_res: ExecResult,
        params: &P,
    ) -> PostResult;
    async fn exec_fallback_async(
        &mut self,
        prep_res: AnySendSync,
        error: FlowError,
        params: &P,
    ) -> ExecResult;
}

// --- Node Implementation Structs ---

/// Internal implementation details for a synchronous node
#[derive(Debug)]
struct SyncNodeImpl<L, S, P>
where
    L: SyncLogic<S, P>,
    S: SharedState,
    P: Params + Default,
{
    name: String,
    logic: L,
    params: P,
    successors: HashMap<String, NodeType<S, P>>,
    max_retries: u32,
    wait: Duration,
    // Internal state like current retry count could go here if needed across _run calls
    // current_retry: u32 // But Python resets this per call
    _phantom_s: PhantomData<S>,
}

impl<L, S, P> SyncNodeImpl<L, S, P>
where
    L: SyncLogic<S, P>,
    S: SharedState,
    P: Params + Default,
{
    fn new(logic: L, max_retries: u32, wait_secs: u64) -> Self {
        Self {
            name: logic.name(),
            logic,
            params: P::default(),
            successors: HashMap::new(),
            max_retries: max_retries.max(1), // Ensure at least 1 try
            wait: Duration::from_secs(wait_secs),
            _phantom_s: PhantomData,
        }
    }
}

// Implement NodeLike for the internal implementation
#[async_trait]
impl<L, S, P> NodeLike<S, P> for SyncNodeImpl<L, S, P>
where
    L: SyncLogic<S, P>,
    S: SharedState,
    P: Params + Default,
{
    fn name(&self) -> String {
        self.name.clone()
    }
    fn set_params(&mut self, params: P) {
        self.params = params;
    }
    fn successors(&self) -> &HashMap<String, NodeType<S, P>> {
        &self.successors
    }
    fn successors_mut(&mut self) -> &mut HashMap<String, NodeType<S, P>> {
        &mut self.successors
    }

    fn _run(&mut self, shared: &mut S) -> PostResult {
        let prep_res = self
            .logic
            .prep(shared, &self.params)
            .map_err(|e| FlowError::PrepError {
                node_name: self.name(),
                source: Box::new(e),
            })?;

        let exec_res_final = 'retry: loop {
            let mut final_err = None;
            for attempt in 0..self.max_retries {
                // Clone prep_res if exec consumes it, otherwise pass reference
                // Assuming exec doesn't consume prep_res here. If it does, cloning is needed.
                // For Box<dyn Any> cloning is tricky. Let's assume exec takes & for now
                // If exec MUST take ownership, PrepResult needs to be clonable or regenerated.
                // Sticking to Python's apparent behavior: pass the same result object.
                let prep_res_exec = dyn_clone::clone_box(&*prep_res); // Clone the boxed value

                match self.logic.exec(prep_res_exec, &self.params) {
                    Ok(res) => break 'retry Ok(res), // Success, exit retry loop
                    Err(e) => {
                        final_err = Some(e); // Store last error
                        if attempt == self.max_retries - 1 {
                            // Max retries reached, break loop to use fallback
                            break;
                        }
                        if self.wait > Duration::ZERO {
                            info!(
                                "Node '{}': Attempt {} failed. Retrying after {:?}...",
                                self.name(),
                                attempt + 1,
                                self.wait
                            );
                            std::thread::sleep(self.wait);
                        } else {
                            info!(
                                "Node '{}': Attempt {} failed. Retrying immediately...",
                                self.name(),
                                attempt + 1
                            );
                        }
                    }
                }
            }
            // If loop finished due to retries, use fallback
            let prep_res_fallback = dyn_clone::clone_box(&*prep_res);
            let fallback_res = self.logic.exec_fallback(
                prep_res_fallback,
                final_err.unwrap_or_else(|| {
                    FlowError::Generic("Fallback triggered without error".into())
                }), // Should always have error
                &self.params,
            );
            break 'retry fallback_res; // Exit retry loop with fallback result
        };

        let post_res = self
            .logic
            .post(shared, prep_res, exec_res_final, &self.params)
            .map_err(|e| FlowError::PostError {
                node_name: self.name(),
                source: Box::new(e),
            })?;

        Ok(post_res)
    }

    // Async run is not directly supported for SyncNode, trigger error or panic
    async fn _run_async(&mut self, _shared: &mut S) -> PostResult {
        Err(FlowError::CannotRunAsyncSynchronously)
    }
}

/// Internal implementation details for an asynchronous node
#[derive(Debug)]
struct AsyncNodeImpl<L, S, P>
where
    L: AsyncLogic<S, P>,
    S: SharedState,
    P: Params + Default,
{
    name: String,
    logic: L,
    params: P,
    successors: HashMap<String, NodeType<S, P>>,
    max_retries: u32,
    wait: Duration,
    _phantom_s: PhantomData<S>,
}

impl<L, S, P> AsyncNodeImpl<L, S, P>
where
    L: AsyncLogic<S, P>,
    S: SharedState,
    P: Params + Default,
{
    fn new(logic: L, max_retries: u32, wait_secs: u64) -> Self {
        Self {
            name: logic.name(),
            logic,
            params: P::default(),
            successors: HashMap::new(),
            max_retries: max_retries.max(1),
            wait: Duration::from_secs(wait_secs),
            _phantom_s: PhantomData,
        }
    }
}

#[async_trait]
impl<L, S, P> NodeLike<S, P> for AsyncNodeImpl<L, S, P>
where
    L: AsyncLogic<S, P>,
    S: SharedState,
    P: Params + Default,
{
    fn name(&self) -> String {
        self.name.clone()
    }
    fn set_params(&mut self, params: P) {
        self.params = params;
    }
    fn successors(&self) -> &HashMap<String, NodeType<S, P>> {
        &self.successors
    }
    fn successors_mut(&mut self) -> &mut HashMap<String, NodeType<S, P>> {
        &mut self.successors
    }

    fn _run(&mut self, _shared: &mut S) -> PostResult {
        // We could potentially block_on here, but it's usually bad practice.
        // The design intends for AsyncNodes to be run by AsyncFlows.
        Err(FlowError::ConfigurationError(
            "Attempted to run an AsyncNode synchronously. Use run_async or an AsyncFlow.".into(),
        ))
    }

    async fn _run_async(&mut self, shared: &mut S) -> PostResult {
        let prep_res = self
            .logic
            .prep_async(shared, &self.params)
            .await
            .map_err(|e| FlowError::PrepError {
                node_name: self.name(),
                source: Box::new(e),
            })?;

        let exec_res_final = 'retry: loop {
            let mut final_err = None;
            for attempt in 0..self.max_retries {
                // Clone prep_res if exec consumes it. Assuming it needs cloning for async.
                let prep_res_exec = dyn_clone::clone_box(&*prep_res);

                match self.logic.exec_async(prep_res_exec, &self.params).await {
                    Ok(res) => break 'retry Ok(res), // Success
                    Err(e) => {
                        final_err = Some(e);
                        if attempt == self.max_retries - 1 {
                            break; // Max retries reached
                        }
                        if self.wait > Duration::ZERO {
                            info!(
                                "Node '{}': Attempt {} failed. Retrying after {:?}...",
                                self.name(),
                                attempt + 1,
                                self.wait
                            );
                            tokio::time::sleep(self.wait).await;
                        } else {
                            info!(
                                "Node '{}': Attempt {} failed. Retrying immediately...",
                                self.name(),
                                attempt + 1
                            );
                        }
                    }
                }
            }
            // Fallback
            let prep_res_fallback = dyn_clone::clone_box(&*prep_res);
            let fallback_res = self
                .logic
                .exec_fallback_async(
                    prep_res_fallback,
                    final_err.unwrap_or_else(|| {
                        FlowError::Generic("Fallback triggered without error".into())
                    }),
                    &self.params,
                )
                .await;
            break 'retry fallback_res;
        };

        let post_res = self
            .logic
            .post_async(shared, prep_res, exec_res_final, &self.params)
            .await
            .map_err(|e| FlowError::PostError {
                node_name: self.name(),
                source: Box::new(e),
            })?;

        Ok(post_res)
    }
}

// --- User-Facing Handles ---
// Provide Arc<Mutex<>> handles to the user for graph building

#[derive(Clone)]
pub struct SyncNodeHandle<S: SharedState, P: Params + Default> {
    // Use Arc<Mutex<>> for shared mutable access
    node_impl: Arc<Mutex<dyn NodeLike<S, P>>>,
}

impl<S: SharedState, P: Params + Default> SyncNodeHandle<S, P> {
    pub fn new<L>(logic: L, max_retries: u32, wait_secs: u64) -> Self
    where
        L: SyncLogic<S, P> + 'static, // Logic must be 'static
    {
        let node_impl = SyncNodeImpl::new(logic, max_retries, wait_secs);
        Self {
            node_impl: Arc::new(Mutex::new(node_impl)),
        }
    }

    /// Run the node directly (outside a flow). Successors are ignored.
    pub fn run(&self, shared: &mut S) -> PostResult {
        let mut node = self.node_impl.lock().unwrap();
        if !node.successors().is_empty() {
            warn!(
                "Node '{}': Running node directly, successors will be ignored. Use a Flow.",
                node.name()
            );
        }
        node._run(shared)
    }

    // Helper to convert handle to NodeType for graph building
    pub fn into_nodetype(self) -> NodeType<S, P> {
        NodeType::Sync(self)
    }

    // Add a name method to the handle for convenience (used in manual Debug)
    pub fn name(&self) -> String {
        self.node_impl.lock().unwrap().name()
    }
}

#[derive(Clone)]
pub struct AsyncNodeHandle<S: SharedState, P: Params + Default> {
    node_impl: Arc<TokioMutex<dyn NodeLike<S, P> + Send + Sync + 'static>>,
}

impl<S: SharedState, P: Params + Default> AsyncNodeHandle<S, P> {
    pub fn new<L>(logic: L, max_retries: u32, wait_secs: u64) -> Self
    where
        L: AsyncLogic<S, P> + Send + Sync + 'static,
        P: Params + Default,
    {
        let node_impl = AsyncNodeImpl::new(logic, max_retries, wait_secs);
        Self {
            node_impl: Arc::new(TokioMutex::new(node_impl)),
        }
    }

    /// Run the node directly (outside a flow). Successors are ignored.
    pub async fn run_async(&self, shared: &mut S) -> PostResult
    where
        P: Params + Default,
    {
        let mut node_guard = self.node_impl.lock().await;
        if !node_guard.successors().is_empty() {
            warn!(
                "Node '{}': Running node directly, successors will be ignored. Use an AsyncFlow.",
                node_guard.name()
            );
        }
        node_guard._run_async(shared).await
    }

    // Helper to convert handle to NodeType for graph building
    pub fn into_nodetype(self) -> NodeType<S, P> {
        NodeType::Async(self)
    }

    // Add a name method to the handle for convenience (used in manual Debug)
    pub fn name(&self) -> String
    where
        P: Default,
    {
        // Use a simple executor that works in any context
        futures::executor::block_on(async { self.node_impl.lock().await.name() })
    }
}

// --- Flow Implementation Structs ---

#[derive(Debug)]
struct BaseFlowData<S: SharedState, P: Params + Default> {
    name: String,
    params: P,
    successors: HashMap<String, NodeType<S, P>>,
    start_node: Option<NodeType<S, P>>,
    _phantom_s: PhantomData<S>,
}

impl<S: SharedState, P: Params + Default> BaseFlowData<S, P> {
    fn new(name: String) -> Self {
        Self {
            name,
            params: P::default(),
            successors: HashMap::new(),
            start_node: None,
            _phantom_s: PhantomData,
        }
    }

    fn set_start_node(&mut self, node: NodeType<S, P>) {
        self.start_node = Some(node);
    }

    fn get_next_node(
        &self,
        current_node_name: &str,
        successors: &HashMap<String, NodeType<S, P>>,
        action: &str,
    ) -> Option<NodeType<S, P>> {
        match successors.get(action).or_else(|| successors.get("default")) {
            Some(node) => Some(node.clone()), // Clone the Arc handle
            None => {
                if !successors.is_empty()
                    && action != "default"
                    && !successors.contains_key("default")
                {
                    warn!(
                         "Flow '{}': Action '{}' not found in successors for node '{}' and no 'default' exists. Flow ends. Available: {:?}",
                         self.name, action, current_node_name, successors.keys().collect::<Vec<_>>()
                     );
                } else if successors.is_empty() {
                    info!(
                        "Flow '{}': Node '{}' has no successors. Flow ends.",
                        self.name, current_node_name
                    );
                } else {
                    // Default action was implicitly tried or action was default and not found
                    info!("Flow '{}': No successor found for action '{}' or default from node '{}'. Flow ends.", self.name, action, current_node_name);
                }
                None
            }
        }
    }
}

// --- Sync Flow ---
#[derive(Debug)]
struct FlowImpl<S: SharedState, P: Params + Default> {
    base: BaseFlowData<S, P>,
    // Flow-specific logic can implement SyncLogic/AsyncLogic if flows can be nested
    // For now, Flow just orchestrates
}

impl<S: SharedState, P: Params + Default> Flow<S, P> {
    pub fn new(name: &str) -> Self {
        Self {
            node_impl: Arc::new(Mutex::new(FlowImpl {
                base: BaseFlowData::new(name.to_string()),
            })),
        }
    }

    /// Set the starting node for the flow.
    pub fn start(&self, node: NodeType<S, P>) -> NodeType<S, P>
    where
        P: Params + Default,
    {
        // Direct lock for std::sync::Mutex
        self.node_impl
            .lock()
            .unwrap()
            .base
            .set_start_node(node.clone());
        node
    }

    /// Run the synchronous flow.
    pub fn run(&self, shared: &mut S) -> PostResult
    where
        P: Params + Default,
    {
        let mut flow_impl = self.node_impl.lock().unwrap();
        flow_impl._run(shared)
    }

    // Helper to convert handle to NodeType for graph building
    pub fn into_nodetype(self) -> NodeType<S, P> {
        NodeType::SyncFlow(self)
    }

    // Expose base methods via the handle
    pub fn name(&self) -> String {
        self.node_impl.lock().unwrap().base.name.clone()
    }

    pub fn set_params(&self, params: P) {
        self.node_impl.lock().unwrap().base.params = params;
    }
}

#[async_trait]
impl<S: SharedState, P: Params + Default> NodeLike<S, P> for FlowImpl<S, P>
where
    S: SharedState,
    P: Params + Default,
{
    fn name(&self) -> String {
        self.base.name.clone()
    }
    fn set_params(&mut self, params: P) {
        self.base.params = params;
    }
    fn successors(&self) -> &HashMap<String, NodeType<S, P>> {
        &self.base.successors
    }
    fn successors_mut(&mut self) -> &mut HashMap<String, NodeType<S, P>> {
        &mut self.base.successors
    }

    /// Orchestrates the synchronous flow execution.
    fn _run(&mut self, shared: &mut S) -> PostResult {
        // Clone the start node Arc, not the node itself
        let mut current_node_opt = self.base.start_node.clone();
        let mut last_action = Ok("".to_string()); // Initialize last action

        while let Some(current_node) = current_node_opt {
            let (current_node_name, successors_clone, run_result) = match &current_node {
                NodeType::Sync(h) => {
                    let mut node_locked = h.node_impl.lock().unwrap();
                    node_locked.set_params(self.base.params.clone());
                    let name = node_locked.name();
                    let successors = node_locked.successors().clone();
                    let result = node_locked._run(shared);
                    drop(node_locked);
                    (name, successors, result)
                }
                NodeType::Async(_) => {
                    return Err(FlowError::ConfigurationError(format!(
                        "SyncFlow '{}' encountered an AsyncNode '{}'. Use AsyncFlow.",
                        self.name(),
                        current_node.name()
                    )));
                }
                NodeType::SyncFlow(f) => {
                    let mut node_locked = f.node_impl.lock().unwrap();
                    node_locked.set_params(self.base.params.clone());
                    let name = node_locked.name();
                    let successors = node_locked.successors().clone();
                    let result = node_locked._run(shared); // Run nested sync flow
                    drop(node_locked);
                    (name, successors, result)
                }
                NodeType::AsyncFlow(_) => {
                    return Err(FlowError::ConfigurationError(format!(
                        "SyncFlow '{}' encountered an AsyncFlow '{}'. Use AsyncFlow.",
                        self.name(),
                        current_node.name()
                    )));
                }
            };

            match run_result {
                Ok(action) => {
                    last_action = Ok(action.clone());
                    current_node_opt =
                        self.base
                            .get_next_node(&current_node_name, &successors_clone, &action);
                }
                Err(e) => {
                    // Propagate the error immediately
                    return Err(FlowError::OrchestrationError(format!(
                        "Flow '{}' failed at node '{}': {}",
                        self.name(),
                        current_node_name,
                        e
                    )));
                }
            }
        }
        // Return the action string from the last successfully executed node
        last_action
    }

    // Async run is not directly supported for SyncFlow
    async fn _run_async(&mut self, _shared: &mut S) -> PostResult {
        Err(FlowError::ConfigurationError(
            "Attempted to run a SyncFlow asynchronously. Use AsyncFlow.".into(),
        ))
    }
}

// --- Async Flow ---
#[derive(Debug)]
struct AsyncFlowImpl<S: SharedState, P: Params + Default> {
    base: BaseFlowData<S, P>,
    // Can potentially hold AsyncLogic if flows implement prep/exec/post
}

impl<S: SharedState, P: Params + Default> AsyncFlow<S, P> {
    pub fn new(name: &str) -> Self {
        Self {
            node_impl: Arc::new(TokioMutex::new(AsyncFlowImpl {
                base: BaseFlowData::new(name.to_string()),
            })),
        }
    }

    /// Set the starting node for the flow.
    pub fn start(&self, node: NodeType<S, P>) -> NodeType<S, P>
    where
        P: Params + Default,
    {
        // Use a simple executor that works in any context
        futures::executor::block_on(async {
            self.node_impl
                .lock()
                .await
                .base
                .set_start_node(node.clone());
        });
        node
    }

    /// Run the asynchronous flow.
    pub async fn run_async(&self, shared: &mut S) -> PostResult
    where
        P: Clone,
    {
        // Using an inner function to break the recursion
        async fn run_inner_impl<S: SharedState, P: Params + Default + Clone>(
            flow: &AsyncFlow<S, P>, // Changed type from &mut AsyncFlowImpl to &AsyncFlow
            shared: &mut S,
        ) -> PostResult {
            // Lock only to get initial state, then release
            let (start_node, initial_params, flow_name) = {
                let flow_impl_guard = flow.node_impl.lock().await;
                (
                    flow_impl_guard.base.start_node.clone(),
                    flow_impl_guard.base.params.clone(),
                    flow_impl_guard.base.name.clone(),
                )
            };

            // Use a local mutable variable for the current node
            let mut current_node_opt = start_node;
            let mut last_action = Ok("".to_string());

            while let Some(current_node) = current_node_opt {
                let (current_node_name, successors_clone) = match &current_node {
                    NodeType::Sync(h) => {
                        let node_locked = h.node_impl.lock().unwrap();
                        (node_locked.name(), node_locked.successors().clone())
                    }
                    NodeType::Async(h) => {
                        let node_locked = h.node_impl.lock().await;
                        (node_locked.name(), node_locked.successors().clone())
                    }
                    NodeType::SyncFlow(f) => {
                        let node_locked = f.node_impl.lock().unwrap();
                        (node_locked.name(), node_locked.successors().clone())
                    }
                    NodeType::AsyncFlow(f) => {
                        let node_locked = f.node_impl.lock().await;
                        (node_locked.name(), node_locked.successors().clone())
                    }
                };

                // Set Params
                current_node.set_params(initial_params.clone());

                let run_result = match &current_node {
                    NodeType::Sync(h) => {
                        let mut node_locked = h.node_impl.lock().unwrap();
                        node_locked._run(shared)
                    }
                    NodeType::Async(h) => h.run_async(shared).await,
                    NodeType::SyncFlow(f) => {
                        let mut node_locked = f.node_impl.lock().unwrap();
                        node_locked._run(shared)
                    }
                    NodeType::AsyncFlow(f) => {
                        // Use Box::pin to break the recursion
                        Box::pin(f.run_async(shared)).await
                    }
                };

                match run_result {
                    Ok(action) => {
                        last_action = Ok(action.clone());
                        let flow_impl_guard = flow.node_impl.lock().await;
                        current_node_opt = flow_impl_guard.base.get_next_node(
                            &current_node_name,
                            &successors_clone,
                            &action,
                        );
                    }
                    Err(e) => {
                        return Err(FlowError::OrchestrationError(format!(
                            "AsyncFlow '{}' failed at node '{}': {}",
                            flow_name, current_node_name, e
                        )));
                    }
                }
            }
            last_action
        }

        // Call the inner function
        Box::pin(run_inner_impl(self, shared)).await
    }

    // Helper to convert handle to NodeType for graph building
    pub fn into_nodetype(self) -> NodeType<S, P> {
        NodeType::AsyncFlow(self)
    }

    // Expose base methods via the handle
    pub fn name(&self) -> String
    where
        P: Default,
    {
        // Use a simple executor that works in any context
        futures::executor::block_on(async { self.node_impl.lock().await.name() })
    }

    pub fn set_params(&self, params: P) {
        // Use a simple executor that works in any context
        futures::executor::block_on(async {
            self.node_impl.lock().await.base.params = params;
        });
    }
}

// Implement NodeLike for AsyncFlowImpl to allow nesting
#[async_trait]
impl<S: SharedState, P: Params + Default> NodeLike<S, P> for AsyncFlowImpl<S, P>
where
    S: SharedState,
    P: Params + Default + Clone,
{
    fn name(&self) -> String {
        self.base.name.clone()
    }
    fn set_params(&mut self, params: P) {
        self.base.params = params;
    }
    fn successors(&self) -> &HashMap<String, NodeType<S, P>> {
        &self.base.successors
    }
    fn successors_mut(&mut self) -> &mut HashMap<String, NodeType<S, P>> {
        &mut self.base.successors
    }

    /// Running an AsyncFlow synchronously is an error
    fn _run(&mut self, _shared: &mut S) -> PostResult {
        Err(FlowError::ConfigurationError(
            "Attempted to run an AsyncFlow synchronously via _run.".into(),
        ))
    }

    /// Running an AsyncFlow asynchronously orchestrates its nodes.
    async fn _run_async(&mut self, shared: &mut S) -> PostResult {
        // Using an inner function to break the recursion
        async fn run_inner_impl<S: SharedState, P: Params + Default + Clone>(
            flow_impl: &mut AsyncFlowImpl<S, P>,
            shared: &mut S,
        ) -> PostResult {
            let mut current_node_opt = flow_impl.base.start_node.clone();
            let mut last_action = Ok("".to_string());
            let flow_name = flow_impl.base.name.clone();
            let flow_params = flow_impl.base.params.clone();

            while let Some(current_node) = current_node_opt {
                let (current_node_name, successors_clone) = match &current_node {
                    NodeType::Sync(h) => {
                        let node_locked = h.node_impl.lock().unwrap();
                        (node_locked.name(), node_locked.successors().clone())
                    }
                    NodeType::Async(h) => {
                        let node_locked = h.node_impl.lock().await;
                        (node_locked.name(), node_locked.successors().clone())
                    }
                    NodeType::SyncFlow(f) => {
                        let node_locked = f.node_impl.lock().unwrap();
                        (node_locked.name(), node_locked.successors().clone())
                    }
                    NodeType::AsyncFlow(f) => {
                        let node_locked = f.node_impl.lock().await;
                        (node_locked.name(), node_locked.successors().clone())
                    }
                };

                current_node.set_params(flow_params.clone());

                let run_result = match &current_node {
                    NodeType::Sync(h) => {
                        let mut node_locked = h.node_impl.lock().unwrap();
                        node_locked._run(shared)
                    }
                    NodeType::Async(h) => h.run_async(shared).await,
                    NodeType::SyncFlow(f) => {
                        let mut node_locked = f.node_impl.lock().unwrap();
                        node_locked._run(shared)
                    }
                    NodeType::AsyncFlow(f) => {
                        // Use Box::pin to break the recursion
                        Box::pin(f.run_async(shared)).await
                    }
                };

                match run_result {
                    Ok(action) => {
                        last_action = Ok(action.clone());
                        current_node_opt = flow_impl.base.get_next_node(
                            &current_node_name,
                            &successors_clone,
                            &action,
                        );
                    }
                    Err(e) => {
                        return Err(FlowError::OrchestrationError(format!(
                            "Nested AsyncFlow '{}' failed at node '{}': {}",
                            flow_name, current_node_name, e
                        )));
                    }
                }
            }
            last_action
        }

        // Call the inner function
        Box::pin(run_inner_impl(self, shared)).await
    }
}

// --- Operator Overloading ---

/// Temporary struct to hold the source node and action for conditional transitions.
pub struct ConditionalTransition<S: SharedState, P: Params + Default> {
    source_node: NodeType<S, P>,
    action: String,
}

// Overload `NodeType - "action"`
impl<S: SharedState, P: Params + Default> Sub<String> for NodeType<S, P> {
    type Output = ConditionalTransition<S, P>;

    fn sub(self, action: String) -> Self::Output {
        ConditionalTransition {
            source_node: self,
            action,
        }
    }
}
// Also implement for references if needed, e.g., &NodeType - "action"
impl<'a, S: SharedState, P: Params + Default> Sub<&'a str> for NodeType<S, P> {
    type Output = ConditionalTransition<S, P>;
    fn sub(self, action: &'a str) -> Self::Output {
        ConditionalTransition {
            source_node: self,
            action: action.to_string(),
        }
    }
}

// Overload `NodeType >> NodeType` (default action)
impl<S: SharedState, P: Params + Default> Shr<NodeType<S, P>> for NodeType<S, P>
where
    P: Default,
{
    type Output = NodeType<S, P>; // Return the target node for chaining

    fn shr(self, target_node: NodeType<S, P>) -> Self::Output {
        self.with_successors_mut(|successors| {
            if successors.contains_key("default") {
                warn!(
                    "Node '{}': Overwriting successor for default action",
                    self.name()
                );
            }
            successors.insert("default".to_string(), target_node.clone());
        });
        target_node
    }
}

// Overload `ConditionalTransition >> NodeType` (specific action)
impl<S: SharedState, P: Params + Default> Shr<NodeType<S, P>> for ConditionalTransition<S, P>
where
    P: Default,
{
    type Output = NodeType<S, P>; // Return the target node

    fn shr(self, target_node: NodeType<S, P>) -> Self::Output {
        self.source_node.with_successors_mut(|successors| {
            if successors.contains_key(&self.action) {
                warn!(
                    "Node '{}': Overwriting successor for action '{}'",
                    self.source_node.name(),
                    self.action
                );
            }
            successors.insert(self.action.clone(), target_node.clone());
        });
        target_node
    }
}

// --- Batch Nodes/Flows (Conceptual) ---

// Batch processing requires defining how inputs map to S and P,
// and how results are aggregated. This adds significant complexity.
// Below are sketches.

// --- Sync Batch Node ---
pub trait SyncBatchLogic<S: SharedState, P: Params + Default, Item: Send + Sync>:
    Send + Sync
{
    fn name(&self) -> String;
    // Prep now returns a list of items or inputs for the batch
    fn prep_batch(&mut self, shared: &mut S, params: &P) -> Result<Vec<Item>, FlowError>;
    // Exec takes a single item from the batch
    fn exec_item(&mut self, item: Item, params: &P) -> ExecResult;
    // Post might aggregate results or operate on the shared state
    fn post_batch(
        &mut self,
        shared: &mut S,
        prep_res: Vec<Item>,
        exec_results: Vec<ExecResult>,
        params: &P,
    ) -> PostResult;
    // Fallback might apply per-item or for the whole batch
    fn exec_item_fallback(&mut self, item: Item, error: FlowError, params: &P) -> ExecResult;
}

// TODO: Implement SyncBatchNodeImpl using SyncBatchLogic, overriding _run to iterate.
// TODO: Implement SyncBatchNodeHandle.

// --- Async Batch Node ---
#[async_trait]
pub trait AsyncBatchLogic<S: SharedState, P: Params + Default, Item: Send + Sync>:
    Send + Sync
{
    fn name(&self) -> String;
    async fn prep_batch_async(
        &mut self,
        shared: &mut S,
        params: &P,
    ) -> Result<Vec<Item>, FlowError>;
    async fn exec_item_async(&mut self, item: Item, params: &P) -> ExecResult;
    async fn post_batch_async(
        &mut self,
        shared: &mut S,
        prep_res: Vec<Item>,
        exec_results: Vec<ExecResult>,
        params: &P,
    ) -> PostResult;
    async fn exec_item_fallback_async(
        &mut self,
        item: Item,
        error: FlowError,
        params: &P,
    ) -> ExecResult;
}

// TODO: Implement AsyncBatchNodeImpl.
// TODO: Implement AsyncParallelBatchNodeImpl using join_all(exec_item_async for item in items).
// TODO: Implement AsyncBatchNodeHandle, AsyncParallelBatchNodeHandle.

// --- Batch Flows ---
// Batch flows would override _run / _run_async.
// Prep would return batch data (e.g., Vec<P> or Vec<Item>).
// The run method would loop through batch items, calling the _orch method for each.

// TODO: Implement BatchFlowImpl / AsyncBatchFlowImpl / AsyncParallelBatchFlowImpl.
// TODO: Implement BatchFlow / AsyncBatchFlow / AsyncParallelBatchFlow handles.

// --- Example Usage ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap; // Alias standard HashMap

    // --- Define Concrete State and Params ---
    #[derive(Clone, Debug, Default)]
    struct MySharedState {
        counter: i32,
        log: Vec<String>,
    }

    #[derive(Clone, Debug, Default)]
    struct MyParams {
        node_specific_value: Option<i32>,
        global_modifier: i32,
    }

    // --- Define Node Logic ---

    #[derive(Debug, Default)] // Added Default
    struct GreeterLogic;
    impl SyncLogic<MySharedState, MyParams> for GreeterLogic {
        fn name(&self) -> String {
            "Greeter".to_string()
        }

        fn prep(&mut self, shared: &mut MySharedState, params: &MyParams) -> PrepResult {
            shared.log.push(format!(
                "{}: Preparing greeting. State counter: {}",
                self.name(),
                shared.counter
            ));
            // Prep result can be anything, here just a unit type inside Box<dyn Any>
            Ok(Box::new(()))
        }

        fn exec(&mut self, _prep_res: AnySendSync, params: &MyParams) -> ExecResult {
            let modifier = params.node_specific_value.unwrap_or(1) + params.global_modifier;
            println!("{}: Executing! Modifier: {}", self.name(), modifier);
            if modifier < 0 {
                Err(FlowError::Generic("Modifier is negative!".into()))
            } else {
                // Exec result can be anything, here a simple string
                Ok(Box::new(format!("Hello there! Modifier was {}", modifier)))
            }
        }

        fn post(
            &mut self,
            shared: &mut MySharedState,
            _prep_res: AnySendSync,
            exec_res: ExecResult,
            _params: &MyParams,
        ) -> PostResult {
            match exec_res {
                Ok(result_boxed) => {
                    // Downcast the result if needed
                    let message = result_boxed.downcast_ref::<String>().unwrap();
                    shared
                        .log
                        .push(format!("{}: Post: Success - {}", self.name(), message));
                    shared.counter += 1;
                    Ok("success".to_string()) // Action to proceed
                }
                Err(e) => {
                    shared
                        .log
                        .push(format!("{}: Post: Failure - {:?}", self.name(), e));
                    shared.counter -= 1;
                    Ok("failure".to_string()) // Action on failure
                }
            }
        }

        fn exec_fallback(
            &mut self,
            _prep_res: AnySendSync,
            error: FlowError,
            _params: &MyParams,
        ) -> ExecResult {
            println!(
                "{}: Running fallback due to error: {:?}",
                self.name(),
                error
            );
            // Fallback can try to return a default value or propagate a different error
            Ok(Box::new("Fallback greeting executed".to_string()))
        }
    }

    #[derive(Debug, Default)] // Added Default
    struct FarewellLogic;
    #[async_trait]
    impl AsyncLogic<MySharedState, MyParams> for FarewellLogic {
        fn name(&self) -> String {
            "Fareweller".to_string()
        }

        async fn prep_async(
            &mut self,
            shared: &mut MySharedState,
            _params: &MyParams,
        ) -> PrepResult {
            shared.log.push(format!(
                "{}: Preparing farewell async. State counter: {}",
                self.name(),
                shared.counter
            ));
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(Box::new(()))
        }
        async fn exec_async(&mut self, _prep_res: AnySendSync, params: &MyParams) -> ExecResult {
            println!(
                "{}: Executing farewell async! Global modifier: {}",
                self.name(),
                params.global_modifier
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok(Box::new("Goodbye async!".to_string()))
        }
        async fn post_async(
            &mut self,
            shared: &mut MySharedState,
            _prep_res: AnySendSync,
            exec_res: ExecResult,
            _params: &MyParams,
        ) -> PostResult {
            match exec_res {
                Ok(result_boxed) => {
                    let message = result_boxed.downcast_ref::<String>().unwrap();
                    shared.log.push(format!(
                        "{}: Post async: Success - {}",
                        self.name(),
                        message
                    ));
                    shared.counter += 10;
                    Ok("done".to_string()) // Final action
                }
                Err(e) => {
                    shared
                        .log
                        .push(format!("{}: Post async: Failure - {:?}", self.name(), e));
                    shared.counter -= 10;
                    Ok("error_exit".to_string())
                }
            }
        }
        async fn exec_fallback_async(
            &mut self,
            _prep_res: AnySendSync,
            error: FlowError,
            _params: &MyParams,
        ) -> ExecResult {
            println!(
                "{}: Running async fallback due to error: {:?}",
                self.name(),
                error
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
            Ok(Box::new("Fallback farewell async executed".to_string()))
        }
    }

    #[derive(Debug, Default)] // Added Default
    struct ErrorLogic;
    impl SyncLogic<MySharedState, MyParams> for ErrorLogic {
        fn name(&self) -> String {
            "ErrorNode".to_string()
        }
        fn prep(&mut self, shared: &mut MySharedState, _params: &MyParams) -> PrepResult {
            Ok(Box::new(()))
        }
        fn exec(&mut self, _prep_res: AnySendSync, _params: &MyParams) -> ExecResult {
            Err(FlowError::Generic("Intentional exec error".into()))
        }
        fn post(
            &mut self,
            shared: &mut MySharedState,
            _prep_res: AnySendSync,
            exec_res: ExecResult,
            _params: &MyParams,
        ) -> PostResult {
            match exec_res {
                Ok(result_boxed) => {
                    let default_msg = "Unknown fallback result".to_string(); // Bind to variable
                    let message = result_boxed
                        .downcast_ref::<String>()
                        .unwrap_or(&default_msg);
                    shared.log.push(format!(
                        "{}: Post after fallback success: {}",
                        self.name(),
                        message
                    ));
                    Ok("fallback_handled".to_string())
                }
                Err(e) => {
                    shared.log.push(format!(
                        "{}: Post after error/fallback failure: {:?}",
                        self.name(),
                        e
                    ));
                    Ok("error_handled".to_string())
                }
            }
        }
        fn exec_fallback(
            &mut self,
            _prep_res: AnySendSync,
            error: FlowError,
            _params: &MyParams,
        ) -> ExecResult {
            println!("{}: Running fallback for: {:?}", self.name(), error);
            Ok(Box::new("Error fallback handled".to_string()))
        }
    }

    #[test]
    fn test_sync_flow() {
        // Create nodes
        let greeter = SyncNodeHandle::new(GreeterLogic, 1, 0).into_nodetype();
        let error_node = SyncNodeHandle::new(ErrorLogic, 2, 1).into_nodetype(); // Error node with retry

        // Create flow
        let flow = Flow::new("SyncGreetingFlow");

        // Build graph using operators
        // Build more complex graph
        let conditional_transition = greeter.clone() - "success"; // greeter - "success"
        conditional_transition >> error_node.clone(); // >> error_node

        let conditional_failure = greeter.clone() - "failure";
        // conditional_failure >> ??? // Define failure path if needed

        flow.start(greeter.clone()); // Set the start node

        // Initial state
        let mut state = MySharedState::default();

        // Run flow
        let result = flow.run(&mut state);

        println!("Sync Flow Result: {:?}", result);
        println!("Final State: {:?}", state);

        assert!(result.is_ok());
        // Error node fallback runs, post uses its result ("should_not_reach" isn't used because fallback succeeds)
        // Error node's post *would* run if fallback failed, returning *its* action.
        // The Flow returns the *action* from the *last node* that ran its post successfully.
        // Since error_node's fallback succeeded, its *post* method runs. Let's adjust ErrorLogic post.

        // Rerun with adjusted ErrorLogic post if needed or check state.
        // ErrorLogic's fallback results in Ok("Error fallback handled").
        // ErrorLogic's post should return the next action based on that Ok result.
        // Let's assume ErrorLogic post is called after successful fallback and returns "fallback_handled"
        // Modify ErrorLogic post:
        /*
         fn post(&mut self, shared: &mut S, _prep_res: AnySendSync, exec_res: ExecResult, _params: &P) -> PostResult {
             match exec_res {
                 Ok(_) => { // Fallback succeeded
                      shared.log.push(format!("{}: Post after fallback success.", self.name()));
                      Ok("fallback_handled".to_string())
                 }
                 Err(_) => { // Should not happen if fallback returns Ok
                      shared.log.push(format!("{}: Post after fallback failure?", self.name()));
                      Ok("fallback_failed".to_string())
                 }
             }
         }
        */
        // With the above change, the result should be Ok("fallback_handled")
        assert_eq!(result.unwrap(), "fallback_handled");
        assert!(state
            .log
            .iter()
            .any(|s| s.contains("Greeter: Post: Success")));
        assert!(state
            .log
            .iter()
            .any(|s| s.contains("ErrorNode: Post after fallback success")));
        assert_eq!(state.counter, 1); // Greeter +1, ErrorNode -1 (from original post logic before fallback)
                                      // Counter depends heavily on when/if post runs after fallback
    }

    #[tokio::test]
    async fn test_async_flow() {
        // Create nodes
        let greeter = SyncNodeHandle::new(GreeterLogic, 1, 0).into_nodetype(); // sync node
        let fareweller = AsyncNodeHandle::new(FarewellLogic, 1, 0).into_nodetype(); // Async node

        // Create flow
        let flow = AsyncFlow::new("AsyncGreetingFlow");

        // Build graph
        // Use a runtime just for initialization - this is safe as it's not nested in an async context
        //let runtime = tokio::runtime::Handle::current();
        //runtime.block_on(async {
        flow.node_impl
            .lock()
            .await
            .base
            .set_start_node(greeter.clone());
        //});

        // Define transitions
        let _ = greeter.clone() - "success" >> fareweller.clone();
        let _ = greeter.clone() - "failure" >> fareweller.clone(); // Also go to farewell on failure

        // Initial state
        let mut state = MySharedState::default();

        // Run flow
        let result = flow.run_async(&mut state).await;

        println!("Async Flow Result: {:?}", result);
        println!("Final State: {:?}", state);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "done"); // Action from fareweller
        assert_eq!(state.counter, 11); // Greeter +1, Fareweller +10
        assert!(state
            .log
            .iter()
            .any(|s| s.contains("Greeter: Post: Success")));
        assert!(state
            .log
            .iter()
            .any(|s| s.contains("Fareweller: Post async: Success")));
    }

    #[tokio::test]
    async fn test_node_direct_run() {
        let mut state = MySharedState::default();
        let greeter_handle = SyncNodeHandle::<MySharedState, MyParams>::new(GreeterLogic, 1, 0);
        let result = greeter_handle.run(&mut state); // Run sync node directly
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(state.counter, 1);

        let mut state2 = MySharedState::default();
        let fareweller_handle =
            AsyncNodeHandle::<MySharedState, MyParams>::new(FarewellLogic, 1, 0);
        let result2 = fareweller_handle.run_async(&mut state2).await; // Run async node directly
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), "done");
        assert_eq!(state2.counter, 10);
    }
}
fn main() {
    println!("Hello, world!");
}
