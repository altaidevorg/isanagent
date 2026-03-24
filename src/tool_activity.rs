use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type ToolExecutionActivityHandleFuture = Pin<Box<dyn Future<Output = ()> + Send>>;
pub type SharedToolExecutionActivity = Arc<dyn ToolExecutionActivity>;

pub trait ToolExecutionActivity: Send + Sync {
    fn start(&self, chat_id: &str, tool_name: &str) -> Box<dyn ToolExecutionActivityHandle>;
}

pub trait ToolExecutionActivityHandle: Send {
    fn stop(self: Box<Self>) -> ToolExecutionActivityHandleFuture;
}
