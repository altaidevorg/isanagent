use agent_rs::*;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
struct MyState {
    context: String,
    memory: HashMap<String, String>,
    conversation_history: Vec<String>,
}

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

fn main() {
    // Create nodes
    let prompt_node = SyncNodeHandle::new(GuardrailNode, 2, 1).into_nodetype();
    let response_node = SyncNodeHandle::new(ResponseNode, 1, 0).into_nodetype();

    // Create flow
    let flow = Flow::<MyState, EmptyParams>::new("AgentConversationFlow");

    // Build the graph with conditional routing
    flow.start(prompt_node.clone());

    // Route based on action strings
    let _ = prompt_node.clone() - "assist" >> response_node.clone();
    let _ = prompt_node.clone() - "default" >> prompt_node.clone();

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
