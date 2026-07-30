use isanagent::acp::types::*;
use isanagent::bus::BusMessage;
use isanagent::channels::acp::AcpChannel;
use tokio::sync::mpsc;

#[tokio::test]
async fn test_acp_channel_initialize_and_session_new() {
    let channel = AcpChannel::new();
    let (bus_tx, _bus_rx) = mpsc::channel::<BusMessage>(10);
    let (raw_writer_tx, raw_writer_rx) = mpsc::channel::<String>(10);

    // 1. Initialize Handshake
    let init_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": { "readTextFile": true, "writeTextFile": true },
                "terminal": true
            },
            "clientInfo": {
                "name": "zed",
                "title": "Zed Editor",
                "version": "0.180.0"
            }
        }
    });

    let resp_str = channel
        .handle_incoming_rpc(&init_req.to_string(), &bus_tx, &raw_writer_tx)
        .await
        .expect("initialize response");

    let resp: JsonRpcResponse = serde_json::from_str(&resp_str).expect("parse initialize response");
    assert_eq!(resp.id, serde_json::json!(1));
    assert!(resp.error.is_none());

    let init_res: InitializeResult =
        serde_json::from_value(resp.result.unwrap()).expect("parse result");
    assert_eq!(init_res.protocol_version, 1);
    assert_eq!(init_res.agent_info.name, "isanagent");
    assert!(init_res.agent_capabilities.load_session);

    // 2. session/new
    let session_new_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "session/new",
        "params": {
            "cwd": "/tmp/test-project",
            "mcpServers": []
        }
    });

    let session_resp_str = channel
        .handle_incoming_rpc(&session_new_req.to_string(), &bus_tx, &raw_writer_tx)
        .await
        .expect("session/new response");

    let session_resp: JsonRpcResponse =
        serde_json::from_str(&session_resp_str).expect("parse session/new response");
    assert_eq!(session_resp.id, serde_json::json!(2));

    let session_res: SessionNewResult =
        serde_json::from_value(session_resp.result.unwrap()).expect("parse session result");
    assert!(session_res.session_id.starts_with("acp_sess_"));

    // 3. session/close
    let close_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "session/close",
        "params": {
            "sessionId": session_res.session_id
        }
    });

    let close_resp_str = channel
        .handle_incoming_rpc(&close_req.to_string(), &bus_tx, &raw_writer_tx)
        .await
        .expect("session/close response");

    let close_resp: JsonRpcResponse =
        serde_json::from_str(&close_resp_str).expect("parse close response");
    assert_eq!(close_resp.id, serde_json::json!(3));
    assert!(close_resp.error.is_none());

    drop(raw_writer_rx);
}

#[tokio::test]
async fn test_acp_channel_prompt_cancellation() {
    let channel = AcpChannel::new();
    let (bus_tx, mut bus_rx) = mpsc::channel::<BusMessage>(10);
    let (raw_writer_tx, mut raw_writer_rx) = mpsc::channel::<String>(10);

    // 1. Create Session
    let session_new_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "session/new",
        "params": {
            "cwd": "/tmp/test-cancel",
            "mcpServers": []
        }
    });

    let session_resp_str = channel
        .handle_incoming_rpc(&session_new_req.to_string(), &bus_tx, &raw_writer_tx)
        .await
        .expect("session/new response");

    let session_resp: JsonRpcResponse = serde_json::from_str(&session_resp_str).unwrap();
    let session_res: SessionNewResult =
        serde_json::from_value(session_resp.result.unwrap()).unwrap();

    // 2. Prompt request (will wait for bus completion or cancel)
    let prompt_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "session/prompt",
        "params": {
            "sessionId": session_res.session_id,
            "prompt": [
                { "type": "text", "text": "Run long task" }
            ]
        }
    });

    let prompt_ret = channel
        .handle_incoming_rpc(&prompt_req.to_string(), &bus_tx, &raw_writer_tx)
        .await;
    assert!(prompt_ret.is_none()); // Returns None because response is async via oneshot

    // Verify inbound message dispatched to bus
    let bus_msg = bus_rx.recv().await.expect("bus message received");
    if let BusMessage::Inbound(inbound) = bus_msg {
        assert_eq!(inbound.chat_id, session_res.session_id);
        assert_eq!(inbound.content, "Run long task");
    } else {
        panic!("expected BusMessage::Inbound");
    }

    // 3. Send session/cancel
    let cancel_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 12,
        "method": "session/cancel",
        "params": {
            "sessionId": session_res.session_id
        }
    });

    let cancel_resp_str = channel
        .handle_incoming_rpc(&cancel_req.to_string(), &bus_tx, &raw_writer_tx)
        .await
        .expect("session/cancel response");

    let cancel_resp: JsonRpcResponse = serde_json::from_str(&cancel_resp_str).unwrap();
    assert_eq!(cancel_resp.id, serde_json::json!(12));

    // Check async prompt response sent via raw_writer_tx
    let async_prompt_line = raw_writer_rx.recv().await.expect("async prompt response");
    let async_prompt_resp: JsonRpcResponse = serde_json::from_str(&async_prompt_line).unwrap();
    assert_eq!(async_prompt_resp.id, serde_json::json!(11));

    let prompt_res: SessionPromptResult =
        serde_json::from_value(async_prompt_resp.result.unwrap()).unwrap();
    assert_eq!(prompt_res.stop_reason, AcpStopReason::Cancelled);
}
