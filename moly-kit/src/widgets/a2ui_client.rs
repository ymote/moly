//! A2UI-aware BotClient wrapper
//!
//! When A2UI is enabled, this client bypasses the wrapped client's streaming
//! and makes direct non-streaming HTTP calls to avoid SSE parsing issues.

use crate::aitk::protocol::{
    Bot, BotId, ClientResult, EntityId, MessageContent, Tool, ToolCall,
};
use crate::aitk::protocol::{BotClient, Message};
use crate::aitk::utils::asynchronous::{BoxPlatformSendFuture, BoxPlatformSendStream};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Global A2UI enabled flag - can be set from PromptInput and read by A2uiClient.
static GLOBAL_A2UI_ENABLED: AtomicBool = AtomicBool::new(false);

/// Set the global A2UI enabled state (call from PromptInput toggle).
pub fn set_global_a2ui_enabled(enabled: bool) {
    ::log::info!("[A2UI] Global enabled set to: {}", enabled);
    GLOBAL_A2UI_ENABLED.store(enabled, Ordering::SeqCst);
}

/// Get the global A2UI enabled state.
pub fn is_global_a2ui_enabled() -> bool {
    GLOBAL_A2UI_ENABLED.load(Ordering::SeqCst)
}

use std::sync::Mutex;

/// Global pending A2UI tool calls - written by Chat widget, read by App.
static GLOBAL_A2UI_TOOL_CALLS: Mutex<Vec<ToolCall>> = Mutex::new(Vec::new());

/// Store pending A2UI tool calls (called from Chat's emit_a2ui_tool_calls).
pub fn set_pending_a2ui_tool_calls(tool_calls: Vec<ToolCall>) {
    ::log::info!(
        "[A2UI] Storing {} pending tool calls",
        tool_calls.len()
    );
    *GLOBAL_A2UI_TOOL_CALLS.lock().unwrap() = tool_calls;
}

/// Take pending A2UI tool calls (called by App, clears the buffer).
pub fn take_pending_a2ui_tool_calls() -> Vec<ToolCall> {
    let mut lock = GLOBAL_A2UI_TOOL_CALLS.lock().unwrap();
    std::mem::take(&mut *lock)
}

// ============================================================================
// Non-streaming response types (OpenAI chat completions format)
// ============================================================================

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<CompletionChoice>,
}

#[derive(Debug, Deserialize)]
struct CompletionChoice {
    message: CompletionMessage,
}

#[derive(Debug, Deserialize)]
struct CompletionMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
}

#[derive(Debug, Deserialize)]
struct ResponseToolCall {
    id: String,
    function: ResponseFunction,
}

#[derive(Debug, Deserialize)]
struct ResponseFunction {
    name: String,
    arguments: String,
}

// ============================================================================
// Outgoing request types (for building the non-streaming request)
// ============================================================================

#[derive(Serialize)]
struct OutgoingMessage {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct OutgoingTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: OutgoingFunction,
}

#[derive(Serialize)]
struct OutgoingFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

// ============================================================================
// A2UI tool definitions
// ============================================================================

/// A2UI tool definitions in aitk Tool format.
fn get_a2ui_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "create_text".to_string(),
            description: Some(
                "Create a text/label component to display static or dynamic text"
                    .to_string(),
            ),
            input_schema: Arc::new(
                serde_json::from_str(r#"{
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique component ID"},
                        "text": {"type": "string", "description": "Static text"},
                        "dataPath": {"type": "string", "description": "JSON pointer"},
                        "style": {"type": "string", "enum": ["h1","h3","caption","body"]}
                    },
                    "required": ["id"]
                }"#).expect("invalid create_text schema"),
            ),
        },
        Tool {
            name: "create_button".to_string(),
            description: Some(
                "Create a clickable button that triggers an action".to_string(),
            ),
            input_schema: Arc::new(
                serde_json::from_str(r#"{
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "action": {"type": "string"},
                        "primary": {"type": "boolean"}
                    },
                    "required": ["id", "label", "action"]
                }"#).expect("invalid create_button schema"),
            ),
        },
        Tool {
            name: "create_textfield".to_string(),
            description: Some("Create a text input field".to_string()),
            input_schema: Arc::new(
                serde_json::from_str(r#"{
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "dataPath": {"type": "string"},
                        "placeholder": {"type": "string"}
                    },
                    "required": ["id", "dataPath"]
                }"#).expect("invalid create_textfield schema"),
            ),
        },
        Tool {
            name: "create_checkbox".to_string(),
            description: Some("Create a checkbox toggle".to_string()),
            input_schema: Arc::new(
                serde_json::from_str(r#"{
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "dataPath": {"type": "string"}
                    },
                    "required": ["id", "label", "dataPath"]
                }"#).expect("invalid create_checkbox schema"),
            ),
        },
        Tool {
            name: "create_slider".to_string(),
            description: Some("Create a slider for numeric values".to_string()),
            input_schema: Arc::new(
                serde_json::from_str(r#"{
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "dataPath": {"type": "string"},
                        "min": {"type": "number"},
                        "max": {"type": "number"},
                        "step": {"type": "number"}
                    },
                    "required": ["id", "dataPath", "min", "max"]
                }"#).expect("invalid create_slider schema"),
            ),
        },
        Tool {
            name: "create_card".to_string(),
            description: Some("Create a card container".to_string()),
            input_schema: Arc::new(
                serde_json::from_str(r#"{
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "childId": {"type": "string"}
                    },
                    "required": ["id", "childId"]
                }"#).expect("invalid create_card schema"),
            ),
        },
        Tool {
            name: "create_column".to_string(),
            description: Some("Create a vertical layout container".to_string()),
            input_schema: Arc::new(
                serde_json::from_str(r#"{
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "children": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["id", "children"]
                }"#).expect("invalid create_column schema"),
            ),
        },
        Tool {
            name: "create_row".to_string(),
            description: Some("Create a horizontal layout container".to_string()),
            input_schema: Arc::new(
                serde_json::from_str(r#"{
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "children": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["id", "children"]
                }"#).expect("invalid create_row schema"),
            ),
        },
        Tool {
            name: "set_data".to_string(),
            description: Some("Set initial data value in the data model".to_string()),
            input_schema: Arc::new(
                serde_json::from_str(r#"{
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "stringValue": {"type": "string"},
                        "numberValue": {"type": "number"},
                        "booleanValue": {"type": "boolean"}
                    },
                    "required": ["path"]
                }"#).expect("invalid set_data schema"),
            ),
        },
        Tool {
            name: "render_ui".to_string(),
            description: Some(
                "Finalize and render the UI. Call this LAST.".to_string(),
            ),
            input_schema: Arc::new(
                serde_json::from_str(r#"{
                    "type": "object",
                    "properties": {
                        "rootId": {"type": "string"},
                        "title": {"type": "string"}
                    },
                    "required": ["rootId"]
                }"#).expect("invalid render_ui schema"),
            ),
        },
    ]
}

/// A2UI system prompt to guide the LLM.
const A2UI_SYSTEM_PROMPT: &str = r#"You are an A2UI generator assistant. \
Create user interfaces by calling the provided tools.

RULES:
1. Create components using tools (create_text, create_button, create_slider, etc.)
2. Use create_column for vertical layouts, create_row for horizontal layouts
3. Use create_card to wrap sections in styled containers
4. Set initial data values with set_data for any bound components
5. ALWAYS call render_ui as the LAST step with the root component ID
6. Use descriptive IDs like "title", "volume-slider", "submit-btn"
7. For sliders/checkboxes, always set initial data with set_data

Example flow for "create a volume control":
1. create_text(id="volume-label", text="Volume", style="body")
2. create_slider(id="volume-slider", dataPath="/volume", min=0, max=100, step=1)
3. create_column(id="root", children=["volume-label", "volume-slider"])
4. set_data(path="/volume", numberValue=50)
5. render_ui(rootId="root")
"#;

// ============================================================================
// A2uiClient
// ============================================================================

/// A wrapper around a BotClient that injects A2UI tools when enabled.
///
/// When A2UI is enabled, it makes direct non-streaming HTTP calls to bypass
/// aitk's SSE parser which cannot handle chunked tool call responses.
pub struct A2uiClient {
    client: Box<dyn BotClient>,
    a2ui_enabled: Arc<AtomicBool>,
    /// API base URL (e.g. "https://api.moonshot.ai/v1")
    api_url: String,
    /// API key for authentication.
    api_key: String,
}

impl Clone for A2uiClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone_box(),
            a2ui_enabled: self.a2ui_enabled.clone(),
            api_url: self.api_url.clone(),
            api_key: self.api_key.clone(),
        }
    }
}

impl A2uiClient {
    /// Create a new A2UI-aware client wrapper.
    ///
    /// `api_url` and `api_key` are used for direct non-streaming calls
    /// when A2UI is enabled.
    pub fn new(
        client: Box<dyn BotClient>,
        api_url: String,
        api_key: String,
    ) -> Self {
        Self {
            client,
            a2ui_enabled: Arc::new(AtomicBool::new(false)),
            api_url,
            api_key,
        }
    }

    /// Enable or disable A2UI tool injection.
    pub fn set_a2ui_enabled(&self, enabled: bool) {
        self.a2ui_enabled.store(enabled, Ordering::SeqCst);
    }

    /// Check if A2UI is currently enabled.
    pub fn is_a2ui_enabled(&self) -> bool {
        self.a2ui_enabled.load(Ordering::SeqCst)
    }
}

/// Convert aitk Message to one or more OpenAI outgoing messages.
///
/// Tool result messages with multiple results are expanded into one
/// outgoing message per result, since the OpenAI API requires each
/// tool result to have its own message with a unique `tool_call_id`.
fn to_outgoing_messages(msg: &Message) -> Vec<OutgoingMessage> {
    let role = match &msg.from {
        EntityId::User => "user",
        EntityId::System => "system",
        EntityId::Bot(_) => "assistant",
        EntityId::Tool => "tool",
        EntityId::App => "user",
    };

    // Tool result messages: expand one message per result
    if !msg.content.tool_results.is_empty() {
        return msg
            .content
            .tool_results
            .iter()
            .map(|r| OutgoingMessage {
                role: "tool",
                content: r.content.clone(),
                tool_calls: None,
                tool_call_id: Some(r.tool_call_id.clone()),
            })
            .collect();
    }

    let tool_calls = if !msg.content.tool_calls.is_empty() {
        Some(
            msg.content
                .tool_calls
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": serde_json::to_string(&tc.arguments)
                                .unwrap_or_default()
                        }
                    })
                })
                .collect(),
        )
    } else {
        None
    };

    vec![OutgoingMessage {
        role,
        content: msg.content.text.clone(),
        tool_calls,
        tool_call_id: None,
    }]
}

/// Convert aitk Tool to OpenAI function tool format.
fn to_outgoing_tool(tool: &Tool) -> OutgoingTool {
    let parameters =
        serde_json::Value::Object((*tool.input_schema).clone());
    OutgoingTool {
        tool_type: "function",
        function: OutgoingFunction {
            name: tool.name.clone(),
            description: tool
                .description
                .as_deref()
                .unwrap_or("")
                .to_string(),
            parameters,
        },
    }
}

/// Make a direct non-streaming HTTP call to the OpenAI-compatible API.
async fn send_non_streaming(
    api_url: String,
    api_key: String,
    model: String,
    messages: Vec<OutgoingMessage>,
    tools: Vec<OutgoingTool>,
) -> Result<MessageContent, String> {
    let url = format!(
        "{}/chat/completions",
        api_url.trim_end_matches('/')
    );

    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": false,
    });

    if !tools.is_empty() {
        body["tools"] = serde_json::to_value(&tools)
            .map_err(|e| format!("Failed to serialize tools: {e}"))?;
    }

    ::log::info!(
        "[A2UI] Non-streaming POST to {} with model {}",
        url,
        model
    );

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<no body>".to_string());
        return Err(format!(
            "API returned status {}: {}",
            status, body
        ));
    }

    let text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response body: {e}"))?;

    ::log::info!("[A2UI] Response length: {} bytes", text.len());

    let completion: ChatCompletionResponse =
        serde_json::from_str(&text).map_err(|e| {
            format!("Failed to parse response JSON: {e}")
        })?;

    let mut content = MessageContent::default();

    if let Some(choice) = completion.choices.first() {
        if let Some(ref text) = choice.message.content {
            content.text = text.clone();
        }

        for tc in &choice.message.tool_calls {
            let arguments: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_default();

            content.tool_calls.push(ToolCall {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                arguments,
                ..Default::default()
            });
        }
    }

    ::log::info!(
        "[A2UI] Parsed {} tool calls, text len: {}",
        content.tool_calls.len(),
        content.text.len()
    );

    Ok(content)
}

impl BotClient for A2uiClient {
    fn bots(
        &mut self,
    ) -> BoxPlatformSendFuture<'static, ClientResult<Vec<Bot>>> {
        self.client.bots()
    }

    fn clone_box(&self) -> Box<dyn BotClient> {
        Box::new(self.clone())
    }

    fn send(
        &mut self,
        bot_id: &BotId,
        messages: &[Message],
        tools: &[Tool],
    ) -> BoxPlatformSendStream<'static, ClientResult<MessageContent>> {
        let instance_enabled =
            self.a2ui_enabled.load(Ordering::SeqCst);
        let global_enabled = is_global_a2ui_enabled();
        let a2ui_enabled = instance_enabled || global_enabled;

        if !a2ui_enabled {
            // A2UI not enabled: forward to wrapped client as-is
            return self.client.send(bot_id, messages, tools);
        }

        ::log::info!(
            "[A2UI] Enabled - making non-streaming call ({} messages)",
            messages.len()
        );

        // Build combined tools
        let mut all_tools: Vec<Tool> = tools.to_vec();
        all_tools.extend(get_a2ui_tools());

        // Build messages with A2UI system prompt
        let mut all_messages = vec![Message {
            from: EntityId::System,
            content: MessageContent {
                text: A2UI_SYSTEM_PROMPT.to_string(),
                ..Default::default()
            },
            ..Default::default()
        }];
        all_messages.extend(messages.to_vec());

        // Convert to outgoing format (tool results expand to
        // multiple messages, one per tool_call_id)
        let outgoing_messages: Vec<OutgoingMessage> = all_messages
            .iter()
            .flat_map(to_outgoing_messages)
            .collect();
        let outgoing_tools: Vec<OutgoingTool> =
            all_tools.iter().map(to_outgoing_tool).collect();

        let model = bot_id.id().to_string();
        let api_url = self.api_url.clone();
        let api_key = self.api_key.clone();

        // Make the non-streaming call.
        Box::pin(async_stream::stream! {
            use crate::aitk::protocol::{ClientError, ClientErrorKind};

            match send_non_streaming(
                api_url,
                api_key,
                model,
                outgoing_messages,
                outgoing_tools,
            ).await {
                Ok(content) => {
                    yield ClientResult::new_ok(content);
                }
                Err(err) => {
                    ::log::error!("[A2UI] Non-streaming call failed: {}", err);
                    yield ClientError::new(
                        ClientErrorKind::Network,
                        err,
                    ).into();
                }
            }
        })
    }
}

/// Check if a tool call is an A2UI tool.
pub fn is_a2ui_tool_call(name: &str) -> bool {
    matches!(
        name,
        "create_text"
            | "create_button"
            | "create_textfield"
            | "create_checkbox"
            | "create_slider"
            | "create_card"
            | "create_column"
            | "create_row"
            | "set_data"
            | "render_ui"
    )
}
