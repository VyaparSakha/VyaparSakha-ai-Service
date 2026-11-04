"""
tests/test_cloudflare_ai.py
===========================
Comprehensive unit and integration test suite for Cloudflare Workers AI
(GLM: @cf/zai-org/glm-4.7-flash and Gemma: @cf/google/gemma-4-26b-a4b-it).

Covers:
1. GLM model selection.
2. Gemma model selection.
3. Default model selection.
4. Missing credentials.
5. Invalid model configuration.
6. Fallback from GLM to Gemma.
7. No fallback for authentication errors.
8. Preservation of LangGraph state.
9. Streaming response behavior.
10. 10k–20k token context handling.
11. Tool-call message preservation.
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from app.lib.cloudflare_ai import (
    MODEL_GLM,
    MODEL_GEMMA,
    CloudflareChatModelWrapper,
    compress_context_if_needed,
    estimate_message_tokens,
    getChatModel,
    get_chat_model,
    is_cloudflare_configured,
    is_retryable_error,
    resolve_model_name,
    validate_cloudflare_config,
    validate_messages,
    MAX_CONTEXT_TOKENS,
)


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure environment is controlled for each test."""
    old_env = os.environ.copy()
    os.environ["CLOUDFLARE_AI_ACCOUNT_ID"] = "test_account_123"
    os.environ["CLOUDFLARE_AI_API_TOKEN"] = "test_token_xyz"
    os.environ["CLOUDFLARE_ACCOUNT_ID"] = "test_account_123"
    os.environ["CLOUDFLARE_API_TOKEN"] = "test_token_xyz"
    os.environ["CLOUDFLARE_AI_MODEL"] = "glm"
    os.environ["AI_ENABLE_FALLBACK"] = "true"
    os.environ["AI_MAX_TOKENS"] = "2000"
    os.environ["AI_TEMPERATURE"] = "0.4"
    yield
    os.environ.clear()
    os.environ.update(old_env)


# ---------------------------------------------------------------------------
# Test 1: GLM Model Selection
# ---------------------------------------------------------------------------
def test_1_glm_model_selection():
    """Verify GLM model is selected via env and explicit argument."""
    os.environ["CLOUDFLARE_AI_MODEL"] = "glm"
    model = getChatModel()
    assert model.primary_model_id == MODEL_GLM
    assert model.fallback_model_id == MODEL_GEMMA
    assert model._config["base_url"] == "https://api.cloudflare.com/client/v4/accounts/test_account_123/ai/v1"
    assert model._config["api_token"] == "test_token_xyz"

    # Explicit argument
    model_explicit = get_chat_model(model="glm")
    assert model_explicit.primary_model_id == MODEL_GLM

    # Full model id
    model_full = getChatModel(model=MODEL_GLM)
    assert model_full.primary_model_id == MODEL_GLM


# ---------------------------------------------------------------------------
# Test 2: Gemma Model Selection
# ---------------------------------------------------------------------------
def test_2_gemma_model_selection():
    """Verify Gemma model is selected via env and explicit argument."""
    os.environ["CLOUDFLARE_AI_MODEL"] = "gemma"
    model = getChatModel()
    assert model.primary_model_id == MODEL_GEMMA
    assert model.fallback_model_id == MODEL_GLM

    # Explicit argument
    model_explicit = get_chat_model(model="gemma")
    assert model_explicit.primary_model_id == MODEL_GEMMA

    # Full model id
    model_full = getChatModel(model=MODEL_GEMMA)
    assert model_full.primary_model_id == MODEL_GEMMA


# ---------------------------------------------------------------------------
# Test 3: Default Model Selection
# ---------------------------------------------------------------------------
def test_3_default_model_selection():
    """Verify default model is GLM when CLOUDFLARE_AI_MODEL is not set."""
    if "CLOUDFLARE_AI_MODEL" in os.environ:
        del os.environ["CLOUDFLARE_AI_MODEL"]

    model = getChatModel()
    assert model.primary_model_id == MODEL_GLM
    assert model.fallback_model_id == MODEL_GEMMA


# ---------------------------------------------------------------------------
# Test 4: Missing Credentials Validation
# ---------------------------------------------------------------------------
def test_4_missing_credentials():
    """Verify clear ValueError when credentials are missing."""
    # Missing Account ID
    os.environ["CLOUDFLARE_AI_ACCOUNT_ID"] = ""
    os.environ["CLOUDFLARE_ACCOUNT_ID"] = ""
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID is required"):
        getChatModel(account_id="")

    os.environ["CLOUDFLARE_AI_ACCOUNT_ID"] = "test_account_123"
    os.environ["CLOUDFLARE_ACCOUNT_ID"] = "test_account_123"
    # Missing API Token
    os.environ["CLOUDFLARE_AI_API_TOKEN"] = ""
    os.environ["CLOUDFLARE_API_TOKEN"] = ""
    with pytest.raises(ValueError, match="CLOUDFLARE_API_TOKEN is required"):
        getChatModel(api_token="")


# ---------------------------------------------------------------------------
# Test 5: Invalid Model Configuration
# ---------------------------------------------------------------------------
def test_5_invalid_model_configuration():
    """Verify clear error when an unsupported model name is supplied."""
    with pytest.raises(ValueError, match="Invalid model 'llama3'. Valid options are: 'glm'"):
        getChatModel(model="llama3")

    with pytest.raises(ValueError, match="Invalid model 'gpt-4'"):
        resolve_model_name("gpt-4")


# ---------------------------------------------------------------------------
# Test 6: Fallback from GLM to Gemma
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_6_fallback_from_glm_to_gemma():
    """Verify that a transient error on GLM automatically falls back to Gemma."""
    model = getChatModel(model="glm", enable_fallback=True)

    mock_primary_chat = MagicMock()
    # Primary fails with transient 429
    class MockRateLimitError(Exception):
        status_code = 429

    mock_primary_chat.ainvoke = AsyncMock(side_effect=MockRateLimitError("429 Too Many Requests - Capacity overloaded"))

    mock_fallback_chat = MagicMock()
    mock_fallback_chat.ainvoke = AsyncMock(
        return_value=AIMessage(content="Response from Gemma fallback", response_metadata={})
    )

    model._primary_chat = mock_primary_chat
    model._fallback_chat = mock_fallback_chat

    messages = [HumanMessage(content="Explain store inventory turnover ratio.")]
    response = await model.ainvoke(messages)

    assert mock_primary_chat.ainvoke.call_count == 1
    assert mock_fallback_chat.ainvoke.call_count == 1
    assert response.content == "Response from Gemma fallback"
    assert response.response_metadata["model_name"] == MODEL_GEMMA
    assert response.response_metadata["fallback_used"] is True


# ---------------------------------------------------------------------------
# Test 7: No Fallback for Authentication Errors
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_7_no_fallback_for_auth_errors():
    """Verify that 401/403 or Invalid API Key errors do NOT trigger fallback."""
    model = getChatModel(model="glm", enable_fallback=True)

    class MockAuthError(Exception):
        status_code = 401

    mock_primary_chat = MagicMock()
    mock_primary_chat.ainvoke = AsyncMock(side_effect=MockAuthError("Unauthorized: Invalid API Token"))

    mock_fallback_chat = MagicMock()
    mock_fallback_chat.ainvoke = AsyncMock()

    model._primary_chat = mock_primary_chat
    model._fallback_chat = mock_fallback_chat

    messages = [HumanMessage(content="Hello")]

    with pytest.raises(MockAuthError, match="Unauthorized"):
        await model.ainvoke(messages)

    # Fallback must NEVER be called for authentication failures
    assert mock_fallback_chat.ainvoke.call_count == 0


# ---------------------------------------------------------------------------
# Test 8: Preservation of LangGraph State
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_8_preservation_of_langgraph_state():
    """Verify that planner_node and think_node preserve LangGraph state fields."""
    from app.agent.nodes.planner_node import planner_node
    from app.agent.nodes.think_node import think_node

    # 1. Test planner_node preserves state and generates JSON plan
    state = {
        "user_prompt": "Audit low stock inventory and recommend restock",
        "store_id": "store_test_99",
        "loop_count": 0,
        "is_owner": True,
        "permissions": ["inventory_view"],
        "messages": [],
    }

    mock_plan_response = AIMessage(
        content='{"goal": "Audit low stock and recommend restock", "steps": ["1. Check stock", "2. Restock"]}'
    )

    with patch("app.agent.nodes.planner_node.getChatModel") as mock_get_chat:
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_plan_response)
        mock_get_chat.return_value = mock_model

        result = await planner_node(state)
        assert "plan" in result
        assert result["plan"]["goal"] == "Audit low stock and recommend restock"
        assert len(result["plan"]["steps"]) == 2

    # 2. Test think_node preserves state fields
    think_state = {
        "user_prompt": "Show store overview",
        "store_id": "store_test_99",
        "user_id": "user_123",
        "loop_count": 0,
        "max_loops": 3,
        "is_owner": True,
        "permissions": ["stores_read"],
        "messages": [HumanMessage(content="Show store overview")],
        "goal": "Show store overview",
        "tool_results": [{"tool_name": "get_store_overview", "success": True, "data": {"name": "Kirana Store"}}],
        "inventory_context": {"low_stock": 2},
    }

    mock_think_response = AIMessage(
        content="Here is your store overview: Kirana Store is healthy with 2 low stock items.",
        tool_calls=[],
    )

    with patch("app.agent.nodes.think_node.getChatModel") as mock_get_chat:
        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        mock_model.ainvoke = AsyncMock(return_value=mock_think_response)
        mock_get_chat.return_value = mock_model

        result = await think_node(think_state)
        assert "messages" in result
        assert result["goal_status"] == "complete"
        assert "Kirana Store" in result["final_answer"]


# ---------------------------------------------------------------------------
# Test 9: Streaming Response Behavior
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_9_streaming_response_behavior():
    """Verify that astream yields chunks cleanly and supports fallback streaming."""
    model = getChatModel(model="glm", enable_fallback=True)

    async def mock_stream_chunks(*args, **kwargs):
        yield AIMessageChunk(content="Vyapar ")
        yield AIMessageChunk(content="Sakha ")
        yield AIMessageChunk(content="AI")

    mock_primary = MagicMock()
    mock_primary.astream = mock_stream_chunks
    model._primary_chat = mock_primary

    chunks = []
    async for chunk in model.astream([HumanMessage(content="Greeting")]):
        chunks.append(chunk.content)

    assert "".join(chunks) == "Vyapar Sakha AI"

    # Test fallback streaming when primary fails before yielding
    class MockServiceUnavailable(Exception):
        status_code = 503

    async def failing_stream(*args, **kwargs):
        raise MockServiceUnavailable("503 Service Unavailable")
        yield  # Make it an async generator

    async def fallback_stream(*args, **kwargs):
        yield AIMessageChunk(content="Gemma ")
        yield AIMessageChunk(content="Streamed")

    mock_failing_primary = MagicMock()
    mock_failing_primary.astream = failing_stream

    mock_fallback = MagicMock()
    mock_fallback.astream = fallback_stream

    model._primary_chat = mock_failing_primary
    model._fallback_chat = mock_fallback

    fb_chunks = []
    async for chunk in model.astream([HumanMessage(content="Hello")]):
        fb_chunks.append(chunk.content)

    assert "".join(fb_chunks) == "Gemma Streamed"


# ---------------------------------------------------------------------------
# Test 10: 10k–20k Token Context Handling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_10_context_token_handling_10k_20k():
    """Verify token estimation and sliding summarization for large contexts (10k-20k)."""
    # Build a message sequence of ~14,000 tokens (approx 50,000 characters)
    sys_msg = SystemMessage(content="System instruction: You are Vyapar Sakha assistant.")
    long_content = "This is a detailed record of sales transactions and inventory logs. " * 300
    
    messages = [sys_msg]
    for i in range(10):
        messages.append(HumanMessage(content=f"Turn {i}: {long_content}"))
        messages.append(AIMessage(content=f"Report {i}: Recorded {len(long_content)} bytes."))
    latest_user_prompt = HumanMessage(content="Now give me the total inventory count.")
    messages.append(latest_user_prompt)

    estimated_tokens = estimate_message_tokens(messages)
    assert estimated_tokens > 12000, f"Expected >12000 tokens, got {estimated_tokens}"

    # Verify compress_context_if_needed compresses older turns
    compressed = await compress_context_if_needed(messages, max_tokens=MAX_CONTEXT_TOKENS)
    new_tokens = estimate_message_tokens(compressed)

    assert new_tokens < estimated_tokens
    # System instruction must be preserved intact
    assert any(m.content == sys_msg.content for m in compressed if isinstance(m, SystemMessage))
    # Latest user message must be preserved intact
    assert compressed[-1].content == latest_user_prompt.content

    # If context fundamentally exceeds 20,000 tokens even with 1 message, raise ValueError
    giant_message = [HumanMessage(content="A" * 90000)]
    with pytest.raises(ValueError, match="exceeds maximum limit of 20000 tokens"):
        await compress_context_if_needed(giant_message, max_tokens=20000)


# ---------------------------------------------------------------------------
# Test 11: Tool-Call Message Preservation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_11_tool_call_message_preservation():
    """Verify that tool definitions, tool_calls, and ToolMessages are preserved."""
    @tool
    def get_stock_level(product_id: str) -> dict:
        """Fetch current stock for product."""
        return {"product_id": product_id, "stock": 42}

    model = getChatModel(model="glm")
    model_with_tools = model.bind_tools([get_stock_level])

    # 1. Primary returns a tool call
    expected_tool_call = {
        "name": "get_stock_level",
        "args": {"product_id": "SKU_RICE_01"},
        "id": "call_abc123",
    }
    mock_ai_tool_response = AIMessage(
        content="",
        tool_calls=[expected_tool_call],
        response_metadata={},
    )

    mock_chat = MagicMock()
    mock_chat.ainvoke = AsyncMock(return_value=mock_ai_tool_response)
    model_with_tools._primary_chat = mock_chat

    messages = [HumanMessage(content="Check stock for SKU_RICE_01")]
    response = await model_with_tools.ainvoke(messages)

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "get_stock_level"
    assert response.tool_calls[0]["args"] == {"product_id": "SKU_RICE_01"}
    assert response.tool_calls[0]["id"] == "call_abc123"

    # 2. Subsequent loop passes ToolMessage response back into conversation history
    tool_msg = ToolMessage(content='{"product_id": "SKU_RICE_01", "stock": 42}', tool_call_id="call_abc123")
    followup_messages = messages + [response, tool_msg]

    mock_final_answer = AIMessage(content="The current stock level for SKU_RICE_01 is 42 units.")
    mock_chat.ainvoke = AsyncMock(return_value=mock_final_answer)

    final_resp = await model_with_tools.ainvoke(followup_messages)
    assert "42 units" in final_resp.content

    # Verify validate_messages preserves tool messages
    validated = validate_messages(followup_messages)
    assert len(validated) == 3
    assert isinstance(validated[1], AIMessage)
    assert isinstance(validated[2], ToolMessage)


# ---------------------------------------------------------------------------
# Test 12: Distinct Account IDs for AI and R2
# ---------------------------------------------------------------------------
def test_12_distinct_account_ids_for_ai_and_r2():
    """Verify that CLOUDFLARE_AI_ACCOUNT_ID and CLOUDFLARE_R2_ACCOUNT_ID can be distinct."""
    os.environ["CLOUDFLARE_AI_ACCOUNT_ID"] = "distinct_ai_account_999"
    os.environ["CLOUDFLARE_AI_API_TOKEN"] = "distinct_ai_token_888"
    os.environ["CLOUDFLARE_R2_ACCOUNT_ID"] = "distinct_r2_account_777"
    os.environ["CLOUDFLARE_ACCOUNT_ID"] = "fallback_account_000"

    # AI model factory must use CLOUDFLARE_AI_ACCOUNT_ID
    config = validate_cloudflare_config()
    assert config["account_id"] == "distinct_ai_account_999"
    assert config["api_token"] == "distinct_ai_token_888"
    assert "distinct_ai_account_999" in config["base_url"]

    # Test Settings properties
    from app.config.settings import Settings
    s = Settings(
        cloudflare_ai_account_id="ai_acc_1",
        cloudflare_r2_account_id="r2_acc_2",
        cloudflare_account_id="shared_acc",
    )
    assert s.effective_cloudflare_ai_account_id == "ai_acc_1"
    assert s.effective_cloudflare_r2_account_id == "r2_acc_2"

