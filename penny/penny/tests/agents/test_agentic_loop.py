"""Tests for agentic loop changes: reasoning, last step, and after_step hook."""

import pytest

from penny.agents.base import Agent
from penny.config import Config
from penny.database import Database
from penny.ollama import OllamaClient
from penny.responses import PennyResponse
from penny.tools.search import SearchTool


def _make_agent(test_db, mock_ollama, *, max_steps=3):
    """Create a minimal Agent for loop testing."""
    db = Database(test_db)
    db.create_tables()
    config = Config(
        channel_type="signal",
        signal_number="+15551234567",
        signal_api_url="http://localhost:8080",
        discord_bot_token=None,
        discord_channel_id=None,
        ollama_api_url="http://localhost:11434",
        ollama_model="test-model",
        perplexity_api_key=None,
        log_level="DEBUG",
        db_path=test_db,
    )
    search_tool = SearchTool(perplexity_api_key="test-key", db=db)
    client = OllamaClient(
        api_url="http://localhost:11434",
        model="test-model",
        db=db,
        max_retries=1,
        retry_delay=0.1,
    )
    agent = Agent(
        system_prompt="test",
        model_client=client,
        tools=[search_tool],
        db=db,
        config=config,
        max_steps=max_steps,
    )
    return agent, db


class TestReasoningStripped:
    """Test that reasoning is popped from tool arguments and stored on the record."""

    @pytest.mark.asyncio
    async def test_reasoning_captured_on_tool_call_record(self, test_db, mock_ollama):
        """Reasoning from tool call args is stored on ToolCallRecord."""
        agent, db = _make_agent(test_db, mock_ollama)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(
                    request,
                    "search",
                    {"query": "weather", "reasoning": "User asked about weather"},
                )
            return mock_ollama._make_text_response(request, "here's the weather!")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("what's the weather?")
        assert response.answer == "here's the weather!"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].reasoning == "User asked about weather"
        # reasoning should NOT be in the arguments dict
        assert "reasoning" not in response.tool_calls[0].arguments

        await agent.close()

    @pytest.mark.asyncio
    async def test_reasoning_none_when_not_provided(self, test_db, mock_ollama):
        """ToolCallRecord.reasoning is None when model doesn't provide it."""
        agent, db = _make_agent(test_db, mock_ollama)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "weather"})
            return mock_ollama._make_text_response(request, "done")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].reasoning is None

        await agent.close()


class TestLastStepToolRemoval:
    """Test that on the final step, tools are removed so the model must produce text."""

    @pytest.mark.asyncio
    async def test_final_step_has_no_tools(self, test_db, mock_ollama):
        """On the last step, the model is called without tools."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=2)

        def handler(request, count):
            if count == 1:
                # Step 1: model makes a tool call
                return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
            # Step 2 (final): model must produce text — verify no tools sent
            return mock_ollama._make_text_response(request, "final answer")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test")
        assert response.answer == "final answer"

        # Step 1 should have tools, step 2 should not
        assert mock_ollama.requests[0]["tools"] is not None
        assert len(mock_ollama.requests[0]["tools"]) > 0
        assert mock_ollama.requests[1]["tools"] is None

        await agent.close()

    @pytest.mark.asyncio
    async def test_hallucinated_tool_calls_ignored_on_final_step(self, test_db, mock_ollama):
        """If model hallucinates tool calls on the final step, they are ignored."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=2)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
            # Final step: model hallucinates a tool call despite no tools offered
            return mock_ollama._make_tool_call_response(request, "search", {"query": "more"})

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test")
        # Should NOT get AGENT_MAX_STEPS — hallucinated call is ignored.
        # Preceding tool calls → FALLBACK_RESPONSE (friendlier than AGENT_EMPTY_RESPONSE).
        assert "couldn't complete" not in response.answer.lower()
        assert response.answer == PennyResponse.FALLBACK_RESPONSE

        await agent.close()

    @pytest.mark.asyncio
    async def test_hallucinated_tool_call_with_text_uses_text(self, test_db, mock_ollama):
        """If model returns both text and tool calls on final step, text is used."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=2)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
            # Final step: model returns text AND a hallucinated tool call
            resp = mock_ollama._make_tool_call_response(request, "search", {"query": "more"})
            resp["message"]["content"] = "here is the answer"
            return resp

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test")
        assert response.answer == "here is the answer"

        await agent.close()


class TestRepeatCallGuard:
    """Test that repeat tool calls are blocked by args, not just name."""

    @pytest.mark.asyncio
    async def test_same_tool_different_args_allowed(self, test_db, mock_ollama):
        """Calling the same tool with different arguments is allowed."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=4)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(
                    request, "search", {"query": "first topic"}
                )
            if count == 2:
                return mock_ollama._make_tool_call_response(
                    request, "search", {"query": "second topic"}
                )
            return mock_ollama._make_text_response(request, "done")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test")
        assert response.answer == "done"
        # Both searches should have executed
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].arguments["query"] == "first topic"
        assert response.tool_calls[1].arguments["query"] == "second topic"

        await agent.close()

    @pytest.mark.asyncio
    async def test_same_tool_same_args_blocked(self, test_db, mock_ollama):
        """Calling the same tool with identical arguments is blocked."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=3)

        def handler(request, count):
            if count <= 2:
                return mock_ollama._make_tool_call_response(
                    request, "search", {"query": "same query"}
                )
            return mock_ollama._make_text_response(request, "done")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test")
        assert response.answer == "done"
        # Only first call should have executed
        assert len(response.tool_calls) == 1

        await agent.close()


class TestEmptyContentFallback:
    """Test that an empty model response falls back to AGENT_EMPTY_RESPONSE."""

    @pytest.mark.asyncio
    async def test_empty_response_returns_agent_empty_response(self, test_db, mock_ollama):
        """When the model returns empty content, AGENT_EMPTY_RESPONSE is returned."""
        agent, db = _make_agent(test_db, mock_ollama)

        def handler(request, count):
            return mock_ollama._make_text_response(request, "")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test prompt")
        assert response.answer == PennyResponse.AGENT_EMPTY_RESPONSE

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_response_after_tool_call(self, test_db, mock_ollama):
        """FALLBACK_RESPONSE is returned when model returns empty after preceding tool calls."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
            return mock_ollama._make_text_response(request, "")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test prompt")
        assert response.answer == PennyResponse.FALLBACK_RESPONSE

        await agent.close()


class TestThinkTagStripping:
    """Test that <think>...</think> blocks are stripped from final responses."""

    @pytest.mark.asyncio
    async def test_think_tags_stripped_from_content(self, test_db, mock_ollama):
        """<think>...</think> blocks in content are removed before sending to user."""
        agent, db = _make_agent(test_db, mock_ollama)

        raw = "<think>Internal reasoning here.</think>\nHere is the real answer."
        mock_ollama.set_response_handler(
            lambda req, count: mock_ollama._make_text_response(req, raw)
        )

        response = await agent.run("test")
        assert "<think>" not in response.answer
        assert "Internal reasoning here." not in response.answer
        assert response.answer == "Here is the real answer."

        await agent.close()

    @pytest.mark.asyncio
    async def test_think_tags_moved_to_thinking_field(self, test_db, mock_ollama):
        """Content inside <think> blocks is captured in the thinking field."""
        agent, db = _make_agent(test_db, mock_ollama)

        raw = "<think>Step-by-step plan.</think>\nFinal response."
        mock_ollama.set_response_handler(
            lambda req, count: mock_ollama._make_text_response(req, raw)
        )

        response = await agent.run("test")
        assert response.thinking == "Step-by-step plan."
        assert response.answer == "Final response."

        await agent.close()

    @pytest.mark.asyncio
    async def test_response_without_think_tags_unchanged(self, test_db, mock_ollama):
        """Responses that contain no <think> tags are returned as-is."""
        agent, db = _make_agent(test_db, mock_ollama)

        mock_ollama.set_response_handler(
            lambda req, count: mock_ollama._make_text_response(req, "Normal answer.")
        )

        response = await agent.run("test")
        assert response.answer == "Normal answer."
        assert response.thinking is None

        await agent.close()


class TestAfterStepHook:
    """Test the after_step hook fires after tool calls."""

    @pytest.mark.asyncio
    async def testafter_step_called_with_step_records(self, test_db, mock_ollama):
        """after_step receives only the records from the current step."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=3)

        captured_step_records = []

        async def captureafter_step(step_records, messages):
            captured_step_records.append(list(step_records))

        agent.after_step = captureafter_step

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(
                    request, "search", {"query": "first", "reasoning": "step 1 reason"}
                )
            if count == 2:
                return mock_ollama._make_tool_call_response(
                    request, "search", {"query": "second", "reasoning": "step 2 reason"}
                )
            return mock_ollama._make_text_response(request, "done")

        mock_ollama.set_response_handler(handler)
        agent.allow_repeat_tools = True

        response = await agent.run("test")
        assert response.answer == "done"

        # Two steps with tool calls → two after_step calls
        assert len(captured_step_records) == 2
        assert len(captured_step_records[0]) == 1
        assert captured_step_records[0][0].reasoning == "step 1 reason"
        assert len(captured_step_records[1]) == 1
        assert captured_step_records[1][0].reasoning == "step 2 reason"

        await agent.close()


class TestEmptyContentRetry:
    """Test that empty content responses trigger a retry with a follow-up prompt."""

    @pytest.mark.asyncio
    async def test_empty_content_on_nonfinal_step_retries_with_followup(self, test_db, mock_ollama):
        """When model returns empty content on a non-final step, agent retries with follow-up."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
            if count == 2:
                # Thinking-only response: empty content, no tool calls
                return mock_ollama._make_text_response(request, "")
            # After follow-up injection, model returns actual text
            return mock_ollama._make_text_response(request, "here's the answer")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test question")
        assert response.answer == "here's the answer"
        # Three model calls: tool call, empty response, final answer
        assert len(mock_ollama.requests) == 3

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_content_on_final_step_retries_and_succeeds(self, test_db, mock_ollama):
        """When model returns empty content on the final step, agent retries once and succeeds."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=1)

        def handler(request, count):
            if count == 1:
                # Final step returns empty content
                return mock_ollama._make_text_response(request, "")
            # Retry (extra step) returns real content
            return mock_ollama._make_text_response(request, "here's the answer")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test question")
        assert response.answer == "here's the answer"
        # Two model calls: empty final step + retry
        assert len(mock_ollama.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_content_twice_returns_fallback(self, test_db, mock_ollama):
        """When model returns empty content on both the final step and retry, returns fallback."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=1)

        def handler(request, count):
            return mock_ollama._make_text_response(request, "")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test question")
        assert response.answer == PennyResponse.AGENT_EMPTY_RESPONSE
        # Two model calls: empty final step + one retry that also returns empty
        assert len(mock_ollama.requests) == 2

        await agent.close()


class TestEmptyContentAfterToolCalls:
    """Tests for combined empty-content fixes: synthesis prompt, think tag stripping,
    retry counter reset, context truncation, and fallback response."""

    @pytest.mark.asyncio
    async def test_synthesis_prompt_after_tool_calls(self, test_db, mock_ollama):
        """When model returns empty after tool calls, retry uses synthesis prompt."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=2)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
            if count == 2:
                return mock_ollama._make_text_response(request, "")
            return mock_ollama._make_text_response(request, "Here's what I found!")

        mock_ollama.set_response_handler(handler)
        response = await agent.run("test question")
        assert response.answer == "Here's what I found!"

        retry_messages = mock_ollama.requests[2]["messages"]
        last_user = next(m for m in reversed(retry_messages) if m["role"] == "user")
        assert "synthesize" in last_user["content"].lower()

        await agent.close()

    @pytest.mark.asyncio
    async def test_generic_prompt_without_tool_calls(self, test_db, mock_ollama):
        """Without tool calls, empty-content retry uses generic prompt."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=1)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_text_response(request, "")
            return mock_ollama._make_text_response(request, "here's my answer")

        mock_ollama.set_response_handler(handler)
        response = await agent.run("test question")
        assert response.answer == "here's my answer"

        retry_messages = mock_ollama.requests[1]["messages"]
        last_user = next(m for m in reversed(retry_messages) if m["role"] == "user")
        assert last_user["content"] == "Please provide your response."

        await agent.close()

    @pytest.mark.asyncio
    async def test_think_only_response_triggers_retry(self, test_db, mock_ollama):
        """Model returning only <think> tags with no body triggers retry."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
            if count == 2:
                return mock_ollama._make_text_response(
                    request, "<think>Let me reason about this...</think>"
                )
            return mock_ollama._make_text_response(request, "here's the answer")

        mock_ollama.set_response_handler(handler)
        response = await agent.run("test question")
        assert response.answer == "here's the answer"
        assert len(mock_ollama.requests) == 3

        await agent.close()

    @pytest.mark.asyncio
    async def test_retry_counter_resets_after_tool_calls(self, test_db, mock_ollama):
        """empty_retries resets after tool calls so synthesis step gets a retry."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=5)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "first"})
            if count == 2:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "second"})
            if count == 3:
                return mock_ollama._make_text_response(request, "")
            if count == 4:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "third"})
            if count == 5:
                return mock_ollama._make_text_response(request, "")
            return mock_ollama._make_text_response(request, "synthesized answer")

        mock_ollama.set_response_handler(handler)
        agent.allow_repeat_tools = True
        response = await agent.run("test question")
        assert response.answer == "synthesized answer"

        await agent.close()

    @pytest.mark.asyncio
    async def test_fallback_response_after_tool_calls(self, test_db, mock_ollama):
        """FALLBACK_RESPONSE (not AGENT_EMPTY_RESPONSE) when empty after tool calls."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
            return mock_ollama._make_text_response(request, "")

        mock_ollama.set_response_handler(handler)
        response = await agent.run("test prompt")
        assert response.answer == PennyResponse.FALLBACK_RESPONSE

        await agent.close()

    @pytest.mark.asyncio
    async def test_tool_result_truncated_at_source(self, test_db, mock_ollama):
        """Tool results exceeding MAX_TOOL_RESULT_CHARS are truncated."""
        from unittest.mock import patch

        from penny.tools.models import ToolResult

        agent, db = _make_agent(test_db, mock_ollama)
        large_result = "x" * (Agent.MAX_TOOL_RESULT_CHARS + 500)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
            messages = request["messages"]
            tool_messages = [m for m in messages if m.get("role") == "tool"]
            assert len(tool_messages) == 1
            content = tool_messages[0]["content"]
            assert len(content) <= Agent.MAX_TOOL_RESULT_CHARS + len(" [truncated]")
            assert content.endswith(" [truncated]")
            return mock_ollama._make_text_response(request, "done")

        mock_ollama.set_response_handler(handler)

        with patch.object(agent._tool_executor, "execute") as mock_exec:
            mock_exec.return_value = ToolResult(tool="search", result=large_result)
            response = await agent.run("test")

        assert response.answer == "done"
        await agent.close()


class TestRefusalRetry:
    """Test that model refusals trigger a retry nudge."""

    @pytest.mark.asyncio
    async def test_refusal_on_nonfinal_step_retries_with_nudge(self, test_db, mock_ollama):
        """When model refuses on a non-final step, agent injects nudge and continues."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=3)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
            if count == 2:
                return mock_ollama._make_text_response(
                    request, "I'm sorry, but I can't help with that."
                )
            return mock_ollama._make_text_response(request, "Here are the vegan smoothie recipes!")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("Give me a list of vegan smoothie recipes")
        assert response.answer == "Here are the vegan smoothie recipes!"
        assert len(mock_ollama.requests) == 3

        await agent.close()

    @pytest.mark.asyncio
    async def test_refusal_on_final_step_retries_inline(self, test_db, mock_ollama):
        """When model refuses on the final step, agent retries once inline."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=1)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_text_response(request, "I cannot help with that request.")
            return mock_ollama._make_text_response(request, "Here is a helpful answer!")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test question")
        assert response.answer == "Here is a helpful answer!"
        assert len(mock_ollama.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_refusal_only_retried_once(self, test_db, mock_ollama):
        """Refusal retry only fires once — second refusal is returned as-is."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=3)

        def handler(request, count):
            return mock_ollama._make_text_response(request, "I'm sorry, I am unable to help.")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test question")
        # Should contain the refusal text (returned as-is after one retry)
        assert "sorry" in response.answer.lower() or "unable" in response.answer.lower()
        # Only two model calls: initial refusal + one retry
        assert len(mock_ollama.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_normal_response_not_retried(self, test_db, mock_ollama):
        """Normal responses are not mistakenly flagged as refusals."""
        agent, db = _make_agent(test_db, mock_ollama)

        mock_ollama.set_response_handler(
            lambda req, count: mock_ollama._make_text_response(req, "Here are your recipes!")
        )

        response = await agent.run("Give me vegan smoothie recipes")
        assert response.answer == "Here are your recipes!"
        assert len(mock_ollama.requests) == 1

        await agent.close()
