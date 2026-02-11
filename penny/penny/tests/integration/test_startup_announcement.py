"""Integration tests for startup announcement feature."""

import pytest

from penny.tests.conftest import TEST_SENDER, wait_until


@pytest.mark.asyncio
async def test_startup_announcement_with_commit(
    signal_server, test_config, mock_ollama, running_penny, monkeypatch
):
    """Test that Penny sends startup announcement with restart message from git commit."""
    # First run: populate database with a sender
    mock_ollama.set_default_flow(search_query="test", final_response="test response 🌟")

    async with running_penny(test_config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="hey penny")
        await signal_server.wait_for_message(timeout=10.0)

        # Verify sender is in database
        senders = penny.db.get_all_senders()
        assert TEST_SENDER in senders

        # Create user profile so they get startup announcements
        penny.db.save_user_info(
            sender=TEST_SENDER,
            name="Test User",
            location="Seattle, WA",
            timezone="America/Los_Angeles",
            date_of_birth="1990-01-01",
        )

    # Clear messages from first run
    signal_server.outgoing_messages.clear()

    # Set commit message in environment variable
    monkeypatch.setenv("GIT_COMMIT_MESSAGE", "feat: add cool new feature")

    # Second run: configure restart message and verify announcement
    mock_ollama.set_default_flow(final_response="i added a cool new feature! check it out")

    async with running_penny(test_config):
        # Announcement is sent before WebSocket listener starts
        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)

        # Should have received startup announcement
        assert len(signal_server.outgoing_messages) >= 1, (
            f"Expected at least 1 message, got {len(signal_server.outgoing_messages)}"
        )

        # Find the announcement message (should be the first or only message)
        announcement = signal_server.outgoing_messages[0]

        # Should be sent to TEST_SENDER
        assert TEST_SENDER in announcement["recipients"], (
            f"Expected {TEST_SENDER} in recipients, got {announcement['recipients']}"
        )

        # Should start with wave emoji and include restart message
        message = announcement["message"]
        assert message.startswith("👋"), f"Expected message to start with 👋, got: {message}"
        assert "i added a cool new feature! check it out" in message, (
            f"Expected restart message in announcement, got: {message}"
        )


@pytest.mark.asyncio
async def test_startup_announcement_fallback_no_git(
    signal_server, test_config, mock_ollama, running_penny, monkeypatch
):
    """Test that Penny falls back to 'I just restarted!' when git commit message unavailable."""
    # First run: populate database
    mock_ollama.set_default_flow(search_query="test", final_response="test response 🌟")

    async with running_penny(test_config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="hey penny")
        await signal_server.wait_for_message(timeout=10.0)

        senders = penny.db.get_all_senders()
        assert TEST_SENDER in senders

        # Create user profile so they get startup announcements
        penny.db.save_user_info(
            sender=TEST_SENDER,
            name="Test User",
            location="Seattle, WA",
            timezone="America/Los_Angeles",
            date_of_birth="1990-01-01",
        )

    # Clear messages
    signal_server.outgoing_messages.clear()

    # Set commit message to "unknown" to simulate missing git info
    monkeypatch.setenv("GIT_COMMIT_MESSAGE", "unknown")

    # Second run: verify fallback message
    async with running_penny(test_config):
        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)

        # Should use fallback message
        assert len(signal_server.outgoing_messages) >= 1, (
            f"Expected at least 1 message, got {len(signal_server.outgoing_messages)}"
        )
        announcement = signal_server.outgoing_messages[0]
        message = announcement["message"]
        assert message == "👋 I just restarted!", f"Expected fallback message, got: {message}"


@pytest.mark.asyncio
async def test_startup_announcement_fallback_llm_error(
    signal_server, test_config, mock_ollama, running_penny, monkeypatch
):
    """Test that Penny falls back when LLM transformation fails."""
    # First run: populate database
    mock_ollama.set_default_flow(search_query="test", final_response="test response 🌟")

    async with running_penny(test_config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="hey penny")
        await signal_server.wait_for_message(timeout=10.0)

        senders = penny.db.get_all_senders()
        assert TEST_SENDER in senders

        # Create user profile so they get startup announcements
        penny.db.save_user_info(
            sender=TEST_SENDER,
            name="Test User",
            location="Seattle, WA",
            timezone="America/Los_Angeles",
            date_of_birth="1990-01-01",
        )

    # Clear messages
    signal_server.outgoing_messages.clear()

    # Set commit message in environment variable
    monkeypatch.setenv("GIT_COMMIT_MESSAGE", "feat: some feature")

    # Second run: configure LLM to fail for restart message generation
    def error_handler(request, count):
        raise RuntimeError("Ollama is down")

    mock_ollama.set_response_handler(error_handler)

    async with running_penny(test_config):
        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)

        # Should use fallback message when LLM fails
        assert len(signal_server.outgoing_messages) >= 1, (
            f"Expected at least 1 message, got {len(signal_server.outgoing_messages)}"
        )
        announcement = signal_server.outgoing_messages[0]
        message = announcement["message"]
        assert message == "👋 I just restarted!", f"Expected fallback message, got: {message}"


@pytest.mark.asyncio
async def test_startup_announcement_no_recipients(
    signal_server, test_config, mock_ollama, running_penny
):
    """Test that Penny doesn't crash when there are no recipients."""
    # Start Penny without any prior message history.
    # Announcement runs before WebSocket listener starts, so by the time
    # running_penny yields (WebSocket connected), the code has already
    # completed. No sleep needed — check immediately.
    async with running_penny(test_config):
        assert len(signal_server.outgoing_messages) == 0


@pytest.mark.asyncio
async def test_startup_announcement_multiple_recipients(
    signal_server, test_config, mock_ollama, running_penny, monkeypatch
):
    """Test that Penny sends startup announcement to all known recipients."""
    # First run: populate database with multiple senders
    mock_ollama.set_default_flow(search_query="test", final_response="test response 🌟")

    sender1 = TEST_SENDER
    sender2 = "+15559998888"

    async with running_penny(test_config) as penny:
        # Send from first sender
        await signal_server.push_message(sender=sender1, content="hey penny")
        await signal_server.wait_for_message(timeout=10.0)

        # Send from second sender
        await signal_server.push_message(sender=sender2, content="hello")
        await signal_server.wait_for_message(timeout=10.0)

        # Verify both senders are in database
        senders = penny.db.get_all_senders()
        assert sender1 in senders
        assert sender2 in senders

        # Create user profiles for both senders so they get startup announcements
        penny.db.save_user_info(
            sender=sender1,
            name="Test User 1",
            location="Seattle, WA",
            timezone="America/Los_Angeles",
            date_of_birth="1990-01-01",
        )
        penny.db.save_user_info(
            sender=sender2,
            name="Test User 2",
            location="New York, NY",
            timezone="America/New_York",
            date_of_birth="1985-05-15",
        )

    # Clear messages from first run
    signal_server.outgoing_messages.clear()

    # Set commit message in environment variable
    monkeypatch.setenv("GIT_COMMIT_MESSAGE", "feat: update something cool")

    # Second run: verify announcements to both recipients
    mock_ollama.set_default_flow(final_response="i updated something cool")

    async with running_penny(test_config):
        expected_recipients = {sender1, sender2}

        def all_recipients_notified():
            all_recipients: set[str] = set()
            for msg in signal_server.outgoing_messages:
                all_recipients.update(msg.get("recipients", []))
            return expected_recipients.issubset(all_recipients)

        await wait_until(all_recipients_notified)

        # Should have sent to both recipients
        # Note: Signal API may send separate messages or batch them
        assert len(signal_server.outgoing_messages) >= 1, (
            f"Expected at least 1 message, got {len(signal_server.outgoing_messages)}"
        )

        # Collect all recipients
        all_recipients: set[str] = set()
        for msg in signal_server.outgoing_messages:
            all_recipients.update(msg.get("recipients", []))

        assert sender1 in all_recipients, f"Expected {sender1} in recipients, got {all_recipients}"
        assert sender2 in all_recipients, f"Expected {sender2} in recipients, got {all_recipients}"

        # All messages should contain the restart message
        for msg in signal_server.outgoing_messages:
            message = msg["message"]
            assert message.startswith("👋"), f"Expected message to start with 👋, got: {message}"
            assert "i updated something cool" in message, (
                f"Expected restart message in announcement, got: {message}"
            )
