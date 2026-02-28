"""User-facing response strings for Penny.

All textual responses that Penny sends to users are defined here.
Parameterized strings use .format() style templates.
"""


class PennyResponse:
    """All user-facing response strings, organized by feature area."""

    # ── General ──────────────────────────────────────────────────────────────

    FALLBACK_RESPONSE = "Sorry, I couldn't generate a response."
    RESTART_FALLBACK = "I just restarted!"

    # ── Agent Errors ─────────────────────────────────────────────────────────

    AGENT_MODEL_ERROR = "Sorry, I encountered an error communicating with the model."
    AGENT_EMPTY_RESPONSE = "Sorry, the model generated an empty response."
    AGENT_MAX_STEPS = "Sorry, I couldn't complete that request within the allowed steps."

    # ── Channel ──────────────────────────────────────────────────────────────

    THREADING_NOT_SUPPORTED_COMMANDS = "Commands can't be used in threads."
    THREADING_NOT_SUPPORTED_TEST = "Test mode can't be used in threads."
    UNKNOWN_COMMAND = "Unknown command: /{command_name}. Use /commands to see available commands."
    COMMAND_ERROR = "Failed to run command: {error}"

    # ── Vision ───────────────────────────────────────────────────────────────

    VISION_NOT_CONFIGURED_MESSAGE = (
        "I can see you sent an image but I don't have vision configured right now."
    )
    VISION_IMAGE_CONTEXT = "User said '{user_text}' and included an image of: {caption}"
    VISION_IMAGE_ONLY_CONTEXT = "User sent an image of: {caption}"

    # ── Learn ────────────────────────────────────────────────────────────────

    LEARN_ACKNOWLEDGED = "Okay, I'll learn more about {topic}"
    LEARN_EMPTY = (
        "Nothing being actively researched right now. "
        "Use `/learn <topic>` to start learning about something."
    )
    LEARN_STATUS_HEADER = "**Learning Status**"
    LEARN_COMPLETE_HEADER = "I finished learning about **{topic}**"
    LEARN_COMPLETE_ENTITY_LINE = "• **{name}** ({fact_count} facts, interest: {score})"
    LEARN_COMPLETE_NO_ENTITIES = "I didn't find any specific topics to track from that."

    # ── Memory ───────────────────────────────────────────────────────────────

    MEMORY_EMPTY = "You don't have any stored memories yet."
    MEMORY_LIST_HEADER = "**Your Memory**"
    MEMORY_ENTITY_NOT_FOUND = "#{number} doesn't match any memory. Use /memory to see the list."
    MEMORY_NO_FACTS = "I know about {name}, but I don't have any specific facts stored yet."
    MEMORY_DELETED = "Deleted '{name}' and {count} fact(s)."
    MEMORY_DELETE_USAGE = "Use `/memory {number} delete` to delete a memory."

    # ── Config ───────────────────────────────────────────────────────────────

    CONFIG_HEADER = "**Runtime Configuration**"
    CONFIG_GROUP_HEADER = "**{group}**"
    CONFIG_FOOTER = "Use `/config <key> <value>` to change a setting."
    CONFIG_UNKNOWN_PARAM = (
        "Unknown config parameter: {key}\nUse /config to see all available parameters."
    )
    CONFIG_PARAM_DISPLAY = "• **{key}**: {value} ({description})"
    CONFIG_INVALID_VALUE = "Invalid value for {key}: {error}"
    CONFIG_UPDATED = "Ok, updated {key} to {value}"

    # ── Profile ──────────────────────────────────────────────────────────────

    PROFILE_NO_PROFILE = (
        "You don't have a profile yet! Set it up with:\n"
        "`/profile <name> <location> <date of birth>`\n\n"
        "For example: `/profile sam denver march 5 1990` \U0001f4dd"
    )
    PROFILE_HEADER = "**Your Profile**"
    PROFILE_NAME = "**Name**: {name}"
    PROFILE_LOCATION = "**Location**: {location}"
    PROFILE_TIMEZONE = "**Timezone**: {timezone}"
    PROFILE_DOB = "**Date of Birth**: {dob}"

    PROFILE_CREATE_PARSE_ERROR = (
        "I couldn't understand that. Please provide your name, location, "
        "and date of birth.\n\n"
        "Example: `/profile sam denver march 5 1990`"
    )
    PROFILE_DATE_PARSE_ERROR = (
        "I couldn't parse '{date}' as a date. Try something like 'january 10 1995' \U0001f4c5"
    )
    PROFILE_TIMEZONE_ERROR = (
        "I couldn't find a timezone for '{location}'. Can you be more specific? \U0001f5fa\ufe0f"
    )
    PROFILE_CREATED = "Got it! Your profile is set up. Welcome, {name}! \U0001f389"

    PROFILE_UPDATE_PARSE_ERROR = (
        "I couldn't understand that. Please provide name and/or location.\n\n"
        "Example: `/profile sam denver`"
    )
    PROFILE_UPDATE_NAME = "name to **{name}**"
    PROFILE_UPDATE_LOCATION = "location to **{location}** ({timezone})"
    PROFILE_UPDATED = "Ok, I updated your {changes}! \u2705"
    PROFILE_UNCHANGED = "Your profile is unchanged \U0001f937"

    # ── Schedule ─────────────────────────────────────────────────────────────

    SCHEDULE_NO_TASKS = "You don't have any scheduled tasks yet \U0001f4c5"
    SCHEDULE_NEED_TIMEZONE = (
        "I need to know your timezone first. Send me your location or tell me your city \U0001f4cd"
    )
    SCHEDULE_PARSE_ERROR = (
        "Sorry, I couldn't understand that schedule format. "
        "Try something like: /schedule daily 9am what's the news?"
    )
    SCHEDULE_INVALID_CRON = (
        "Sorry, I couldn't figure out the timing. "
        "Try something like: /schedule daily 9am what's the news?"
    )
    SCHEDULE_DELETED_NO_REMAINING = "No more scheduled tasks."
    SCHEDULE_STILL_SCHEDULED = "**Still scheduled:**"
    SCHEDULE_INVALID_NUMBER = "Invalid schedule number: {number}"
    SCHEDULE_NO_SCHEDULE_WITH_NUMBER = "No schedule with number {number}"
    SCHEDULE_DELETED_PREFIX = "Deleted '{timing} {prompt}' \u2705"
    SCHEDULE_ADDED = "Added {timing}: {prompt} \u2705"

    # ── Email ────────────────────────────────────────────────────────────────

    EMAIL_NO_QUERY_TEXT = "Please ask a question about your email. Usage: /email <question>"
    EMAIL_ERROR = "Failed to search email: {error}"

    # ── Test ─────────────────────────────────────────────────────────────────

    TEST_MODE_PREFIX = "[TEST] "
    TEST_USAGE = "Please provide a prompt. Usage: /test <prompt>"
    TEST_NESTED_ERROR = "Nested commands are not supported in test mode."
    TEST_ERROR = "Failed to run test: {error}"
    TEST_NO_RESPONSE = "No response generated."

    # ── Draw ─────────────────────────────────────────────────────────────────

    DRAW_USAGE = "Please describe what you want to draw. Usage: /draw <prompt>"
    DRAW_ERROR = "Failed to generate image: {error}"

    # ── Bug ──────────────────────────────────────────────────────────────────

    BUG_USAGE = "Please provide a bug description. Usage: /bug <description>"
    BUG_FILED = "Bug filed! {issue_url}"
    BUG_ERROR = "Failed to create issue: {error}"

    # ── Feature ──────────────────────────────────────────────────────────────

    FEATURE_USAGE = "Please provide a feature description. Usage: /feature <description>"
    FEATURE_FILED = "Feature request filed! {issue_url}"
    FEATURE_ERROR = "Failed to create issue: {error}"

    # ── Debug ────────────────────────────────────────────────────────────────

    DEBUG_TEMPLATE = """**Debug Information**

**Git Commit**: {commit}
**Uptime**: {uptime}
**Channel**: {channel}
**Database**: {messages:,} messages, {threads} active threads
**Models**: {fg_model} (foreground), {bg_model} (background)
**Memory**: {memory}

**Background Tasks**:
{task_status}
"""
    DEBUG_UPTIME = "{days} days, {hours} hours, {minutes} minutes"
    DEBUG_NO_SCHEDULER = "Unknown (no scheduler)"
    DEBUG_TASK_NEVER = "• **{name}**: never run"
    DEBUG_TASK_SECONDS = "• **{name}**: {seconds}s ago"
    DEBUG_TASK_MINUTES = "• **{name}**: {minutes}m ago"
    DEBUG_TASK_HOURS = "• **{name}**: {hours}h ago"

    # ── Commands Index ───────────────────────────────────────────────────────

    COMMANDS_HEADER = "**Available Commands**"
    COMMANDS_UNKNOWN = "Unknown command: /{name}. Use /commands to see available commands."
    COMMANDS_HELP_HEADER = "**Command: /{name}**"

    # ── Unlearn ────────────────────────────────────────────────────────────────

    UNLEARN_EMPTY = "No learn history yet. Use `/learn <topic>` to start learning about something."
    UNLEARN_LIST_HEADER = "**Learn History**"
    UNLEARN_INVALID_NUMBER = "#{number} doesn't match any topic. Use /unlearn to see the list."
    UNLEARN_HEADER = "Forgetting what I learned about **{topic}**"
    UNLEARN_ENTITY_LINE = "• {name} ({fact_count} facts)"
    UNLEARN_NO_ENTITIES = "No entities were discovered from this topic."

    # ── Mute ──────────────────────────────────────────────────────────────────

    MUTE_ENABLED = "Notifications muted. Use /unmute when you want them back."
    MUTE_ALREADY = "Notifications are already muted."
    UNMUTE_ENABLED = "Notifications unmuted."
    UNMUTE_ALREADY = "Notifications aren't muted."

    # ── Follow ────────────────────────────────────────────────────────────────

    NEWS_NOT_CONFIGURED = (
        "Event tracking requires a NewsAPI.org key. Set NEWS_API_KEY in your .env and restart."
    )

    FOLLOW_ACKNOWLEDGED = "Got it, I'll keep track of **{topic}** for you ({timing} updates)."
    FOLLOW_NEED_TIMEZONE = (
        "I need to know your timezone first. Send me your location or tell me your city \U0001f4cd"
    )
    FOLLOW_PARSE_ERROR = (
        "Sorry, I couldn't understand that. Try something like: /follow daily 9:30am usa news"
    )
    FOLLOW_EMPTY = (
        "You're not following anything yet. Use `/follow <topic>` to start monitoring something."
    )
    FOLLOW_LIST_HEADER = "**Following**"
    FOLLOW_CANCELLED = "Stopped following **{topic}**."
    FOLLOW_NOT_FOUND = "#{number} doesn't match any follow. Use /follow to see the list."
    FOLLOW_QUERY_TERMS_ERROR = "Sorry, I couldn't generate search terms for that topic."

    # ── Events ─────────────────────────────────────────────────────────────────

    EVENTS_EMPTY = "No recent events. Use `/follow <topic>` to start tracking."
    EVENTS_LIST_HEADER = "**Recent Events**"
    EVENTS_NOT_FOUND = "#{number} doesn't match any event. Use /events to see the list."

    # ── Search ───────────────────────────────────────────────────────────────

    NO_RESULTS_TEXT = "No results found"
    SEARCH_ERROR = "Failed to search: {error}"
