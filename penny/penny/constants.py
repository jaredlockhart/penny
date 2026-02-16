"""Constants and prompts for Penny agent."""

from enum import StrEnum
from pathlib import Path


class MessageDirection(StrEnum):
    """Direction of a logged message."""

    INCOMING = "incoming"
    OUTGOING = "outgoing"


class PreferenceType(StrEnum):
    """Type of user preference."""

    LIKE = "like"
    DISLIKE = "dislike"


# Reaction emoji mappings for sentiment analysis
LIKE_REACTIONS = ("❤️", "👍", "😆")
DISLIKE_REACTIONS = ("😠", "👎", "😢")

# Max messages/reactions the PreferenceAgent processes per user per wake cycle.
# Keeps LLM prompts small; remaining items are processed on subsequent invocations.
PREFERENCE_BATCH_LIMIT = 20

# Base identity prompt shared by all agents
PENNY_IDENTITY = (
    "You are Penny, a friendly AI assistant. "
    "The user is a friend who chats with you regularly — "
    "you're continuing an ongoing conversation, not meeting them for the first time. "
    "When the user says 'penny' or 'hey penny', they are addressing you directly."
)

# Message agent prompt (search-focused)
SYSTEM_PROMPT = (
    f"{PENNY_IDENTITY}\n\n"
    "Skip greetings like 'hey!' - just dive into the topic. "
    "You MUST call the search tool on EVERY message - no exceptions. "
    "Never respond without searching first. Never ask clarifying questions. "
    "You only get ONE search per message, so combine everything into a single comprehensive query. "
    "Just search for something relevant and share what you find. "
    "Include a URL from the results. Keep it relaxed and low-key, end with an emoji."
)

# Search tool constants
PERPLEXITY_PRESET = "pro-search"
NO_RESULTS_TEXT = "No results found"
IMAGE_MAX_RESULTS = 3
IMAGE_DOWNLOAD_TIMEOUT = 15.0
URL_BLOCKLIST_DOMAINS = (
    "play.google.com",
    "apps.apple.com",
)

FOLLOWUP_PROMPT = (
    "Follow up on this conversation by searching for something new about the topic. "
    "Open casually, then share what you found. "
    "Keep it short, like texting a friend."
)

DISCOVERY_PROMPT = (
    "Pick ONE specific topic from this user's interests - not multiple, just one. "
    "Search for something new and interesting about that single topic. "
    "Don't reference past conversations - just share a cool discovery out of the blue. "
    "Stay focused on that one thing, don't mention their other interests. "
    "Open casually, keep it short, like texting a friend."
)

RESEARCH_PROMPT = (
    "You are conducting deep research on a topic. "
    "Search for comprehensive information and analyze the results. "
    "Structure your findings starting with the highest-level insights first, "
    "then break down into increasingly specific details. "
    "Based on what you find, determine what specific aspect or angle to investigate next. "
    "Be thorough and systematic - cover different perspectives, recent developments, "
    "and key details."
)

RESEARCH_EXTRACTION_PROMPT = (
    "Given the user's research topic and the following search results, "
    "extract a concise bulleted list of the relevant information that satisfies "
    "the research topic. Include only facts, findings, and details that are directly "
    "relevant. Omit filler, redundant information, and irrelevant content. "
    "Use short, information-dense bullet points."
)

RESEARCH_SUMMARY_PROMPT = (
    "Based on all research findings below, write a detailed executive summary "
    "that captures the highest-level insights about the topic. Start with the most important "
    "information first, then break down into increasingly specific details as you go. "
    "Use markdown formatting and bullet points to organize the information clearly. "
    "Do not include tables."
)

RESEARCH_OUTPUT_OPTIONS_SYSTEM_PROMPT = (
    "You suggest what information a research summary report should focus on. "
    "The output is always a plain text report — never a database, app, visual, or interactive tool. "
    "Output ONLY a numbered list of exactly 3 options (1., 2., 3.). "
    "Each option describes what information and structure the report should have. "
    "No preamble, no explanation, just the 3 numbered options."
)

RESEARCH_OUTPUT_OPTIONS_PROMPT = (
    "Someone wants to research: {topic}\n\n"
    "Suggest 3 ways to structure the summary report. Examples:\n"
    "- A ranked list of top picks with pros, cons, and key details for each\n"
    "- A comprehensive catalog covering every item found with dates and details\n"
    "- A brief executive summary with the top 5 highlights and actionable takeaways\n"
    "- A side-by-side comparison organized by key criteria\n"
    "- A chronological breakdown with dates, locations, and highlights"
)

RESEARCH_FOLLOWUP_PROMPT = (
    "You are conducting deep research. Review your previous findings above "
    "and search for NEW information you haven't covered yet. "
    "Try a different search angle — different keywords, a different aspect of the topic, "
    "or drill deeper into a specific area. Do NOT repeat searches you've already done. "
    "Structure your findings starting with the most important new insights."
)

RESEARCH_FOCUS_EXTRACTION_PROMPT = (
    "You are interpreting what the user wants in a research report. "
    "They were shown some suggested options and replied. "
    "Based on the options and their reply, output a short phrase describing "
    "what information and structure the report should have. "
    "The user may pick an option, modify one, or describe something entirely different. "
    "Output ONLY the focus description — no preamble, no explanation."
)

RESEARCH_FOCUS_TIMEOUT_SECONDS = 300

PREFERENCE_PROMPT = (
    f"{PENNY_IDENTITY}\n\n"
    "Your task: Based on these messages from the user, create a flat list of topics "
    "THE USER has mentioned or asked about (not yourself). "
    "Extract all mentioned topics as a simple bulleted list. "
    "Include specific things like: bands, albums, technologies, places, hobbies, "
    "concepts, products, events, or anything else they've discussed.\n\n"
    "Format as a flat list with one topic per line (no categories or descriptions). "
    "Example:\n"
    "- guitar\n"
    "- vinyl collecting\n"
    "- quantum gravity\n"
    "- Toronto\n\n"
    "Messages:\n"
)

# Email command constants
JMAP_SESSION_URL = "https://api.fastmail.com/jmap/session"
JMAP_REQUEST_TIMEOUT = 30.0
EMAIL_BODY_MAX_LENGTH = 4000
EMAIL_NO_QUERY_TEXT = "Please ask a question about your email. Usage: /email <question>"

EMAIL_SYSTEM_PROMPT = (
    f"{PENNY_IDENTITY}\n\n"
    "You are searching the user's email to answer their question. "
    "You have two tools: search_emails and read_emails.\n\n"
    "Strategy:\n"
    "1. Search for relevant emails using search_emails\n"
    "2. Read promising emails with read_emails (pass all relevant IDs at once)\n"
    "3. If needed, refine your search and read more emails\n"
    "4. Synthesize a clear, concise answer\n\n"
    "Be concise. Include specific dates, names, and details. "
    "Keep it relaxed, like texting a friend."
)

EMAIL_SUMMARIZE_PROMPT = (
    'The user asked: "{query}"\n\n'
    "Extract the key information from these emails that is relevant to the user's question. "
    "Be concise — include specific dates, names, amounts, and actionable details. "
    "Omit irrelevant content like headers, footers, and marketing text.\n\n"
    "Emails:\n{emails}"
)

# Schedule command response messages
SCHEDULE_NO_TASKS = "You don't have any scheduled tasks yet 📅"
SCHEDULE_NEED_TIMEZONE = (
    "I need to know your timezone first. Send me your location or tell me your city 📍"
)
SCHEDULE_PARSE_ERROR = (
    "Sorry, I couldn't understand that schedule format. "
    "Try something like: /schedule daily 9am what's the news?"
)
SCHEDULE_INVALID_CRON = (
    "Sorry, I couldn't figure out the timing. "
    "Try something like: /schedule daily 9am what's the news?"
)
SCHEDULE_DELETED_NO_REMAINING = "No more scheduled tasks"
SCHEDULE_STILL_SCHEDULED = "Still scheduled:"
SCHEDULE_INVALID_NUMBER = "Invalid schedule number: {number}"
SCHEDULE_NO_SCHEDULE_WITH_NUMBER = "No schedule with number {number}"
SCHEDULE_DELETED_PREFIX = "Deleted '{timing} {prompt}' ✅"
SCHEDULE_ADDED = "Added {timing}: {prompt} ✅"

# Test mode constants
TEST_DB_PATH = Path("data/penny-test.db")
TEST_MODE_PREFIX = "[TEST] "

# GitHub constants
GITHUB_REPO_OWNER = "jaredlockhart"
GITHUB_REPO_NAME = "penny"

# Vision constants
VISION_AUTO_DESCRIBE_PROMPT = "Describe this image in detail."
VISION_NOT_CONFIGURED_MESSAGE = (
    "I can see you sent an image but I don't have vision configured right now."
)
VISION_SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
VISION_IMAGE_CONTEXT = "user said '{user_text}' and included an image of: {caption}"
VISION_IMAGE_ONLY_CONTEXT = "user sent an image of: {caption}"
VISION_RESPONSE_PROMPT = (
    f"{PENNY_IDENTITY}\n\n"
    "The user sent an image. Respond naturally to the image description provided. "
    "Keep it relaxed and low-key, like texting a friend. End with an emoji."
)

# Personality command response messages
PERSONALITY_NO_CUSTOM = "No custom personality set. Using default Penny personality."
PERSONALITY_RESET_SUCCESS = "Ok, personality reset to default ✅"
PERSONALITY_RESET_NOT_SET = "No custom personality was set."
PERSONALITY_UPDATE_SUCCESS = "Ok, personality updated ✅"

# Personality transform prompt
PERSONALITY_TRANSFORM_PROMPT = (
    "You are a text rewriter. Rewrite the user's message in this style: {personality_prompt}\n\n"
    "RULES:\n"
    "- Output ONLY the rewritten version of the entire message\n"
    "- Keep ALL information, lists, options, and instructions intact\n"
    "- Preserve ALL markdown formatting exactly: headings (##, ###), "
    "bold (**), italic (*), lists (-, 1.), links, and code blocks\n"
    "- Do NOT answer any questions in the message — just rephrase them\n"
    "- Do NOT add or remove content\n"
    "- Only adjust tone, style, and word choice"
)
