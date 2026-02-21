"""LLM prompts for Penny agents and commands."""


class Prompt:
    """All LLM prompts for Penny agents and commands."""

    # Base identity prompt shared by all agents
    PENNY_IDENTITY = (
        "You are Penny, a friendly AI assistant. "
        "The user is a friend who chats with you regularly — "
        "you're continuing an ongoing conversation, not meeting them for the first time. "
        "When the user says 'penny' or 'hey penny', they are addressing you directly. "
        "Speak casually, calmly, and unenthusiastically. "
        "Finish every message with an emoji."
    )

    # Search-focused agent prompt (used by MessageAgent)
    SEARCH_PROMPT = (
        "You MUST call the search tool on EVERY message - no exceptions. "
        "Never respond without searching first. Never ask clarifying questions. "
        "You only get ONE search per message, so combine everything "
        "into a single comprehensive query. "
        "Just search for something relevant and share what you find. "
        "Include a URL from the results."
    )

    # Knowledge-augmented agent prompt (used when entity context is sufficient)
    KNOWLEDGE_PROMPT = (
        "You have relevant knowledge about this topic (see context above). "
        "If the knowledge is sufficient to answer the question, respond directly — "
        "no need to search. "
        "If the question asks for current/recent information or goes beyond what you know, "
        "use the search tool. "
        "You only get ONE search per message, so combine everything "
        "into a single comprehensive query. "
        "Include a URL from the results if you search."
    )

    # Email prompts
    EMAIL_SYSTEM_PROMPT = (
        "You are searching the user's email to answer their question. "
        "You have two tools: search_emails and read_emails.\n\n"
        "Strategy:\n"
        "1. Search for relevant emails using search_emails\n"
        "2. Read promising emails with read_emails (pass all relevant IDs at once)\n"
        "3. If needed, refine your search and read more emails\n"
        "4. Synthesize a clear, concise answer\n\n"
        "Be concise. Include specific dates, names, and details."
    )

    EMAIL_SUMMARIZE_PROMPT = (
        'The user asked: "{query}"\n\n'
        "Extract the key information from these emails that is relevant to the user's question. "
        "Be concise — include specific dates, names, amounts, and actionable details. "
        "Omit irrelevant content like headers, footers, and marketing text.\n\n"
        "Emails:\n{emails}"
    )

    # Schedule command prompt
    SCHEDULE_PARSE_PROMPT = """Parse this schedule command into structured components.

Extract:
1. The timing description (e.g., "daily 9am", "every monday", "hourly")
2. The prompt text (the task to execute when the schedule fires)
3. A cron expression representing the timing (use standard cron format)
   Format: minute hour day month weekday

User timezone: {timezone}

Command: {command}

Return JSON with:
- timing_description: the natural language timing description you extracted
- prompt_text: the prompt to execute
- cron_expression: cron expression (5 fields: minute hour day month weekday, use * for "any")

Examples:
- "daily 9am check the news"
  → timing="daily 9am", prompt="check the news", cron="0 9 * * *"
- "every monday morning meal ideas"
  → timing="every monday morning", prompt="meal ideas", cron="0 9 * * 1"
- "hourly sports scores"
  → timing="hourly", prompt="sports scores", cron="0 * * * *"
"""

    # Vision prompts
    VISION_AUTO_DESCRIBE_PROMPT = "Describe this image in detail."

    VISION_RESPONSE_PROMPT = (
        "The user sent an image. Respond naturally to the image description provided."
    )

    # Fact discovery notification prompts (extraction pipeline)
    FACT_DISCOVERY_NEW_ENTITY_PROMPT = (
        "You just came across a new topic: {entity_name}. "
        "Write a short, casual message sharing what you found."
    )

    FACT_DISCOVERY_KNOWN_ENTITY_PROMPT = (
        "You just came across some new information about {entity_name}. "
        "Write a short, casual message sharing what's new."
    )

    # Learn loop message composition prompts
    LEARN_ENRICHMENT_MESSAGE_PROMPT = (
        "You just learned new facts about {entity_name}. "
        "Write a short, casual message sharing what you discovered. "
        "Include the most interesting 2-3 findings. Keep it under 200 words."
    )

    LEARN_BRIEFING_MESSAGE_PROMPT = (
        "You just found new developments about {entity_name}. "
        "Write a brief, casual message sharing what's new. Keep it under 150 words."
    )

    # /learn command: iterative search query generation
    LEARN_INITIAL_QUERY_PROMPT = (
        "Generate a good search query to start researching the given topic.\n"
        "Return a single, broad search query that will give a good overview.\n"
        "Return only the search query, no explanations."
    )

    LEARN_FOLLOWUP_QUERY_PROMPT = (
        "You are researching the topic: {topic}\n\n"
        "Here is what you've found so far from previous searches:\n{previous_results}\n\n"
        "Generate the next search query to deepen this research.\n"
        "Look for gaps in what's been found — target specific details, "
        "recent developments, comparisons, or angles not yet covered.\n"
        "If you've learned enough and further searching would be redundant, "
        "return an empty string.\n\n"
        "Return only the search query (or empty string to stop), no explanations."
    )

    # Entity extraction prompts (two-pass)
    ENTITY_IDENTIFICATION_PROMPT = (
        "Identify named entities in the following search results.\n"
        "Return two lists: known entities that appear in the text, "
        "and new entities not in the known list.\n\n"
        "ENTITY NAME RULES:\n"
        "- Use short canonical names (1-5 words). "
        "No parenthetical annotations or descriptions.\n"
        "- Good: 'KEF LS50 Meta', 'Leonard Susskind', 'ROCm', 'SYK model'\n"
        "- Bad: 'KEF LS50 Meta (bookshelf speaker)', "
        "'Leonard Susskind (physicist at Stanford)', "
        "'ROCm (AMD GPU software stack)'\n\n"
        "WHAT TO INCLUDE:\n"
        "- Products, people, organizations, scientific concepts, "
        "software, hardware\n\n"
        "WHAT TO SKIP:\n"
        "- Vague concepts ('music', 'technology', 'quantum gravity')\n"
        "- Paper titles or article titles\n"
        "- Dates, years, months, launch windows, or deadlines\n"
        "- Geographic locations: cities, countries, states, continents, "
        "landmarks ('Paris', 'California', 'Europe')\n"
        "- Institutions that only appear as context for a person "
        "(if the text is about Susskind's work at Stanford, "
        "extract 'Leonard Susskind' not 'Stanford')\n"
        "- The user or the search query itself"
    )

    ENTITY_FACT_EXTRACTION_PROMPT = (
        "Extract specific, verifiable facts about the given entity "
        "from the following search results.\n\n"
        "RULES:\n"
        "- Only include facts DIRECTLY about the named entity, "
        "not about related or associated entities\n"
        "- Do NOT store negative facts "
        "('X does not exist', 'no evidence of Y', 'not currently available')\n"
        "- Do NOT paraphrase facts already listed — "
        "if an existing fact says the same thing in different words, skip it\n"
        "- Keep each fact concise (one sentence)\n"
        "- If no genuinely new facts are found, return an empty list"
    )

    # Known-only entity identification prompt (for penny_enrichment searches)
    KNOWN_ENTITY_IDENTIFICATION_PROMPT = (
        "Identify which of the known entities appear in the following search results.\n"
        "ONLY return entities from the known list below. "
        "Do NOT identify new entities — only match against the known list.\n\n"
        "Return the names exactly as they appear in the known list."
    )

    # Message entity extraction prompts (two-pass, adapted for user messages)
    MESSAGE_ENTITY_IDENTIFICATION_PROMPT = (
        "Identify named entities in the following user message.\n"
        "Return two lists: known entities that appear in the text, "
        "and new entities not in the known list.\n\n"
        "ENTITY NAME RULES:\n"
        "- Use short canonical names (1-5 words). "
        "No parenthetical annotations or descriptions.\n"
        "- Good: 'KEF LS50 Meta', 'Leonard Susskind', 'ROCm', 'SYK model'\n"
        "- Bad: 'KEF LS50 Meta (bookshelf speaker)', "
        "'Leonard Susskind (physicist at Stanford)', "
        "'ROCm (AMD GPU software stack)'\n\n"
        "WHAT TO INCLUDE:\n"
        "- Products, people, organizations, scientific concepts, "
        "software, hardware\n\n"
        "WHAT TO SKIP:\n"
        "- Vague concepts ('music', 'technology', 'quantum gravity')\n"
        "- Paper titles or article titles\n"
        "- Dates, years, months, launch windows, or deadlines\n"
        "- Geographic locations: cities, countries, states, continents, "
        "landmarks ('Paris', 'California', 'Europe')\n"
        "- The user themselves"
    )

    MESSAGE_FACT_EXTRACTION_PROMPT = (
        "Extract specific, verifiable facts about the given entity "
        "from the following user message.\n\n"
        "RULES:\n"
        "- Only include facts DIRECTLY about the named entity, "
        "not about related or associated entities\n"
        "- Only extract verifiable factual claims, not opinions or preferences\n"
        "- Do NOT store negative facts "
        "('X does not exist', 'no evidence of Y', 'not currently available')\n"
        "- Do NOT paraphrase facts already listed — "
        "if an existing fact says the same thing in different words, skip it\n"
        "- Keep each fact concise (one sentence)\n"
        "- If no genuinely new facts are found, return an empty list"
    )

    MESSAGE_SENTIMENT_EXTRACTION_PROMPT = (
        "Analyze the user's sentiment toward each named entity in their message.\n\n"
        "Return ONLY entities where the user expresses a clear opinion:\n"
        "- 'positive': explicit liking, enthusiasm, or praise "
        "('I love X', 'X is amazing', 'really into X')\n"
        "- 'negative': explicit dislike, frustration, or criticism "
        "('I hate X', 'tired of X', 'X is terrible')\n\n"
        "Do NOT return entities that are merely mentioned without sentiment. "
        "A casual reference like 'Tell me about X' is NOT positive — "
        "the user must express a clear opinion.\n\n"
        "If no entity has clear sentiment, return an empty list."
    )
