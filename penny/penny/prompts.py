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
        "When sharing information, use markdown formatting: "
        "**bold** for key terms and titles, bullet points for lists of items, "
        "and clear paragraph breaks for readability. "
        "Finish every message with an emoji."
    )

    # Search-focused agent prompt (used by MessageAgent)
    SEARCH_PROMPT = (
        "You MUST call the search tool on EVERY message - no exceptions. "
        "Never respond without searching first. Never ask clarifying questions. "
        "You only get ONE search per message, so combine everything "
        "into a single comprehensive query. "
        "Just search for something relevant and share what you find. "
        "Include a URL from the results. "
        "Format your response with **bold** for key names and terms, "
        "and use bullet points when listing multiple items or findings."
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
        "Include a URL from the results if you search. "
        "Format your response with **bold** for key names and terms, "
        "and use bullet points when listing multiple items or findings."
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
        "Be concise. Include specific dates, names, and details. "
        "Use **bold** for key terms, dates, and names. "
        "Use bullet points when summarizing multiple emails or findings."
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
        "Write a short, casual message sharing what you found. "
        "Open by telling the user you found a new topic "
        "worth tracking: **{entity_name}**{descriptor}. "
        "Synthesize the facts below into natural sentences "
        "— don't just list them verbatim. "
        "Use **bold** for the topic name."
    )

    FACT_DISCOVERY_KNOWN_ENTITY_PROMPT = (
        "You just came across some new information about {entity_name}. "
        "Write a short, casual message sharing what's new. "
        "Open by telling the user this is an update "
        "on **{entity_name}**{descriptor}. "
        "Synthesize the facts below into natural sentences "
        "— don't just list them verbatim. "
        "Use **bold** for the topic name."
    )

    # Learn-topic-aware variants (when facts originated from a /learn command)
    FACT_DISCOVERY_NEW_ENTITY_LEARN_PROMPT = (
        "While researching {learn_topic} (something the user asked you to look into), "
        "you came across a new topic: {entity_name}. "
        "Write a short, casual message sharing what you found. "
        "Open by telling the user you found a new topic "
        "while looking into **{learn_topic}**: "
        "**{entity_name}**{descriptor}. "
        "Synthesize the facts below into natural sentences "
        "— don't just list them verbatim. "
        "Use **bold** for topic names."
    )

    FACT_DISCOVERY_KNOWN_ENTITY_LEARN_PROMPT = (
        "While researching {learn_topic} (something the user asked you to look into), "
        "you came across some new information about {entity_name}. "
        "Write a short, casual message sharing what's new. "
        "Open by telling the user this is an update on "
        "**{entity_name}**{descriptor}, found while "
        "looking into **{learn_topic}**. "
        "Synthesize the facts below into natural sentences "
        "— don't just list them verbatim. "
        "Use **bold** for topic names."
    )

    # Learn completion summary prompt
    LEARN_COMPLETION_SUMMARY_PROMPT = (
        "You just finished researching **{topic}** (something the user asked you to look into). "
        'Open with "Here\'s what I learned about {topic}" then write a casual '
        "summary of what you found. "
        "Group findings by topic, highlight the most interesting facts, "
        "and use **bold** for topic names and bullet points for key facts. "
        "Keep it concise but informative."
    )

    # Learn agent message composition prompts
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
        "- The user or the search query itself\n\n"
        "TAGLINE:\n"
        "For each new entity, provide a short tagline (3-8 words) describing "
        "what the entity is. The tagline is a summary you extrapolate, "
        "not a verbatim quote from the text.\n"
        "- 'KEF LS50 Meta' → 'bookshelf speaker by kef'\n"
        "- 'Leonard Susskind' → 'theoretical physicist at stanford'\n"
        "- 'Genesis' → 'british progressive rock band'\n"
        "- 'ROCm' → 'amd gpu software platform'"
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
        "- The user themselves\n\n"
        "TAGLINE:\n"
        "For each new entity, provide a short tagline (3-8 words) describing "
        "what the entity is. The tagline is a summary you extrapolate, "
        "not a verbatim quote from the text.\n"
        "- 'KEF LS50 Meta' → 'bookshelf speaker by kef'\n"
        "- 'Leonard Susskind' → 'theoretical physicist at stanford'\n"
        "- 'Genesis' → 'british progressive rock band'\n"
        "- 'ROCm' → 'amd gpu software platform'"
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

    # /follow command: parse timing + topic from natural language input
    FOLLOW_PARSE_PROMPT = """Parse this follow command into structured components.

Extract:
1. The timing description (e.g., "daily 9:30am", "every monday", "hourly")
2. The topic text (the subject to monitor for news)
3. A cron expression representing the timing (use standard cron format)
   Format: minute hour day month weekday

If no timing is specified, default to "daily" with cron "0 9 * * *".

User timezone: {timezone}

Command: {command}

Return JSON with:
- timing_description: the natural language timing description you extracted
- topic_text: the topic to monitor
- cron_expression: cron expression (5 fields: minute hour day month weekday, use * for "any")

Examples:
- "daily 9:30am usa news"
  → timing="daily 9:30am", topic="usa news", cron="30 9 * * *"
- "every monday morning tech"
  → timing="every monday morning", topic="tech", cron="0 9 * * 1"
- "hourly spacex launches"
  → timing="hourly", topic="spacex launches", cron="0 * * * *"
- "artificial intelligence"
  → timing="daily", topic="artificial intelligence", cron="0 9 * * *"
"""

    # /follow command: generate search query terms from a user topic
    FOLLOW_QUERY_TERMS_PROMPT = (
        "Generate search query terms for monitoring news about the given topic.\n"
        "Return a JSON object with a single key 'query_terms': a list of 2-4 short search phrases "
        "that would find relevant news articles.\n\n"
        "The terms should cover different angles or synonyms to maximize coverage.\n"
        "Each term should be 1-4 words.\n\n"
        "Example:\n"
        'Topic: "artificial intelligence safety"\n'
        "Response: "
        '{{"query_terms": ["AI safety", "AI risk", "AI alignment"]}}\n\n'
        "Topic: {topic}\n"
        "Return only the JSON object, no explanations."
    )

    # Event agent: extract topic tags from a headline for relevance matching
    EVENT_TAG_EXTRACTION_PROMPT = (
        "Extract 2-4 one-word topic tags from this headline. "
        "Return ONLY a JSON list of lowercase strings.\n"
        'Example: "Recent breakthroughs in quantum biology"'
        ' -> ["science", "biology", "quantum"]\n'
        'Example: "Tesla stock surges after record deliveries"'
        ' -> ["business", "automotive", "tesla"]\n\n'
        'Headline: "{headline}"'
    )

    # Event agent: extract entity names from a news article
    EVENT_ENTITY_EXTRACTION_PROMPT = (
        "Identify named entities mentioned in the following news article.\n"
        "Return a JSON object with a single key 'entities': a list of entity names.\n\n"
        "ENTITY NAME RULES:\n"
        "- Use short canonical names (1-5 words)\n"
        "- Include: products, people, organizations, scientific concepts, software\n"
        "- Exclude: vague concepts, dates, locations, article titles\n\n"
        "Return only the JSON object, no explanations."
    )

    # Event notification prompt (for proactive event announcements)
    EVENT_NOTIFICATION_PROMPT = (
        "You just saw a news headline relevant to the user's follow topic. "
        "Write a short, casual heads-up message about it. "
        "Open by telling the user this is an update on their "
        "follow topic (mention the topic by name). "
        "Synthesize the headline and summary into a natural message. "
        "Use **bold** for key names and topics. "
        "Keep it concise — one short paragraph. "
        "End with the source URL on its own line so the user can read the full story."
    )

    # Enrichment entity discovery prompt
    ENRICHMENT_ENTITY_DISCOVERY_PROMPT = (
        "Identify notable entities RELATED TO {entity_name} "
        "mentioned in the following search results.\n"
        "Return only NEW entities — do NOT include {entity_name} itself "
        "or any entity from the known list.\n\n"
        "ENTITY NAME RULES:\n"
        "- Use short canonical names (1-5 words). "
        "No parenthetical annotations or descriptions.\n"
        "- Good: 'Uni-Q driver', 'Andrew Jones', 'Metamaterial Absorption Technology'\n"
        "- Bad: 'Uni-Q driver (coaxial driver by KEF)', "
        "'Andrew Jones (speaker designer)'\n\n"
        "WHAT TO INCLUDE:\n"
        "- Sub-components, technologies, people, related products, "
        "organizations closely tied to {entity_name}\n\n"
        "WHAT TO SKIP:\n"
        "- Vague concepts ('audio quality', 'speaker design')\n"
        "- Dates, years, locations\n"
        "- Entities only loosely mentioned in passing\n"
        "- The entity being researched: {entity_name}\n\n"
        "TAGLINE:\n"
        "For each entity, provide a short tagline (3-8 words) describing what it is.\n"
        "- 'Uni-Q driver' → 'coaxial driver array by kef'\n"
        "- 'Andrew Jones' → 'speaker designer at kef'"
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
