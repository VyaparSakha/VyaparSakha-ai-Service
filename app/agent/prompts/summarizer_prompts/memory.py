"""
app/agent/prompts/summarizer_prompts/memory.py
================================================
Instruction for compressing mem0 long-term memory results.

Used by ``app/agent/memory.py`` to summarize the (potentially huge)
list of mem0 search results into a compact, factual summary of user
preferences and store knowledge before it enters the main LLM context.
"""

SUMMARIZER_MEMORY_INSTRUCTION = (
    "You are compressing long-term memory for an AI agent. "
    "Produce a compact, factual summary of the user preferences "
    "and store knowledge below. Give STRICT HIGHER PRIORITY to facts "
    "and preferences explicitly stated by the USER over assistant recommendations. "
    "Keep only concrete facts, preferences, and patterns. Drop filler and duplicates. "
    "Return plain text, no markdown headers."
)

MEMORY_QUERY_INSTRUCTION = (
    "Create one concise semantic search query for retrieving relevant long-term "
    "memory for Vyapar Copilot. Use the current user request below. "
    "Keep only stable topics, preferences, store facts, prior decisions, and "
    "recurring patterns directly relevant to the user request. "
    "Do not include generic assistant scope text, IDs, secrets, refusal text, "
    "or temporary numeric results. Return one plain-text query of at most 20 words."
)

MEMORY_EXTRACTION_INSTRUCTION = (
    "Extract only durable long-term memory from this retail assistant exchange. "
    "Keep store-specific facts, recurring inventory or sales patterns, explicit "
    "business decisions, and user preferences about language, tone, or detail. "
    "Exclude greetings, one-off questions, temporary tool results, model/refusal "
    "text, personal identifiers, credentials, and sensitive financial data. "
    "If there is nothing durable, return exactly NONE. Return concise plain text."
)

MULTI_LEVEL_MEMORY_EXTRACTION_INSTRUCTION = (
    "You are an expert AI Long-Term Memory Extractor for a retail business copilot (Vyapar Sakha).\n"
    "Analyze the conversation exchange between the User (store owner/manager) and the Assistant.\n"
    "Extract ONLY durable, reusable facts and categorize them strictly into three distinct memory levels:\n\n"
    "1. USER PREFERENCES ('user_preferences'):\n"
    "   - Communication style (e.g. prefers detailed point-wise reports, bullet points, concise summaries, tables).\n"
    "   - Language/tone preferences (e.g. prefers Hindi, English, Hinglish, formal tone).\n"
    "   - Long-term strategic business goals stated by user (e.g. 'targeting 15% margin growth in Q3').\n"
    "   - EXCLUDE: One-off questions, specific current stock queries, temporary status requests, greetings, code/programming queries.\n\n"
    "2. STORE KNOWLEDGE ('store_knowledge'):\n"
    "   - Store-specific operational rules & facts (e.g. 'Supplier Amul delivers on Tuesdays at 8 AM', 'Store is closed on Sundays').\n"
    "   - Recurring local sales trends, peak footfall hours, local store policies.\n"
    "   - Explicit inventory decisions made for this store (e.g. 'Minimum safety stock for Milk set to 20 units').\n"
    "   - EXCLUDE: Ephemeral tool responses, temporary stock levels ('Laptop stock is 2'), raw chat turns, generic advice.\n\n"
    "3. MULTI-STORE KNOWLEDGE ('multi_store_knowledge'):\n"
    "   - Cross-store inventory transfer rules (e.g. 'Rebalance excess stock from Main Warehouse to Store B when stock > 100').\n"
    "   - Multi-location volume discounts or chain-wide supplier agreements.\n"
    "   - Enterprise multi-store business strategies.\n"
    "   - EXCLUDE: Single-store facts, individual store stock counts.\n\n"
    "CRITICAL RULES:\n"
    "- If a category has NO new durable facts in this exchange, return an empty array [] for that category.\n"
    "- Do NOT extract raw Q&A logs, greetings ('hi', 'hello'), out-of-scope requests (e.g. Java code), or temporary numerical stock counts.\n"
    "- Return ONLY a valid JSON object matching this exact schema:\n"
    "{\n"
    '  "user_preferences": ["concise fact 1", ...],\n'
    '  "store_knowledge": ["concise fact 1", ...],\n'
    '  "multi_store_knowledge": ["concise fact 1", ...]\n'
    "}"
)

MEMORY_RECONCILIATION_INSTRUCTION = (
    "You are an expert AI Memory Reconciler for a retail store copilot.\n"
    "Compare the NEW CANDIDATE FACT against existing stored memories.\n"
    "Maintain strict consistency without duplication, contradiction, or stale facts.\n\n"
    "Determine appropriate action for each item:\n"
    "- UPDATE: If the new fact updates or refines an existing memory (output consolidated text).\n"
    "- DELETE: If the new fact contradicts or invalidates an existing memory.\n"
    "- ADD: If the new fact is completely new and not covered by existing memories.\n"
    "- NO_CHANGE: If the new fact is already fully covered and identical to an existing memory.\n\n"
    "Return ONLY a valid JSON object matching this schema:\n"
    "{\n"
    '  "actions": [\n'
    '    {"action": "UPDATE", "id": "mem_id", "text": "updated consolidated fact"},\n'
    '    {"action": "DELETE", "id": "mem_id"},\n'
    '    {"action": "ADD", "text": "new memory text"},\n'
    '    {"action": "NO_CHANGE", "id": "mem_id"}\n'
    "  ]\n"
    "}"
)
