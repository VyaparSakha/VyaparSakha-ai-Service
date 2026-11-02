"""
app/routes/voice_routes.py
==========================
WebSocket endpoint for Gemini Realtime API (Multimodal Live API).

This route acts as a relay between the Vyapar Sakha frontend and the
Google Gemini Multimodal API via raw websockets.
"""

import json
import os
import structlog
import asyncio
import base64
import math
import struct
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import websockets

from bson import ObjectId
from app.config.settings import get_settings
from app.config.database import get_database
from app.agent.service.sales.summary import fetch_sales_summary, _resolve_store_id
from app.agent.memory.redis_cache import get_redis
from app.agent.tools.registry import VYAPAR_TOOLS
from app.agent.nodes.memory_query import _bootstrap_user_memory, _bootstrap_store_memory
from app.agent.prompts.voice_prompt import build_voice_system_prompt
_build_voice_system_prompt = build_voice_system_prompt
from app.agent.tools.visual_builders import build_visual_from_tool
from app.agent.tools.visual_builders import build_visual_for_tool

router = APIRouter()
LOGGER = structlog.get_logger("vyaparsathi.ai.voice")


def _decode_gemini_message(msg):
    """Decode Gemini messages safely. Some frames are binary audio chunks, not JSON."""
    if isinstance(msg, (bytes, bytearray)):
        payload = bytes(msg)
        if not payload:
            return None
        # Fast path: binary PCM audio frames do not start with JSON '{'
        if payload.startswith(b"{"):
            try:
                return json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {"binary": payload}
        return {"binary": payload}

    if isinstance(msg, str):
        text = msg.strip()
        if not text:
            return None
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"binary": text.encode("utf-8")}
        return {"binary": text.encode("utf-8")}

    return None


def _clean_gemini_schema(node: Any) -> Any:
    """Recursively clean JSON Schema for Gemini Live API compatibility.

    Gemini Live API rejects unknown fields such as additionalProperties, title,
    default, $defs, allOf, etc.
    """
    if not isinstance(node, dict):
        if isinstance(node, list):
            return [_clean_gemini_schema(item) for item in node]
        return node

    cleaned: Dict[str, Any] = {}

    # Handle anyOf / oneOf (e.g. Optional[T] / Union[T, None])
    if "anyOf" in node or "oneOf" in node:
        sub_schemas = node.get("anyOf") or node.get("oneOf", [])
        non_null = [s for s in sub_schemas if isinstance(s, dict) and s.get("type") != "null"]
        if non_null:
            merged = _clean_gemini_schema(non_null[0])
            if isinstance(merged, dict):
                cleaned.update(merged)
                if any(isinstance(s, dict) and s.get("type") == "null" for s in sub_schemas):
                    cleaned["nullable"] = True
        else:
            cleaned["type"] = "string"

    # Copy allowed schema fields
    for k, v in node.items():
        if k in (
            "additionalProperties", "title", "$defs", "$ref", "default", "examples",
            "prefixItems", "discriminator", "readOnly", "writeOnly", "xml", "externalDocs",
            "example", "deprecated"
        ):
            continue
        if k == "properties" and isinstance(v, dict):
            cleaned["properties"] = {
                prop_name: _clean_gemini_schema(prop_val)
                for prop_name, prop_val in v.items()
            }
        elif k == "items":
            cleaned["items"] = _clean_gemini_schema(v)
        elif k not in ("anyOf", "oneOf"):
            cleaned[k] = _clean_gemini_schema(v) if isinstance(v, (dict, list)) else v

    # Ensure valid type on objects with properties
    if "properties" in cleaned and "type" not in cleaned:
        cleaned["type"] = "object"

    return cleaned


def _get_gemini_tools():
    gemini_tools = []
    for tool in VYAPAR_TOOLS:
        args_schema = getattr(tool, "args_schema", None)
        raw_parameters = args_schema.model_json_schema() if args_schema else {
            "type": "object",
            "properties": {},
        }
        parameters = _clean_gemini_schema(raw_parameters)
        gemini_tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        })
    return [{"functionDeclarations": gemini_tools}]


def _get_live_model_candidates():
    """Return the officially supported Gemini Live model names in priority order.

    Google currently documents Gemini 3.8 Live as the default Live API model and
    lists 2.5 Flash Live as a valid native-audio model. Older model names such as
    gemini-2.0-flash-live-001 are no longer valid and fail with 1008 policy
    violations.
    """
    return [
        "models/gemini-3.8-live",
        "models/gemini-2.5-flash-native-audio-preview-12-2025",
        "models/gemini-3.1-flash-live-preview",
    ]


def _build_gemini_audio_message(audio_bytes: bytes):
    """Build the Live API realtime audio message for raw 16 kHz PCM."""
    return {
        "realtimeInput": {
            "audio": {
                "mimeType": "audio/pcm;rate=16000",
                "data": base64.b64encode(audio_bytes).decode("ascii"),
            }
        }
    }


def _build_gemini_audio_stream_end_message():
    """Build the Live API signal that ends an explicitly managed audio turn."""
    return {"realtimeInput": {"activityEnd": {}}}


def _build_gemini_activity_start_message():
    """Build the Live API signal that starts an explicitly managed audio turn."""
    return {"realtimeInput": {"activityStart": {}}}


def _pcm_rms(audio_bytes: bytes):
    """Return the RMS level of little-endian 16-bit PCM for transport diagnostics."""
    sample_count = len(audio_bytes) // 2
    if sample_count == 0:
        return 0.0
    samples = struct.unpack(f"<{sample_count}h", audio_bytes[:sample_count * 2])
    return math.sqrt(sum(sample * sample for sample in samples) / sample_count)


async def _connect_gemini_live(api_key: str, system_prompt: str):
    """Connect to Gemini Live using the supported model set, with fallback retries."""
    last_error = None
    gemini_ws = None
    for model_name in _get_live_model_candidates():
        ws_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"
        try:
            gemini_ws = await websockets.connect(ws_url)
            setup_msg = {
                "setup": {
                    "model": model_name,
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}]
                    },
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {"voiceName": "Kore"}
                            }
                        }
                    },
                    "inputAudioTranscription": {},
                    "outputAudioTranscription": {},
                    "tools": _get_gemini_tools()
                }
            }
            await gemini_ws.send(json.dumps(setup_msg))

            setup_response = await gemini_ws.recv()
            LOGGER.info("gemini_realtime_connected", model=model_name, response=setup_response)
            return gemini_ws
        except Exception as exc:  # Keep retrying on unsupported/legacy model names.
            last_error = exc
            if gemini_ws is not None:
                await gemini_ws.close()
                gemini_ws = None
            LOGGER.warning(
                "gemini_live_model_retry",
                model=model_name,
                error=str(exc),
            )

    raise last_error or RuntimeError("Unable to establish Gemini Live websocket connection")


from app.agent.memory.redis_cache import (
    get_cached_store_memory,
    get_cached_user_memory,
    set_cached_store_memory,
    set_cached_user_memory,
)


async def _bootstrap_voice_memory_context(user_id: str, store_id: str):
    """
    Multi-Tier Memory Bootstrap:
    1. Check fast Redis cache first.
    2. Fall back to Pinecone vector DB if cache misses.
    3. Cache retrieved memories back into Redis for sub-millisecond future turns.
    """
    user_memories = None
    store_memories = None

    # 1. Check Redis Cache
    try:
        user_memories = await get_cached_user_memory(user_id)
        store_memories = await get_cached_store_memory(store_id)
    except Exception as exc:
        LOGGER.warning("voice_redis_cache_read_failed", error=str(exc))

    # 2. Fall back to Pinecone for user memories if not cached
    if user_memories is None:
        try:
            user_memories = await _bootstrap_user_memory(user_id, "user preferences and details")
            if user_memories:
                await set_cached_user_memory(user_id, user_memories)
        except Exception as exc:
            LOGGER.warning("voice_user_memory_bootstrap_failed", user_id=user_id, error=str(exc))
            user_memories = []

    # 3. Fall back to Pinecone for store memories if not cached
    if store_memories is None:
        try:
            store_memories = await _bootstrap_store_memory(store_id, "store details and facts", intent="general")
            if store_memories:
                await set_cached_store_memory(store_id, store_memories)
        except Exception as exc:
            LOGGER.warning("voice_store_memory_bootstrap_failed", store_id=store_id, error=str(exc))
            store_memories = []

    return user_memories or [], store_memories or []


async def _bootstrap_store_details_context(user_id: str, store_id: str) -> str:
    """Fetch complete live store profile, contact email, user details, and key business KPIs from MongoDB."""
    try:
        db = get_database()
        store_oid = await _resolve_store_id(db, store_id)

        # 1. Fetch Store Profile document
        store_doc = {}
        if store_oid:
            store_doc = (await db["stores"].find_one({"_id": store_oid})) or {}

        store_name = store_doc.get("name") or store_doc.get("storeName") or "Vyapar Store"
        store_email = store_doc.get("email") or store_doc.get("contactEmail") or "Not configured"
        store_phone = store_doc.get("phone") or store_doc.get("contactPhone") or "Not configured"
        store_address = store_doc.get("address") or store_doc.get("city") or "India"
        gst_number = store_doc.get("gstin") or store_doc.get("gstNumber") or "N/A"

        # 2. Fetch User Profile document
        user_name = "Store Owner"
        user_email = store_email
        user_query_clauses = []
        if user_id:
            if ObjectId.is_valid(user_id):
                user_query_clauses.append({"_id": ObjectId(user_id)})
            user_query_clauses.extend([
                {"firebaseUid": user_id},
                {"uid": user_id},
                {"_id": user_id},
                {"email": user_id},
            ])

        # Also fallback to store's owner fields if available
        if store_doc:
            if store_doc.get("owner"):
                o_ref = store_doc["owner"]
                user_query_clauses.append({"_id": o_ref if isinstance(o_ref, ObjectId) else ObjectId(str(o_ref))})
            if store_doc.get("ownerFirebaseUid"):
                user_query_clauses.append({"firebaseUid": store_doc["ownerFirebaseUid"]})

        if user_query_clauses:
            try:
                user_doc = await db["users"].find_one({"$or": user_query_clauses})
                if user_doc:
                    user_name = user_doc.get("name") or user_doc.get("displayName") or user_doc.get("fullName") or user_name
                    user_email = user_doc.get("email") or user_doc.get("userEmail") or store_email
            except Exception as u_err:
                LOGGER.warning("voice_user_lookup_error", error=str(u_err))

        # 3. Calculate Key Store Metrics
        total_products = 0
        low_stock_count = 0
        total_suppliers = 0
        total_customers = 0

        if store_oid:
            total_products = await db["products"].count_documents({"store": store_oid, "isActive": {"$ne": False}})
            low_stock_count = await db["products"].count_documents({
                "store": store_oid,
                "isActive": {"$ne": False},
                "$expr": {"$lte": ["$stock", {"$ifNull": ["$minStock", 5]}]}
            })
            total_suppliers = await db["sellers"].count_documents({"store": store_oid})
            total_customers = await db["buyers"].count_documents({"store": store_oid})

        # 4. Fetch 30-Day Sales KPIs
        sales_kpi = await fetch_sales_summary(store_id, days_lookback=30)
        total_rev = sales_kpi.get("total_revenue", 0.0)
        sales_count = sales_kpi.get("total_sales_count", 0)

        # 5. Fetch Recent Dispatched Emails / Notes from Redis Diary
        recent_notes = []
        try:
            redis_client = await get_redis()
            diary_key = f"store:{store_id}:notes"
            recent_notes_raw = await redis_client.lrange(diary_key, 0, 5)
            recent_notes = [n.decode("utf-8") for n in recent_notes_raw if n]
        except Exception:
            pass

        context_lines = [
            "==================================================",
            "LIVE STORE PROFILE & BUSINESS CONTEXT",
            "==================================================",
            f"- Store Name: {store_name}",
            f"- Store ID: {store_id}",
            f"- Owner / Merchant Name: {user_name}",
            f"- Store Official Email: {store_email}",
            f"- User Account Email: {user_email}",
            f"- Contact Phone: {store_phone}",
            f"- Location: {store_address}",
            f"- GSTIN: {gst_number}",
            "",
            "CURRENT INVENTORY & FINANCIAL OVERVIEW (30 DAYS):",
            f"- Active Catalog Products: {total_products} items",
            f"- Low-Stock Products: {low_stock_count} items requiring restock",
            f"- Connected Suppliers / Distributors: {total_suppliers} vendors",
            f"- Registered Customers / Buyers: {total_customers} clients",
            f"- 30-Day Total Sales Revenue: ₹{total_rev:,.2f} ({sales_count} completed orders)",
        ]

        if recent_notes:
            context_lines.extend([
                "",
                "RECENT MERCHANT DIARY & EMAIL DISPATCH LOGS:",
                * [f"  * {note}" for note in recent_notes]
            ])

        return "\n".join(context_lines)
    except Exception as exc:
        LOGGER.error("voice_store_details_bootstrap_failed", store_id=store_id, error=str(exc))
        return f"STORE PROFILE: Store ID: {store_id}"


async def _persist_voice_session_history_and_memory(
    user_id: str,
    store_id: str,
    transcript: List[Dict[str, str]],
):
    """
    Post-Call Persistence Pipeline:
    1. Deduplicate & clean session transcript.
    2. Create ChatSession document in MongoDB ('agent_chats').
    3. Save each user & assistant message to MongoDB ('agent_chat_messages').
    4. Generate & update intelligent chat session title.
    5. Extract & persist long-term memories into Pinecone & Redis cache.
    """
    if not transcript:
        LOGGER.info("voice_session_persistence_skipped_empty_transcript")
        return

    try:
        from app.services.chat_history_service import (
            create_chat_session,
            add_user_message,
            add_assistant_message,
            generate_chat_title,
            update_chat_session,
        )
        from app.agent.memory import process_and_persist_memory

        # 1. Create Chat Session in MongoDB with 'New Chat' placeholder
        session = await create_chat_session(
            user_id=user_id,
            store_id=store_id,
            title="New Chat",
            metadata={"channel": "voice_realtime", "message_count": len(transcript)},
        )

        # 2. Add all messages to MongoDB agent_chat_messages
        valid_msg_count = 0
        user_prompts = []
        assistant_responses = []

        for msg in transcript:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                await add_user_message(session.chat_id, content)
                user_prompts.append(content)
                valid_msg_count += 1
            elif role == "assistant":
                await add_assistant_message(session.chat_id, content, metadata={"channel": "voice_realtime"})
                assistant_responses.append(content)
                valid_msg_count += 1

        if valid_msg_count == 0:
            LOGGER.warning("voice_session_no_valid_messages_saved", chat_id=session.chat_id)
            return

        # 3. Determine prompt seed for Title Generation
        prompt_seed = user_prompts[0] if user_prompts else (assistant_responses[0] if assistant_responses else "Voice Assistant Conversation")

        # 4. Generate and save smart title with 🎙️ Voice Call badge
        raw_title = await generate_chat_title(session.chat_id, prompt_seed)
        if raw_title and raw_title != "New Chat":
            smart_title = f"🎙️ {raw_title}" if not raw_title.startswith("🎙️") else raw_title
        else:
            smart_title = "🎙️ Voice Call Session"

        await update_chat_session(session.chat_id, title=smart_title, message_count=valid_msg_count)

        LOGGER.info(
            "voice_session_history_persisted",
            chat_id=session.chat_id,
            user_id=user_id,
            store_id=store_id,
            title=smart_title,
            message_count=valid_msg_count,
        )

        # 5. Extract Long-Term Memory & Persist to Pinecone / Redis
        memory_result = await process_and_persist_memory(
            user_id=user_id,
            store_id=store_id,
            messages=transcript,
        )
        LOGGER.info("voice_session_memory_persisted", memory_result=memory_result)

    except Exception as exc:
        LOGGER.error("voice_session_persistence_failed", user_id=user_id, store_id=store_id, error=str(exc), exc_info=True)


def _get_tool_status_label(fn_name: str, args: dict) -> tuple[str, str]:
    if fn_name == "parse_supplier_invoice_image":
        return ("Scanning paper bill photo with Vision AI...", "Invoice items extracted successfully!")
    elif fn_name == "execute_python_code":
        return ("Running custom Python calculation...", "Python calculation completed!")
    elif fn_name == "tool_create_purchase":
        s = args.get("seller_name", "seller")
        return (f"Recording official purchase order from {s} in Purchases & Sellers pages...", "Purchase order created & added to Purchases page!")
    elif fn_name == "tool_receive_purchase":
        p = args.get("purchase_identifier", "invoice")
        return (f"Receiving purchase order {p} & adding items to inventory stock...", f"Purchase order {p} received and stock updated!")
    elif fn_name == "tool_update_purchase":
        p = args.get("purchase_identifier", "invoice")
        return (f"Updating purchase order {p}...", "Purchase order updated successfully!")
    elif fn_name == "tool_delete_purchase":
        p = args.get("purchase_identifier", "invoice")
        return (f"Deleting purchase order {p} & restoring stock...", "Purchase order deleted!")
    elif fn_name == "tool_adjust_stock":
        p = args.get("product_name") or "product"
        qty = args.get("quantity_to_add", "")
        return (f"Adding {qty} units to {p} stock...", f"{p} stock updated successfully!")
    elif fn_name == "search_products":
        q = args.get("query") or args.get("product_name") or args.get("name", "")
        return (f"Searching products {f'for {q}' if q else ''}...", "Product details retrieved!")
    elif fn_name == "get_product_details":
        p = args.get("product_id") or args.get("product_name") or args.get("name", "")
        return (f"Fetching details for {p}...", "Product details retrieved!")
    elif fn_name == "search_sellers":
        q = args.get("query", "")
        return (f"Searching connected sellers and vendors {f'for {q}' if q else ''}...", "Seller details retrieved!")
    elif fn_name == "search_buyers":
        q = args.get("query", "")
        return (f"Searching customer and buyer records {f'for {q}' if q else ''}...", "Buyer details retrieved!")
    elif fn_name == "search_purchases":
        s = args.get("seller_name", "")
        return (f"Searching purchase orders and supplier bills {f'from {s}' if s else ''}...", "Purchase orders retrieved!")
    elif fn_name == "create_smart_purchase_order":
        return ("Calculating lead-time forecast & generating Purchase Order...", "Purchase Order draft ready & saved to Diary!")
    elif fn_name == "send_store_email":
        rec = args.get("recipient_email") or args.get("to") or "recipient"
        return (f"Sending official email to {rec}...", "Email dispatched successfully!")
    elif fn_name == "write_scratchpad_note":
        return ("Writing note to your Merchant Diary...", "Note saved to Merchant Diary!")
    elif fn_name == "read_scratchpad_notes":
        return ("Reading notes from your Merchant Diary...", "Diary notes retrieved!")
    elif fn_name == "update_scratchpad_note":
        return ("Updating note in your Merchant Diary...", "Diary note updated!")
    elif fn_name == "delete_scratchpad_note":
        return ("Removing note from your Merchant Diary...", "Note removed!")
    elif fn_name == "search_memory":
        q = args.get("query", "")
        return (f"Searching store memory for '{q[:30]}'...", "Memory retrieved!")
    elif fn_name == "remember_store_fact":
        return ("Saving new fact to store memory...", "Memory saved and cached!")
    elif fn_name == "get_owner_goals_and_preferences":
        return ("Retrieving your personal business goals...", "Owner goals loaded!")
    elif fn_name == "set_owner_goal_or_preference":
        return ("Updating your personal business goal...", "Goal saved and cached!")
    elif fn_name == "get_daily_action_checklist":
        return ("Preparing your daily action checklist...", "Daily action plan ready!")
    elif fn_name == "analyze_customer_credit_risk":
        return ("Analyzing customer credit & udhaar risk...", "Credit risk analysis ready!")
    elif fn_name == "calculate_restock_budget":
        return ("Calculating restock budget & distributor payouts...", "Restock budget ready!")
    elif fn_name == "get_festival_demand_planner":
        return ("Scanning Indian festival calendar & calculating seasonal demand...", "Festival demand planner ready!")
    elif fn_name == "generate_deal_bundle_recommendation":
        return ("Generating profitable promotional combos...", "Bundle deals ready!")
    elif fn_name == "get_store_goal_progress_report":
        return ("Calculating progress towards your monthly goal...", "Goal progress report ready!")
    elif fn_name == "manage_merchant_memories":
        return ("Accessing your long-term merchant preferences & profile...", "Merchant memory updated!")
    elif fn_name == "manage_proactive_insights":
        return ("Scanning stock velocity, prices, & festival calendar for proactive advice...", "Proactive recommendations ready!")
    elif fn_name == "get_store_baselines":
        return ("Calculating 30-day store benchmarks, peak hours, & sales velocity...", "Store baselines ready!")
    elif fn_name == "check_supplier_price_trends":
        return ("Checking historical supplier purchase prices & price hike trends...", "Price trends retrieved!")
    elif fn_name == "get_customer_credit_ledger":
        return ("Fetching customer Udhar/Khata credit ledger & overdue balances...", "Credit ledger ready!")
    elif fn_name in ("get_low_stock_products", "get_inventory_summary"):
        return ("Scanning inventory levels...", "Inventory data ready!")
    elif fn_name in ("get_sales_summary", "get_daily_sales_trend", "get_top_selling_products"):
        return ("Analyzing sales performance...", "Sales analytics ready!")
    elif fn_name in ("get_demand_forecast", "get_restock_priorities"):
        return ("Calculating demand forecast...", "Forecast complete!")
    elif fn_name == "tool_update_product":
        return (f"Updating product {args.get('name', '')}...", "Product updated!")
    elif fn_name == "tool_navigate_page":
        p = args.get("page", "")
        return (f"Navigating to {p}...", "Navigation complete!")
    elif fn_name == "tool_add_billing_item":
        return (f"Adding {args.get('quantity', 1)} x {args.get('product_name', 'item')} to POS...", "Added to bill!")
    elif fn_name == "tool_generate_bill":
        return ("Generating final bill...", "Bill generated successfully!")
    elif fn_name == "tool_create_automation":
        t = args.get("title", "rule")
        return (f"Scheduling automated rule '{t}' in background scheduler...", "Automation rule scheduled successfully!")
    elif fn_name == "tool_list_automations":
        return ("Fetching your scheduled automation workflows...", "Automations retrieved!")
    elif fn_name == "tool_toggle_automation":
        s = args.get("status", "ACTIVE")
        return (f"Setting automation rule to {s}...", f"Automation {s.lower()} successfully!")
    elif fn_name == "tool_trigger_automation":
        return ("Triggering automation workflow immediately...", "Automation executed successfully!")
    elif fn_name == "tool_delete_automation":
        return ("Deleting automation rule from scheduler...", "Automation deleted!")
    elif fn_name == "invoke_email_composer":
        rec = args.get("recipient_name") or args.get("recipient_email") or "recipient"
        return (f"Composing beautifully styled business email for {rec}...", "Email draft composed & formatted!")
    elif fn_name == "invoke_document_generation":
        t = args.get("title", "document")
        return (f"Generating business document '{t}' and uploading to R2...", "Document generated successfully!")
    else:
        clean_name = fn_name.replace("get_", "").replace("tool_", "").replace("_", " ").title()
        return (f"Executing {clean_name}...", f"{clean_name} complete!")


@router.websocket("/ws/voice")
async def voice_assistant_websocket(
    websocket: WebSocket,
    user_id: str = Query(..., description="The user's ID"),
    store_id: str = Query(..., description="The store's ID")
):
    await websocket.accept()

    settings = get_settings()
    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        await websocket.close(code=1008)
        return

    # 1. Pre-load Memory and Store Profile Context in Parallel
    LOGGER.info("voice_loading_context", user_id=user_id, store_id=store_id)
    bootstrap_results = await asyncio.gather(
        _bootstrap_voice_memory_context(user_id, store_id),
        _bootstrap_store_details_context(user_id, store_id),
        return_exceptions=True
    )
    
    mem_res, store_context = bootstrap_results[0], bootstrap_results[1]
    if isinstance(mem_res, Exception) or not isinstance(mem_res, tuple):
        user_memories, store_memories = [], []
    else:
        user_memories, store_memories = mem_res[0], mem_res[1]

    if isinstance(store_context, Exception):
        store_context = f"STORE PROFILE: Store ID: {store_id}"

    if user_memories or store_memories:
        memory_context = (
            "Here are your long-term memories about the user and their store:\n\n"
            "USER MEMORIES:\n" + "\n".join(m.get("content", "") for m in user_memories) + "\n\n"
            "STORE MEMORIES:\n" + "\n".join(m.get("content", "") for m in store_memories)
        )
    else:
        memory_context = (
            "No long-term memory is currently available for this user/store. "
            "Proceed with a fresh voice conversation using the current context only."
        )

    system_prompt = build_voice_system_prompt(user_id, store_id, memory_context, store_context)

    gemini_ws = None
    try:
        # 2. Connect via raw websockets using the valid Gemini Live models.
        gemini_ws = await _connect_gemini_live(api_key, system_prompt)
        LOGGER.info("setup_complete")
        # Signal the frontend that Gemini is ready
        await websocket.send_json({"type": "ready"})

        # Session scope transcript & image caches
        last_uploaded_image = {"b64": None, "mime": None}
        session_transcript: List[Dict[str, str]] = []
        current_user_speech: List[str] = []
        current_ai_speech: List[str] = []
        user_audio_received = False

        # Send client audio / text / image control frames to Gemini
        async def receive_from_client():
            nonlocal user_audio_received
            audio_chunk_count = 0
            activity_started = False
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        raise WebSocketDisconnect

                    if message.get("bytes") is not None:
                        user_audio_received = True
                        if not activity_started:
                            await gemini_ws.send(json.dumps(_build_gemini_activity_start_message()))
                            activity_started = True
                            LOGGER.info("voice_activity_start_forwarded_to_gemini")
                        # Forward raw 16 kHz PCM using Gemini Live's current audio shape.
                        chunk = _build_gemini_audio_message(message["bytes"])
                        await gemini_ws.send(json.dumps(chunk))
                        audio_chunk_count += 1
                        if audio_chunk_count == 1 or audio_chunk_count % 25 == 0:
                            LOGGER.info(
                                "voice_audio_forwarded_to_gemini",
                                chunks=audio_chunk_count,
                                bytes=len(message["bytes"]),
                                pcm_rms=round(_pcm_rms(message["bytes"]), 2),
                            )
                        continue

                    if message.get("text"):
                        control = json.loads(message["text"])
                        c_type = control.get("type")

                        if c_type == "audio_stream_end":
                            if activity_started:
                                await gemini_ws.send(json.dumps(_build_gemini_audio_stream_end_message()))
                                activity_started = False
                            LOGGER.info(
                                "voice_activity_end_forwarded_to_gemini",
                                chunks=audio_chunk_count,
                            )
                            await websocket.send_json({
                                "type": "audio_received",
                                "chunks": audio_chunk_count,
                            })

                        elif c_type == "user_text" and control.get("text"):
                            user_prompt = control["text"]
                            session_transcript.append({"role": "user", "content": user_prompt})
                            LOGGER.info("voice_text_prompt_received", text=user_prompt)
                            turn_msg = {
                                "clientContent": {
                                    "turns": [
                                        {
                                            "role": "user",
                                            "parts": [{"text": user_prompt}]
                                        }
                                    ],
                                    "turnComplete": True
                                }
                            }
                            await gemini_ws.send(json.dumps(turn_msg))

                        elif c_type == "image_input" and control.get("data"):
                            b64_data = control["data"]
                            mime_type = control.get("mime_type", "image/jpeg")
                            caption = (control.get("text") or "").strip()

                            # Save to session cache for Vision OCR tool fallback
                            last_uploaded_image["b64"] = b64_data
                            last_uploaded_image["mime"] = mime_type

                            prompt_caption = caption if caption else "I have uploaded an image or paper bill photo. Please analyze it carefully."
                            session_transcript.append({"role": "user", "content": f"[Uploaded Image] {prompt_caption}"})

                            if not caption:
                                caption = (
                                    "I have uploaded an image or paper bill photo. "
                                    "Please analyze it carefully. If it is a paper invoice or bill, "
                                    "use parse_supplier_invoice_image to extract products, line items, and prices, and tell me what you found."
                                )

                            LOGGER.info("voice_image_input_received", mime_type=mime_type, caption=caption)

                            # 1. Forward media chunk to Gemini Realtime input
                            img_chunk = {
                                "realtimeInput": {
                                    "mediaChunks": [
                                        {
                                            "mimeType": mime_type,
                                            "data": b64_data
                                        }
                                    ]
                                }
                            }
                            await gemini_ws.send(json.dumps(img_chunk))

                            # 2. Send complete user turn with inlineData and text prompt to force turn execution
                            turn_msg = {
                                "clientContent": {
                                    "turns": [
                                        {
                                            "role": "user",
                                            "parts": [
                                                {
                                                    "inlineData": {
                                                        "mimeType": mime_type,
                                                        "data": b64_data
                                                    }
                                                },
                                                {
                                                    "text": caption
                                                }
                                            ]
                                        }
                                    ],
                                    "turnComplete": True
                                }
                            }
                            await gemini_ws.send(json.dumps(turn_msg))
            except WebSocketDisconnect:
                LOGGER.info("client_disconnected")
            except Exception as e:
                LOGGER.error("client_receive_error", error=str(e))

        # Receive Gemini audio/tool-calls and send back to client
        async def receive_from_gemini():
            nonlocal user_audio_received
            from app.agent.tools.registry import get_tool_by_name
            try:
                while True:
                    msg = await gemini_ws.recv()
                    LOGGER.debug("gemini_frame_received", type=type(msg).__name__)
                    decoded = _decode_gemini_message(msg)

                    if decoded is None:
                        continue

                    if "binary" in decoded:
                        await websocket.send_bytes(decoded["binary"])
                        continue

                    data = decoded

                    if "serverContent" in data:
                        server_content = data["serverContent"]
                        LOGGER.debug(
                            "gemini_server_content_received",
                            turn_complete=server_content.get("turnComplete"),
                        )


                        # Handle Audio parts and text parts
                        if "outputTranscription" in server_content:
                            out_text = server_content["outputTranscription"].get("text", "")
                            if out_text:
                                if not current_ai_speech or current_ai_speech[-1] != out_text:
                                    current_ai_speech.append(out_text)
                                await websocket.send_json({
                                    "type": "ai_transcript",
                                    "text": out_text,
                                })

                        if "modelTurn" in server_content:
                            for part in server_content["modelTurn"].get("parts", []):
                                if "text" in part and part["text"]:
                                    if not current_ai_speech or current_ai_speech[-1] != part["text"]:
                                        current_ai_speech.append(part["text"])
                                    await websocket.send_json({
                                        "type": "ai_text",
                                        "text": part["text"],
                                    })
                                if "inlineData" in part:
                                    pcm_base64 = part["inlineData"].get("data", "")
                                    if pcm_base64:
                                        pcm_bytes = base64.b64decode(pcm_base64)
                                        await websocket.send_bytes(pcm_bytes)

                        if "inputTranscription" in server_content:
                            in_text = server_content["inputTranscription"].get("text", "")
                            if in_text:
                                current_user_speech.append(in_text)
                                await websocket.send_json({
                                    "type": "user_transcript",
                                    "text": in_text,
                                })

                        if server_content.get("turnComplete"):
                            if user_audio_received and not current_user_speech:
                                current_user_speech.append("Voice Input")
                                user_audio_received = False

                            if current_user_speech:
                                u_full = " ".join(current_user_speech).strip()
                                if u_full:
                                    session_transcript.append({"role": "user", "content": u_full})
                                current_user_speech.clear()

                            if current_ai_speech:
                                ai_full = " ".join(current_ai_speech).strip()
                                if ai_full:
                                    session_transcript.append({"role": "assistant", "content": ai_full})
                                current_ai_speech.clear()

                            await websocket.send_json({
                                "type": "turn_complete",
                            })

                    # Handle tool calls
                    if "toolCall" in data:
                        tool_call = data["toolCall"]
                        fn_calls = tool_call.get("functionCalls", [])

                        async def _execute_single_tool(fn: dict) -> dict:
                            fn_name = fn.get("name")
                            fn_args = fn.get("args", {})
                            fn_id = fn.get("id", fn_name)
                            LOGGER.info("voice_tool_call", tool=fn_name, args=fn_args)

                            start_label, complete_label = _get_tool_status_label(fn_name, fn_args)
                            # Notify client that tool execution has started
                            try:
                                await websocket.send_json({
                                    "type": "tool_start",
                                    "tool": fn_name,
                                    "label": start_label,
                                    "args": fn_args,
                                })
                            except Exception as ws_err:
                                LOGGER.warning("voice_ws_tool_start_send_failed", error=str(ws_err))

                            result = None
                            serialized_result = ""
                            try:
                                tool = get_tool_by_name(fn_name)
                                if tool:
                                    # Inject context that tools normally get from state
                                    fn_args.setdefault("user_id", user_id)
                                    fn_args.setdefault("store_id", store_id)

                                    # Image fallback for Vision OCR tool
                                    if fn_name == "parse_supplier_invoice_image":
                                        img_val = str(fn_args.get("image_input", "")).strip()
                                        if not img_val or not (img_val.startswith("data:") or img_val.startswith("http://") or img_val.startswith("https://") or len(img_val) > 500):
                                            if last_uploaded_image.get("b64"):
                                                fn_args["image_input"] = f"data:{last_uploaded_image['mime']};base64,{last_uploaded_image['b64']}"

                                    result = await tool.ainvoke(
                                        fn_args,
                                        config={"configurable": {"user_id": user_id, "store_id": store_id}},
                                    )

                                    # Automatically execute compiled Subgraphs if tool returned a subgraph marker
                                    if isinstance(result, dict) and "__subgraph__" in result:
                                        subgraph_name = result.get("__subgraph__")
                                        LOGGER.info("voice_executing_subgraph", subgraph=subgraph_name)
                                        subgraph_input = {
                                            "store_id": result.get("store_id") or store_id,
                                            "user_id": result.get("user_id") or user_id,
                                        }
                                        for k, v in result.items():
                                            if k != "__subgraph__":
                                                subgraph_input[k] = v

                                        if subgraph_name == "email_composer":
                                            from app.agent.subgraphs.email_composer.graph import email_composer_graph
                                            comp_res = await email_composer_graph.ainvoke(subgraph_input)
                                            result = comp_res.get("email_output", comp_res)
                                        elif subgraph_name == "document_generation":
                                            from app.agent.subgraphs.document_generation.graph import document_generation_graph
                                            doc_res = await document_generation_graph.ainvoke(subgraph_input)
                                            result = doc_res.get("file_metadata", doc_res)
                                        elif subgraph_name == "morning_briefing":
                                            from app.agent.subgraphs.morning_briefing.graph import morning_briefing_graph
                                            b_res = await morning_briefing_graph.ainvoke(subgraph_input)
                                            result = b_res.get("briefing_output", b_res)
                                        elif subgraph_name == "smart_restock":
                                            from app.agent.subgraphs.smart_restock.graph import smart_restock_graph
                                            r_res = await smart_restock_graph.ainvoke(subgraph_input)
                                            result = r_res.get("restock_output", r_res)

                                    if isinstance(result, (dict, list)):
                                        serialized_result = json.dumps(result, ensure_ascii=False)
                                    else:
                                        serialized_result = str(result)
                                else:
                                    serialized_result = f"Tool '{fn_name}' not found."
                                    result = {"error": serialized_result}
                            except Exception as exc:
                                LOGGER.error("voice_tool_error", tool=fn_name, error=str(exc), exc_info=True)
                                serialized_result = f"Error executing {fn_name}: {exc}"
                                result = {"error": serialized_result}

                            # Generate visual spec if tool produced visual-compatible analytics
                            visual_spec = None
                            try:
                                visual_spec = build_visual_from_tool(fn_name, result)
                            except Exception as v_err:
                                LOGGER.warning("voice_visual_build_failed", tool=fn_name, error=str(v_err))

                            # Generate visual spec if tool produced visual-compatible analytics
                            visual_spec = None
                            try:
                                visual_spec = build_visual_for_tool(fn_name, result)
                            except Exception as v_err:
                                LOGGER.warning("voice_visual_build_failed", tool=fn_name, error=str(v_err))

                            # Notify client that tool execution is complete
                            try:
                                tool_complete_payload = {
                                    "type": "tool_complete",
                                    "tool": fn_name,
                                    "label": complete_label,
                                    "args": fn_args,
                                    "result": result,
                                }
                                if visual_spec:
                                    tool_complete_payload["visual"] = visual_spec
                                await websocket.send_json(tool_complete_payload)
                            except Exception as ws_err:
                                LOGGER.warning("voice_ws_tool_complete_send_failed", error=str(ws_err))

                            res_payload = result if isinstance(result, dict) else {"output": serialized_result}

                            return {
                                "id": fn_id,
                                "name": fn_name,
                                "response": res_payload,
                            }

                        if fn_calls:
                            try:
                                tool_responses = await asyncio.gather(
                                    *[_execute_single_tool(fn) for fn in fn_calls]
                                )
                                # Send all tool results back to Gemini
                                response_msg = {
                                    "toolResponse": {
                                        "functionResponses": list(tool_responses)
                                    }
                                }
                                await gemini_ws.send(json.dumps(response_msg))
                            except Exception as tool_batch_err:
                                LOGGER.error("voice_tool_batch_error", error=str(tool_batch_err), exc_info=True)

            except websockets.exceptions.ConnectionClosed:
                LOGGER.info("gemini_disconnected")
            except asyncio.CancelledError:
                LOGGER.info("gemini_receive_cancelled")
            except Exception as e:
                LOGGER.error("gemini_receive_error", error=str(e), exc_info=True)

        client_task = asyncio.create_task(receive_from_client())
        gemini_task = asyncio.create_task(receive_from_gemini())

        done, pending = await asyncio.wait(
            {client_task, gemini_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        for task in done:
            exc = task.exception()
            if exc is not None:
                LOGGER.error("voice_loop_task_error", error=str(exc), exc_info=True)

    except Exception as e:
        LOGGER.error("voice_websocket_error", error=str(e), exc_info=True)
    finally:
        if gemini_ws is not None:
            try:
                await gemini_ws.close()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass

        # Flush any uncommitted speech turns
        if user_audio_received and not current_user_speech:
            current_user_speech.append("Voice Input")
        if current_user_speech:
            u_full = " ".join(current_user_speech).strip()
            if u_full:
                session_transcript.append({"role": "user", "content": u_full})
        if current_ai_speech:
            ai_full = " ".join(current_ai_speech).strip()
            if ai_full:
                session_transcript.append({"role": "assistant", "content": ai_full})

        if session_transcript:
            LOGGER.info("voice_session_ended_persisting_history", message_count=len(session_transcript))
            # Run persistence in the background so it doesn't block the socket from fully closing
            asyncio.create_task(
                _persist_voice_session_history_and_memory(user_id, store_id, list(session_transcript))
            )
