import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from bson import ObjectId

from app.routes.voice_routes import (
    _get_gemini_tools,
    _resolve_voice_user_role_and_permissions,
    _bootstrap_store_details_context,
)
from app.agent.prompts.voice_prompt import build_voice_system_prompt


def test_gemini_tools_filtered_for_cashier():
    """Cashier with only billing:create should never see owner tools or inventory write tools."""
    tools = _get_gemini_tools(is_owner=False, permissions=["billing:create"])
    decls = tools[0]["functionDeclarations"]
    tool_names = {d["name"] for d in decls}

    # Should have billing & base tools
    assert "tool_add_billing_item" in tool_names
    assert "tool_generate_bill" in tool_names
    assert "search_products" in tool_names
    assert "get_product_details" in tool_names

    # Must NOT have owner-only tools
    assert "get_profit_loss_report" not in tool_names
    assert "get_profit_margin_analysis" not in tool_names
    assert "tool_create_automation" not in tool_names
    assert "send_store_email" not in tool_names
    assert "compare_stores" not in tool_names

    # Must NOT have inventory or sales reporting tools
    assert "tool_create_product" not in tool_names
    assert "tool_adjust_stock" not in tool_names
    assert "get_sales_summary" not in tool_names


def test_gemini_tools_filtered_for_stock_manager():
    """Stock manager with inventory permissions should see inventory tools but not owner or billing tools."""
    tools = _get_gemini_tools(is_owner=False, permissions=["inventory:add", "inventory:manage"])
    decls = tools[0]["functionDeclarations"]
    tool_names = {d["name"] for d in decls}

    assert "get_inventory_summary" in tool_names
    assert "get_low_stock_products" in tool_names
    assert "tool_adjust_stock" in tool_names
    assert "tool_create_product" in tool_names

    # Must NOT have owner-only tools
    assert "get_profit_loss_report" not in tool_names
    assert "tool_create_automation" not in tool_names
    assert "send_store_email" not in tool_names

    # Must NOT have billing tools
    assert "tool_add_billing_item" not in tool_names
    assert "tool_generate_bill" not in tool_names


def test_gemini_tools_unrestricted_for_store_owner():
    """Store owner must receive full toolset including all owner-only and analytical tools."""
    tools = _get_gemini_tools(is_owner=True, permissions=["*"])
    decls = tools[0]["functionDeclarations"]
    tool_names = {d["name"] for d in decls}

    assert "get_profit_loss_report" in tool_names
    assert "get_sales_summary" in tool_names
    assert "tool_create_automation" in tool_names
    assert "send_store_email" in tool_names
    assert "tool_add_billing_item" in tool_names
    assert "tool_create_product" in tool_names


def test_voice_prompt_includes_rbac_boundaries_for_employee():
    """Employee prompt must clearly state role limits and refusal instructions for confidential data."""
    prompt = build_voice_system_prompt(
        user_id="emp-123",
        store_id="store-456",
        memory_context="No memories",
        store_context="",
        user_role="Cashier",
        is_owner=False,
        permissions=["billing:create"],
        user_name="Aman Sharma",
    )

    assert "CRITICAL ROLE-BASED ACCESS CONTROL (RBAC)" in prompt
    assert "Assigned Role: Cashier" in prompt
    assert "Ownership Status: EMPLOYEE (NOT THE STORE OWNER)" in prompt
    assert "REFUSAL PROTOCOL FOR UNAUTHORIZED INQUIRIES" in prompt
    assert "Maaf kijiye Aman Sharma, yeh financial information sirf store owner ke liye confidential hai" in prompt
    assert "billing:create" in prompt


def test_voice_prompt_for_owner_has_full_privileges():
    """Owner prompt should give full access without refusal restrictions."""
    prompt = build_voice_system_prompt(
        user_id="owner-123",
        store_id="store-456",
        memory_context="Owner notes",
        store_context="",
        user_role="Owner",
        is_owner=True,
        permissions=["*"],
        user_name="Rajesh Gupta",
    )

    assert "USER IDENTITY & ACCESS LEVEL: STORE OWNER (FULL PRIVILEGES)" in prompt
    assert "Rajesh Gupta (Store Owner & Proprietor)" in prompt
    assert "Access Level: UNRESTRICTED" in prompt
    assert "EMPLOYEE (NOT THE STORE OWNER)" not in prompt


@pytest.mark.asyncio
async def test_bootstrap_store_details_redacts_revenue_for_cashier():
    """Cashier without reports:view_sales should have 30-day revenue redacted and no diary logs."""
    store_oid = ObjectId()
    fake_stores = MagicMock()
    fake_stores.find_one = AsyncMock(return_value={
        "_id": store_oid,
        "name": "Gupta Kirana",
        "email": "gupta@store.com",
    })
    fake_products = MagicMock()
    fake_products.count_documents = AsyncMock(return_value=42)
    fake_sellers = MagicMock()
    fake_sellers.count_documents = AsyncMock(return_value=5)
    fake_buyers = MagicMock()
    fake_buyers.count_documents = AsyncMock(return_value=12)

    collections = {
        "stores": fake_stores,
        "products": fake_products,
        "sellers": fake_sellers,
        "buyers": fake_buyers,
    }
    fake_db = MagicMock()
    fake_db.__getitem__.side_effect = lambda k: collections.get(k, MagicMock())

    with patch("app.routes.voice_routes.get_database", return_value=fake_db), \
         patch("app.routes.voice_routes._resolve_store_id", new_callable=AsyncMock, return_value=store_oid), \
         patch("app.routes.voice_routes.fetch_sales_summary", new_callable=AsyncMock) as mock_sales:

        context = await _bootstrap_store_details_context(
            user_id="emp-1",
            store_id=str(store_oid),
            user_role="Cashier",
            is_owner=False,
            permissions=["billing:create"],
            user_name="Aman",
            owner_name="Rajesh Gupta",
        )

        assert "Aman (Role: Cashier)" in context
        assert "Store Owner / Proprietor: Rajesh Gupta" in context
        assert "[RESTRICTED]" in context
        assert "Active Catalog Products: 42 items" in context
        # Sales fetch should not even be called when unauthorized
        mock_sales.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_voice_user_role_from_mongodb():
    """Should correctly identify owner vs employee from MongoDB collections."""
    store_oid = ObjectId()
    owner_user_oid = ObjectId()
    emp_user_oid = ObjectId()
    role_oid = ObjectId()

    # Owner test
    fake_stores = MagicMock()
    fake_stores.find_one = AsyncMock(return_value={
        "_id": store_oid,
        "owner": owner_user_oid,
    })

    fake_users = MagicMock()
    fake_users.find_one = AsyncMock(return_value={
        "_id": owner_user_oid,
        "name": "Owner Boss",
    })

    fake_employees = MagicMock()
    fake_employees.find_one = AsyncMock(return_value=None)

    fake_roles = MagicMock()
    fake_roles.find_one = AsyncMock(return_value=None)

    collections = {
        "stores": fake_stores,
        "users": fake_users,
        "employees": fake_employees,
        "roles": fake_roles,
    }
    fake_db = MagicMock()
    fake_db.__getitem__.side_effect = lambda k: collections[k]

    is_owner, role, perms, user_name, owner_name = await _resolve_voice_user_role_and_permissions(
        db=fake_db,
        user_id=str(owner_user_oid),
        store_id=str(store_oid),
    )
    assert is_owner is True
    assert role == "Owner"
    assert perms == ["*"]
    assert user_name == "Owner Boss"

    # Employee test
    fake_users.find_one = AsyncMock(return_value={
        "_id": emp_user_oid,
        "name": "Staff Rahul",
    })
    fake_employees.find_one = AsyncMock(return_value={
        "store": store_oid,
        "user": emp_user_oid,
        "status": "active",
        "role": role_oid,
    })
    fake_roles.find_one = AsyncMock(return_value={
        "_id": role_oid,
        "name": "Cashier",
        "permissions": ["billing:create"],
    })

    is_owner, role, perms, user_name, owner_name = await _resolve_voice_user_role_and_permissions(
        db=fake_db,
        user_id=str(emp_user_oid),
        store_id=str(store_oid),
    )
    assert is_owner is False
    assert role == "Cashier"
    assert perms == ["billing:create"]
    assert user_name == "Staff Rahul"
