import pytest
from fastapi.testclient import TestClient
from app.main import create_app


def test_copilot_websocket_ping_pong():
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/copilot?store_id=store_123&user_id=test_user") as ws:
        ws.send_json({"action": "ping"})
        data = ws.receive_json()
        assert data == {"event": "pong", "data": {}}


def test_copilot_websocket_navigation_fast_path():
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/copilot?store_id=store_123&user_id=test_user") as ws:
        ws.send_json({
            "action": "message",
            "message": "take me to products page",
            "store_id": "store_123",
        })

        events_received = []
        while True:
            msg = ws.receive_json()
            events_received.append(msg["event"])
            if msg["event"] == "done":
                break

        assert "token" in events_received
        assert "navigate" in events_received
        assert "done" in events_received

