from fastapi import APIRouter

from app.controllers.health_controller import root
from app.lib.llm import get_llm_status


router = APIRouter(tags=["health"])


router.add_api_route("/", root, methods=["GET"])
router.add_api_route("/health", root, methods=["GET"])
router.add_api_route("/llm", get_llm_status, methods=["GET"])