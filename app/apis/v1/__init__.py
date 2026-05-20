from fastapi import APIRouter

from app.apis.v1.analysis_routers import analysis_router
from app.apis.v1.auth_routers import auth_router
from app.apis.v1.challenge_routers import challenge_router
from app.apis.v1.chatbot_routers import chatbot_router
from app.apis.v1.dashboard_routers import dashboard_router
from app.apis.v1.diet_routers import diet_router
from app.apis.v1.exam_routers import exam_router
from app.apis.v1.faq_routers import faq_router
from app.apis.v1.health_routers import health_router
from app.apis.v1.llm_log_routers import llm_log_router
from app.apis.v1.main_routers import main_router
from app.apis.v1.medication_routers import medication_router
from app.apis.v1.mypage_routers import mypage_router
from app.apis.v1.notification_routers import notification_router
from app.apis.v1.setting_routers import setting_router
from app.apis.v1.system_routers import system_router
from app.apis.v1.user_routers import user_router

v1_routers = APIRouter(prefix="/api/v1")
v1_routers.include_router(main_router)
v1_routers.include_router(system_router)
v1_routers.include_router(auth_router)
v1_routers.include_router(user_router)
v1_routers.include_router(health_router)
v1_routers.include_router(exam_router)
v1_routers.include_router(analysis_router)
v1_routers.include_router(challenge_router)
v1_routers.include_router(chatbot_router)
v1_routers.include_router(notification_router)
v1_routers.include_router(setting_router)
v1_routers.include_router(faq_router)
v1_routers.include_router(diet_router)
v1_routers.include_router(medication_router)
v1_routers.include_router(llm_log_router)
v1_routers.include_router(dashboard_router)
v1_routers.include_router(mypage_router)
