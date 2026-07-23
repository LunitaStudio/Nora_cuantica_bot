"""Dominio puro, sin dependencias de infraestructura."""

from nora_quantica.domain.engine import StateEngine
from nora_quantica.domain.models import ConversationState, MessageImpact

__all__ = ["ConversationState", "MessageImpact", "StateEngine"]

