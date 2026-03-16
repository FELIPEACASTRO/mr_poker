from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing(service_name: str = "mr_poker") -> None:
    """Configure OpenTelemetry tracing (no-op if libraries not installed)."""
    if os.getenv("POKER_OTEL_ENABLED", "false").lower() != "true":
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # Try OTLP exporter first, fall back to console
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            endpoint = os.getenv("POKER_OTEL_ENDPOINT", "http://localhost:4317")
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)

        # Auto-instrument FastAPI if available (checked at app startup)
        logger.info("OpenTelemetry tracing configured for %s", service_name)
    except ImportError:
        logger.debug("OpenTelemetry not installed, tracing disabled")
