# parser/llamaextract_client.py
from __future__ import annotations

from pathlib import Path
from typing import Type

from llama_cloud_services import LlamaExtract
from pydantic import BaseModel

from parser.llamaextract_schema import CFEBillExtract
from parser.llamaextract_schemas_by_tariff import (
    CFEBillClassifierExtract,
    CFEPDBTExtract,
    CFEGDBTExtract,
    CFEGDMTHExtract,
    CFEGDMTOExtract,
)


GENERAL_AGENT_NAME = "allux-cfe-bill-parser-general-v1"
CLASSIFIER_AGENT_NAME = "allux-cfe-bill-classifier-v1"
PDBT_AGENT_NAME = "allux-cfe-bill-pdbt-v1"
GDBT_AGENT_NAME = "allux-cfe-bill-gdbt-v1"
GDMTH_AGENT_NAME = "allux-cfe-bill-gdmth-v1"
GDMTO_AGENT_NAME = "allux-cfe-bill-gdmto-v1"


def _make_agent(name: str, schema: Type[BaseModel]):
    """
    Create a LlamaExtract agent.

    For now we create by name/schema. If the SDK supports get_agent reliably,
    we can later reuse existing agents instead of creating them repeatedly.
    """
    extractor = LlamaExtract()

    try:
        return extractor.get_agent(name)
    except Exception:
        return extractor.create_agent(
            name=name,
            data_schema=schema,
        )


def _extract_with_schema(
    pdf_path: Path,
    agent_name: str,
    schema: Type[BaseModel],
) -> dict:
    agent = _make_agent(agent_name, schema)
    result = agent.extract(str(pdf_path))

    data = getattr(result, "data", result)

    if hasattr(data, "model_dump"):
        return data.model_dump()

    if isinstance(data, dict):
        return data

    if hasattr(data, "dict"):
        return data.dict()

    raise TypeError(f"Unexpected LlamaExtract result type: {type(data)}")


def classify_cfe_bill(pdf_path: Path) -> dict:
    """
    Lightweight first pass to identify tariff and basic document type.
    """
    return _extract_with_schema(
        pdf_path=pdf_path,
        agent_name=CLASSIFIER_AGENT_NAME,
        schema=CFEBillClassifierExtract,
    )


def choose_schema_from_tariff(tarifa: str | None):
    """
    Select specialized schema and agent name from tariff.
    """
    if not tarifa:
        return GENERAL_AGENT_NAME, CFEBillExtract

    t = str(tarifa).strip().upper().replace(" ", "")

    if t.startswith("PDBT"):
        return PDBT_AGENT_NAME, CFEPDBTExtract

    if t.startswith("GDMTH"):
        return GDMTH_AGENT_NAME, CFEGDMTHExtract

    if t.startswith("GDMTO"):
        return GDMTO_AGENT_NAME, CFEGDMTOExtract

    if t.startswith("GDBT"):
        return GDBT_AGENT_NAME, CFEGDBTExtract

    return GENERAL_AGENT_NAME, CFEBillExtract


def extract_cfe_bill(pdf_path: Path, schema_mode: str = "auto") -> dict:
    """
    Extract one CFE bill.

    schema_mode:
      - "general": use the original general schema.
      - "auto": classify tariff first, then use specialized schema.
      - "pdbt": force PDBT schema.
      - "gdmth": force GDMTH schema.
      - "gdmto": force GDMTO schema.
      - "gdbt": force GDBT schema.
    """
    schema_mode = schema_mode.lower().strip()

    if schema_mode == "general":
        data = _extract_with_schema(
            pdf_path=pdf_path,
            agent_name=GENERAL_AGENT_NAME,
            schema=CFEBillExtract,
        )
        data["_schema_mode"] = "general"
        data["_selected_tariff"] = data.get("tarifa")
        return data

    if schema_mode == "pdbt":
        data = _extract_with_schema(
            pdf_path=pdf_path,
            agent_name=PDBT_AGENT_NAME,
            schema=CFEPDBTExtract,
        )
        data["_schema_mode"] = "pdbt"
        data["_selected_tariff"] = data.get("tarifa")
        return data

    if schema_mode == "gdmth":
        data = _extract_with_schema(
            pdf_path=pdf_path,
            agent_name=GDMTH_AGENT_NAME,
            schema=CFEGDMTHExtract,
        )
        data["_schema_mode"] = "gdmth"
        data["_selected_tariff"] = data.get("tarifa")
        return data

    if schema_mode == "gdmto":
        data = _extract_with_schema(
            pdf_path=pdf_path,
            agent_name=GDMTO_AGENT_NAME,
            schema=CFEGDMTOExtract,
        )
        data["_schema_mode"] = "gdmto"
        data["_selected_tariff"] = data.get("tarifa")
        return data

    if schema_mode == "gdbt":
        data = _extract_with_schema(
            pdf_path=pdf_path,
            agent_name=GDBT_AGENT_NAME,
            schema=CFEGDBTExtract,
        )
        data["_schema_mode"] = "gdbt"
        data["_selected_tariff"] = data.get("tarifa")
        return data

    if schema_mode != "auto":
        raise ValueError(f"Unknown schema_mode: {schema_mode}")

    # Auto mode: classify first, then choose detailed schema.
    classifier_data = classify_cfe_bill(pdf_path)
    tarifa = classifier_data.get("tarifa")

    agent_name, schema = choose_schema_from_tariff(tarifa)

    data = _extract_with_schema(
        pdf_path=pdf_path,
        agent_name=agent_name,
        schema=schema,
    )

    # Keep classifier metadata for auditability.
    data["_schema_mode"] = "auto"
    data["_classifier_tariff"] = tarifa
    data["_selected_tariff"] = data.get("tarifa") or tarifa
    data["_classifier_document_type"] = classifier_data.get("document_type")
    data["_classifier_cliente_nombre"] = classifier_data.get("cliente_nombre")
    data["_classifier_no_servicio"] = classifier_data.get("no_servicio")

    return data