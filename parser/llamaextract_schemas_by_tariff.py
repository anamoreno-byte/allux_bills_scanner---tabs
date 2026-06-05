# parser/llamaextract_schemas_by_tariff.py
from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Common historical rows
# ------------------------------------------------------------
class CFEHistoricoPDBTRow(BaseModel):
    periodo: Optional[str] = Field(
        None,
        description="Periodo histórico del recibo PDBT, por ejemplo 'del 08 ABR 25 al 10 JUN 25'.",
    )
    kwh: Optional[float] = Field(
        None,
        description="Consumo histórico en kWh.",
    )
    importe: Optional[float] = Field(
        None,
        description="Importe histórico del periodo en MXN.",
    )
    pagos: Optional[float] = Field(
        None,
        description="Pagos realizados, si aparecen en la tabla histórica.",
    )
    pendiente_pago: Optional[float] = Field(
        None,
        description="Importe pendiente de pago, si aparece.",
    )


class CFEHistoricoGDMTHRow(BaseModel):
    periodo: Optional[str] = Field(
        None,
        description="Periodo histórico mensual, por ejemplo FEB 26, ENE 26, DIC 25.",
    )
    demanda_kw: Optional[float] = Field(
        None,
        description="Demanda histórica en kW.",
    )
    consumo_total_kwh: Optional[float] = Field(
        None,
        description="Consumo total histórico en kWh.",
    )
    factor_potencia_pct: Optional[float] = Field(
        None,
        description="Factor de potencia histórico en porcentaje.",
    )
    factor_carga_pct: Optional[float] = Field(
        None,
        description="Factor de carga histórico en porcentaje.",
    )
    precio_medio_mxn: Optional[float] = Field(
        None,
        description="Precio medio histórico en MXN.",
    )


# ------------------------------------------------------------
# Lightweight classifier schema
# ------------------------------------------------------------
class CFEBillClassifierExtract(BaseModel):
    source_utility: Optional[str] = Field(
        None,
        description="Empresa o fuente del documento. Ejemplos: CFE, IBERDROLA, UNKNOWN.",
    )
    document_type: Optional[str] = Field(
        None,
        description="Tipo de documento. Usar cfe_bill si es un recibo CFE.",
    )
    cliente_nombre: Optional[str] = Field(
        None,
        description="Nombre del cliente o razón social.",
    )
    no_servicio: Optional[str] = Field(
        None,
        description="Número de servicio CFE. Extraer solo los dígitos si es posible.",
    )
    cuenta: Optional[str] = Field(
        None,
        description="Cuenta del recibo CFE.",
    )
    tarifa: Optional[str] = Field(
        None,
        description="Tarifa eléctrica. Buscar etiqueta TARIFA. Ejemplos: PDBT, GDBT, GDMTH, GDMTO.",
    )


# ------------------------------------------------------------
# Base fields common to CFE bills
# ------------------------------------------------------------
class CFEBillCommonExtract(BaseModel):
    source_utility: Optional[str] = Field(
        None,
        description="Empresa o fuente del documento. Usar CFE si es Comisión Federal de Electricidad.",
    )
    document_type: Optional[str] = Field(
        None,
        description="Tipo de documento. Usar cfe_bill si es un aviso-recibo CFE.",
    )

    cliente_nombre: Optional[str] = Field(
        None,
        description="Nombre del cliente o razón social que aparece en el recibo.",
    )
    no_servicio: Optional[str] = Field(
        None,
        description="Número de servicio CFE. Buscar 'NO. DE SERVICIO'. Extraer solo dígitos.",
    )
    cuenta: Optional[str] = Field(
        None,
        description="Cuenta del recibo. Buscar etiqueta 'CUENTA'.",
    )
    rmu: Optional[str] = Field(
        None,
        description="RMU completo. Buscar etiqueta 'RMU'. Conservar letras, números y guiones relevantes.",
    )
    rpu: Optional[str] = Field(
        None,
        description="RPU si aparece. Si no aparece, dejar null.",
    )

    limite_pago: Optional[str] = Field(
        None,
        description="Fecha límite de pago en formato YYYY-MM-DD. Puede aparecer como 'LÍMITE DE PAGO' o 'FECHA LÍMITE DE PAGO'.",
    )
    corte_a_partir: Optional[str] = Field(
        None,
        description="Fecha de corte a partir en formato YYYY-MM-DD.",
    )
    periodo_inicio: Optional[str] = Field(
        None,
        description="Fecha inicial del PERIODO FACTURADO en formato YYYY-MM-DD.",
    )
    periodo_fin: Optional[str] = Field(
        None,
        description="Fecha final del PERIODO FACTURADO en formato YYYY-MM-DD.",
    )

    tarifa: Optional[str] = Field(
        None,
        description="Tarifa eléctrica normalizada. Ejemplos: PDBT, GDBT, GDMTH, GDMTO.",
    )
    medidor: Optional[str] = Field(
        None,
        description="Número de medidor. Puede aparecer como MEDIDOR o NO. MEDIDOR.",
    )
    multiplicador: Optional[float] = Field(
        None,
        description="Multiplicador del medidor.",
    )
    no_hilos: Optional[str] = Field(
        None,
        description="Número de hilos del servicio.",
    )

    importe_total: Optional[float] = Field(
        None,
        description="Total a pagar del recibo. Buscar TOTAL A PAGAR o Total final.",
    )
    cargo_fijo: Optional[float] = Field(
        None,
        description="Cargo fijo del desglose del importe.",
    )
    subtotal: Optional[float] = Field(
        None,
        description="Subtotal general del recibo.",
    )
    iva: Optional[float] = Field(
        None,
        description="IVA del recibo.",
    )
    total_linea: Optional[float] = Field(
        None,
        description="Total final del desglose, si aparece.",
    )


# ------------------------------------------------------------
# PDBT schema
# ------------------------------------------------------------
class CFEPDBTExtract(CFEBillCommonExtract):
    """
    Schema specialized for CFE PDBT bills.

    PDBT bills usually emphasize total energy consumption, readings,
    total amount, and simple historical consumption.
    They may not include connected load or contracted demand.
    """

    lectura_actual_kwh: Optional[float] = Field(
        None,
        description="Lectura actual de energía en kWh.",
    )
    lectura_anterior_kwh: Optional[float] = Field(
        None,
        description="Lectura anterior de energía en kWh.",
    )
    kwh_total: Optional[float] = Field(
        None,
        description="Consumo total del periodo en kWh. En la tabla de Energía (kWh), corresponde a Total periodo.",
    )

    energia: Optional[float] = Field(
        None,
        description="Importe del concepto Energía en el desglose del importe.",
    )
    fac_del_periodo: Optional[float] = Field(
        None,
        description="Facturación del periodo o Fac. del Periodo.",
    )
    siim: Optional[float] = Field(
        None,
        description="SIIM, si aparece en el desglose.",
    )
    adeudo_anterior: Optional[float] = Field(
        None,
        description="Adeudo anterior, si aparece.",
    )
    su_pago: Optional[float] = Field(
        None,
        description="Su pago, si aparece.",
    )

    tiene_consumo_historico: Optional[bool] = Field(
        None,
        description="Indica si el recibo tiene tabla de consumo histórico.",
    )
    historico_count: Optional[int] = Field(
        None,
        description="Número de filas extraídas del consumo histórico.",
    )
    historico_rows: Optional[List[CFEHistoricoPDBTRow]] = Field(
        default_factory=list,
        description="Filas de la tabla de consumo histórico PDBT.",
    )


# ------------------------------------------------------------
# GDMTH schema
# ------------------------------------------------------------
class CFEGDMTHExtract(CFEBillCommonExtract):
    """
    Schema specialized for CFE GDMTH bills.

    GDMTH bills include demand, time-of-use energy blocks,
    power factor, contracted demand and connected load.
    """

    carga_conectada_kw: Optional[float] = Field(
        None,
        description="Carga conectada en kW. Buscar 'CARGA CONECTADA kW'.",
    )
    demanda_contratada_kw: Optional[float] = Field(
        None,
        description="Demanda contratada en kW. Buscar 'DEMANDA CONTRATADA kW'.",
    )

    kwh_base: Optional[float] = Field(
        None,
        description="Consumo kWh base.",
    )
    kwh_intermedia: Optional[float] = Field(
        None,
        description="Consumo kWh intermedia.",
    )
    kwh_punta: Optional[float] = Field(
        None,
        description="Consumo kWh punta.",
    )
    kwh_total: Optional[float] = Field(
        None,
        description="Consumo total kWh del periodo. Si no aparece explícito, sumar kWh base + intermedia + punta.",
    )

    kw_base: Optional[float] = Field(
        None,
        description="Demanda kW base.",
    )
    kw_intermedia: Optional[float] = Field(
        None,
        description="Demanda kW intermedia.",
    )
    kw_punta: Optional[float] = Field(
        None,
        description="Demanda kW punta.",
    )
    kwmax: Optional[float] = Field(
        None,
        description="KWMax o demanda máxima del periodo.",
    )
    kvarh: Optional[float] = Field(
        None,
        description="kVArh del periodo.",
    )
    factor_potencia_pct: Optional[float] = Field(
        None,
        description="Factor de potencia del periodo en porcentaje.",
    )

    energia: Optional[float] = Field(
        None,
        description="Importe del concepto Energía.",
    )
    bonificacion_factor_potencia: Optional[float] = Field(
        None,
        description="Bonificación o cargo por factor de potencia. Puede ser negativo.",
    )
    facturacion_del_periodo: Optional[float] = Field(
        None,
        description="Facturación del periodo.",
    )
    adeudo_anterior: Optional[float] = Field(
        None,
        description="Adeudo anterior.",
    )

    costos_mem_total: Optional[float] = Field(
        None,
        description="Total de costos de la energía en el Mercado Eléctrico Mayorista, si aparece.",
    )

    tiene_consumo_historico: Optional[bool] = Field(
        None,
        description="Indica si el recibo tiene tabla de consumo histórico.",
    )
    historico_count: Optional[int] = Field(
        None,
        description="Número de filas extraídas de consumo histórico.",
    )
    historico_rows: Optional[List[CFEHistoricoGDMTHRow]] = Field(
        default_factory=list,
        description="Filas de la tabla de consumo histórico GDMTH.",
    )


# ------------------------------------------------------------
# GDMTO / GDBT placeholders
# ------------------------------------------------------------
class CFEGDMTOExtract(CFEGDMTHExtract):
    """
    Preliminary schema for GDMTO.

    It currently reuses the GDMTH structure because both may contain
    demand-related fields. We can specialize later when we inspect
    representative GDMTO bills.
    """
    pass


class CFEGDBTExtract(CFEBillCommonExtract):
    """
    Preliminary schema for GDBT.

    We can specialize once we inspect representative GDBT receipts.
    """
    carga_conectada_kw: Optional[float] = Field(
        None,
        description="Carga conectada en kW, if present.",
    )
    demanda_contratada_kw: Optional[float] = Field(
        None,
        description="Demanda contratada en kW, if present.",
    )
    kwh_total: Optional[float] = Field(
        None,
        description="Consumo total kWh.",
    )