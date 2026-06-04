# parser/llamaextract_schema.py
from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field


class HistoricoRow(BaseModel):
    periodo: Optional[str] = Field(
        None,
        description="Periodo histórico de consumo tal como aparece en el recibo."
    )
    kwh: Optional[float] = Field(
        None,
        description="Consumo histórico en kWh para ese periodo."
    )
    importe: Optional[float] = Field(
        None,
        description="Importe histórico asociado al periodo, si aparece."
    )
    demanda_kw: Optional[float] = Field(
        None,
        description="Demanda histórica en kW, si aparece."
    )


class CFEBillExtract(BaseModel):
    source_utility: Optional[str] = Field(
        None,
        description="Empresa o fuente del recibo. Ejemplos: CFE, IBERDROLA, UNKNOWN."
    )

    document_type: Optional[str] = Field(
        None,
        description="Tipo de documento. Ejemplos: cfe_bill, iberdrola_cfdi, unknown."
    )

    cliente_nombre: Optional[str] = Field(
        None,
        description="Nombre del cliente o razón social que aparece en el recibo."
    )

    no_servicio: Optional[str] = Field(
        None,
        description="Número de servicio CFE. Debe conservarse como texto, solo dígitos si es posible."
    )

    cuenta: Optional[str] = Field(
        None,
        description="Cuenta del recibo. Mantener letras y números."
    )

    rmu: Optional[str] = Field(
        None,
        description="Registro Móvil de Usuario, RMU, si aparece."
    )

    rpu: Optional[str] = Field(
        None,
        description="Registro Permanente de Usuario, RPU, si aparece."
    )

    limite_pago: Optional[str] = Field(
        None,
        description="Fecha límite de pago en formato ISO YYYY-MM-DD si es posible."
    )

    corte_a_partir: Optional[str] = Field(
        None,
        description="Fecha de corte a partir de, en formato ISO YYYY-MM-DD si es posible."
    )

    periodo_inicio: Optional[str] = Field(
        None,
        description="Fecha inicial del periodo facturado, en formato ISO YYYY-MM-DD."
    )

    periodo_fin: Optional[str] = Field(
        None,
        description="Fecha final del periodo facturado, en formato ISO YYYY-MM-DD."
    )

    tarifa: Optional[str] = Field(
        None,
        description="Tarifa eléctrica. Ejemplos: PDBT, GDBT, GDMTO, GDMTH."
    )

    carga_conectada_kw: Optional[float] = Field(
        None,
        description="Carga conectada en kW."
    )

    demanda_contratada_kw: Optional[float] = Field(
        None,
        description="Demanda contratada en kW."
    )

    medidor: Optional[str] = Field(
        None,
        description="Número de medidor."
    )

    multiplicador: Optional[float] = Field(
        None,
        description="Multiplicador del medidor."
    )

    no_hilos: Optional[str] = Field(
        None,
        description="Número de hilos del servicio."
    )

    lectura_actual_kwh: Optional[float] = Field(
        None,
        description="Lectura actual de energía en kWh."
    )

    lectura_anterior_kwh: Optional[float] = Field(
        None,
        description="Lectura anterior de energía en kWh."
    )

    kwh_total: Optional[float] = Field(
        None,
        description="Consumo total facturado en kWh."
    )

    importe_total: Optional[float] = Field(
        None,
        description="Total a pagar del recibo."
    )

    cargo_fijo: Optional[float] = Field(
        None,
        description="Cargo fijo del recibo."
    )

    subtotal_energia: Optional[float] = Field(
        None,
        description="Subtotal asociado al concepto de energía."
    )

    subtotal: Optional[float] = Field(
        None,
        description="Subtotal general antes de IVA u otros cargos."
    )

    iva: Optional[float] = Field(
        None,
        description="IVA del recibo."
    )

    fac_del_periodo: Optional[float] = Field(
        None,
        description="Facturación del periodo."
    )

    total_linea: Optional[float] = Field(
        None,
        description="Total de la línea final del desglose, si aparece."
    )

    tiene_consumo_historico: Optional[bool] = Field(
        None,
        description="Indica si el recibo contiene tabla o bloque de consumo histórico."
    )

    historico_count: Optional[int] = Field(
        None,
        description="Número de filas históricas extraídas."
    )

    historico_rows: Optional[List[HistoricoRow]] = Field(
        default_factory=list,
        description="Tabla de consumo histórico del recibo."
    )