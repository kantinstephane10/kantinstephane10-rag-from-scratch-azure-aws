"""Validate intake settings and calculate an *illustrative* project budget.

Estimates are operator-supplied assumptions; no AI agent pretends to know hours.
"""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

PRODUCT_TYPES = {'agent', 'agentic', 'rag'}
MODEL_PROFILES = {'standard', 'deep-review', 'cost-aware'}


def _number(value, label, maximum=10000000):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f'{label} must be a number.')
    if not number.is_finite() or number < 0 or number > maximum:
        raise ValueError(f'{label} must be between 0 and {maximum}.')
    return number


def validate_intake(data):
    if not isinstance(data, dict):
        raise ValueError('Project intake must be an object.')
    kind = data.get('product_type', 'rag')
    profile = data.get('model_profile', 'standard')
    if kind not in PRODUCT_TYPES:
        raise ValueError('Unknown AI product type.')
    if profile not in MODEL_PROFILES:
        raise ValueError('Unknown GPT model profile.')
    if type(data.get('provider_consent', False)) is not bool:
        raise ValueError('Provider consent must be a boolean.')
    hours = _number(data.get('estimated_hours', 0), 'Estimated hours')
    rate = _number(data.get('hourly_rate', 0), 'Hourly rate')
    one_time = _number(data.get('one_time_expenses', 0), 'One-time expenses')
    monthly = _number(data.get('monthly_cost', 0), 'Monthly cost')
    contingency = _number(data.get('contingency_pct', 0), 'Contingency', 100)
    # Estimate covers engineering and one-time items only; recurring cost is separate.
    subtotal = (hours * rate) + one_time
    total = (subtotal * (Decimal(100) + contingency) / Decimal(100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return {'product_type': kind, 'model_profile': profile,
            'provider_consent': data.get('provider_consent', False),
            'estimated_hours': float(hours), 'hourly_rate': float(rate),
            'one_time_expenses': float(one_time), 'monthly_cost': float(monthly),
            'contingency_pct': float(contingency), 'estimated_total': float(total),
            'currency': 'USD'}
