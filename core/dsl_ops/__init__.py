# dsl_ops/__init__.py
from .ops import get_field, find_latest, sum_where, topk_by_sum, list_where

OPS = {
    "get_field": get_field,
    "find_latest": find_latest,
    "sum_where": sum_where,
    "topk_by_sum": topk_by_sum,
    "list_where": list_where,
}