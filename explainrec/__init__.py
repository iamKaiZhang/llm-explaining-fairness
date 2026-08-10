"""LLM-explained recommendations on MovieLens 100k.

Pipeline: estimate ratings -> solve a constrained allocation LP ->
let an LLM translate natural-language queries into problem
modifications -> re-solve -> compare -> explain.
"""

__version__ = "0.1.0"
