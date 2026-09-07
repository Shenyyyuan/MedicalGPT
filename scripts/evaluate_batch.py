"""Replacement for the legacy evaluator whose question strings were corrupted.

Run: python -m scripts.evaluate_batch --input reviewed_ratings.jsonl --output reports/eval
See docs/EVALUATION_PROTOCOL.md. Raw keyword hits are not medical acceptance.
"""
import sys
from scripts.audit_alignment import main
if __name__ == '__main__':
    sys.argv.insert(1, 'evaluate')
    main()
