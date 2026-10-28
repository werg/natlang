"""Exact third-party model sources the installed Transformers release does not provide.

Each subpackage holds byte-identical files from a pinned upstream revision, with their SHA-256 recorded and checked
before import. Nothing here is fetched or executed through ``trust_remote_code`` at runtime.
"""
