"""Vigil MCP server — exposes memory-health checks as native AI-agent tools.

A minimal stdio JSON-RPC MCP server (protocol version "2024-11-05"). Reads
newline-delimited JSON-RPC requests from stdin, writes responses to stdout.
Run it as the command for an MCP stdio client (e.g. Claude Desktop / Claude
Code), or via the `vigil-mcp` console script.

Tools:
  vigil_scan   — run a full or selective health scan on a memory directory
  vigil_check  — pre-write contradiction gate for proposed new text
  vigil_health — full scan + write updated health scores into ChromaDB

Each tool returns {"content": [{"type": "text", "text": <json>}]} where <json>
is the serialized list of issues.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "vigil"

try:
    from vigil import __version__ as VIGIL_VERSION
except Exception:  # pragma: no cover - version is best-effort metadata
    VIGIL_VERSION = "0.0.0"


# --- Tool schemas ---

TOOLS = [
    {
        "name": "vigil_scan",
        "description": (
            "Run a memory-health scan over a directory of markdown memory "
            "files. Detects contradictions, duplicates, isolated entries, "
            "stale memories, orphan references, and missing provenance. "
            "Returns the found issues as JSON."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_dir": {
                    "type": "string",
                    "description": "Directory of markdown memory files to scan.",
                },
                "store": {
                    "type": "string",
                    "description": "ChromaDB store path (default: $MEMORY_STORE or <memory_dir>/.vigil/).",
                },
                "collection": {
                    "type": "string",
                    "description": "ChromaDB collection name (default: $MEMORY_COLLECTION or 'agent_memory').",
                },
                "checks": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["contradictions", "duplicates", "isolated",
                                 "stale", "orphans", "provenance"],
                    },
                    "description": "Subset of checks to run. Omit to run all.",
                },
            },
            "required": ["memory_dir"],
        },
    },
    {
        "name": "vigil_check",
        "description": (
            "Pre-write contradiction gate. Given proposed new memory text and "
            "a memory directory, returns any existing memories the new text "
            "would contradict or supersede. Call this BEFORE writing a new memory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_dir": {
                    "type": "string",
                    "description": "Directory of markdown memory files (locates the store).",
                },
                "text": {
                    "type": "string",
                    "description": "The proposed new memory text to check.",
                },
                "source": {
                    "type": "string",
                    "description": "Source file stem to exclude from comparison (optional).",
                },
                "store": {
                    "type": "string",
                    "description": "ChromaDB store path (default: $MEMORY_STORE or <memory_dir>/.vigil/).",
                },
                "collection": {
                    "type": "string",
                    "description": "ChromaDB collection name (default: $MEMORY_COLLECTION or 'agent_memory').",
                },
            },
            "required": ["memory_dir", "text"],
        },
    },
    {
        "name": "vigil_health",
        "description": (
            "Run a full health scan AND write per-file health scores into "
            "ChromaDB metadata (so downstream RAG can deprioritize unhealthy "
            "memories). Returns issues plus per-file scores and a count of "
            "records updated."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_dir": {
                    "type": "string",
                    "description": "Directory of markdown memory files to scan and score.",
                },
                "store": {
                    "type": "string",
                    "description": "ChromaDB store path (default: $MEMORY_STORE or <memory_dir>/.vigil/).",
                },
                "collection": {
                    "type": "string",
                    "description": "ChromaDB collection name (default: $MEMORY_COLLECTION or 'agent_memory').",
                },
            },
            "required": ["memory_dir"],
        },
    },
]


# --- Helpers ---

def _issue_to_dict(issue) -> dict:
    return {
        "severity": issue.severity,
        "category": issue.category,
        "message": issue.message,
        "files": issue.files,
        "details": issue.details,
    }


def _results_to_dict(results: dict) -> dict:
    return {cat: [_issue_to_dict(i) for i in issues]
            for cat, issues in results.items()}


def _resolve_store(args: dict, memory_dir: Path) -> Path:
    from vigil.indexer import default_store_dir
    store = args.get("store")
    return Path(store) if store else default_store_dir(memory_dir)


# --- Tool implementations ---

def _tool_vigil_scan(args: dict) -> dict:
    from vigil.scanner import full_scan

    memory_dir = Path(args["memory_dir"])
    store_dir = _resolve_store(args, memory_dir)
    results = full_scan(
        memory_dir=memory_dir,
        store_dir=store_dir,
        checks=args.get("checks"),
        collection_name=args.get("collection"),
    )
    total = sum(len(v) for v in results.values())
    return {"total_issues": total, "issues": _results_to_dict(results)}


def _tool_vigil_check(args: dict) -> dict:
    from vigil.scanner import pre_write_check

    memory_dir = Path(args["memory_dir"])
    store_dir = _resolve_store(args, memory_dir)
    issues = pre_write_check(
        args.get("text", ""),
        store_dir=store_dir,
        source_file=args.get("source", ""),
        collection_name=args.get("collection"),
    )
    return {
        "conflict_count": len(issues),
        "conflicts": [_issue_to_dict(i) for i in issues],
    }


def _tool_vigil_health(args: dict) -> dict:
    from vigil.scanner import (
        full_scan, compute_health_scores, update_health_scores,
    )

    memory_dir = Path(args["memory_dir"])
    store_dir = _resolve_store(args, memory_dir)
    coll = args.get("collection")

    results = full_scan(memory_dir=memory_dir, store_dir=store_dir, collection_name=coll)
    scores = compute_health_scores(results)
    n_updated = update_health_scores(scores, store_dir, collection_name=coll)

    total = sum(len(v) for v in results.values())
    return {
        "total_issues": total,
        "records_scored": n_updated,
        "scores": {k: round(v, 3) for k, v in sorted(scores.items())},
        "issues": _results_to_dict(results),
    }


TOOL_IMPLS = {
    "vigil_scan": _tool_vigil_scan,
    "vigil_check": _tool_vigil_check,
    "vigil_health": _tool_vigil_health,
}


# --- JSON-RPC dispatch ---

def _result(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_request(req: dict):
    """Dispatch a single JSON-RPC request. Returns a response dict, or None
    for notifications (which take no response)."""
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params") or {}

    # Notifications have no id and expect no response.
    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return _result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": VIGIL_VERSION},
        })

    if method == "ping":
        return _result(req_id, {})

    if method == "tools/list":
        return _result(req_id, {"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        impl = TOOL_IMPLS.get(name)
        if impl is None:
            return _error(req_id, -32602, f"Unknown tool: {name}")
        try:
            payload = impl(args)
            text = json.dumps(payload, indent=2, default=str)
            return _result(req_id, {
                "content": [{"type": "text", "text": text}],
            })
        except Exception as e:  # surface tool errors as MCP tool errors
            return _result(req_id, {
                "content": [{"type": "text", "text": f"Error running {name}: {e}"}],
                "isError": True,
            })

    if req_id is None:
        # Unknown notification — ignore.
        return None
    return _error(req_id, -32601, f"Method not found: {method}")


def serve(stdin=None, stdout=None):
    """Run the stdio serve loop over newline-delimited JSON-RPC."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = _error(None, -32700, "Parse error")
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
            continue

        resp = handle_request(req)
        if resp is not None:
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()


def main():
    serve()


if __name__ == "__main__":
    main()
