"""Call every LegalEase MCP tool once over stdio: a demo and a smoke test.

    python backend/scripts/mcp_client.py
    python backend/scripts/mcp_client.py --question "Qui doit s'inscrire au RNE ?"

It starts the server itself (run_mcp.py), waits for the model, and exits with 1 if a tool fails.
A real run needs the Qdrant server with the index, and GROQ_API_KEY in .env.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).resolve().parent / "run_mcp.py"


async def call(client: Client, tool: str, arguments: dict, wait_seconds: float = 0) -> dict:
    """The tool's structured result. While the model is loading, retry for up to wait_seconds."""
    deadline = anyio.current_time() + wait_seconds
    while True:
        result = await client.call_tool(tool, arguments)
        if not result.is_error:
            return result.structured_content
        message = result.content[0].text
        if "still loading" in message and anyio.current_time() < deadline:
            print("  model still loading, retrying...")
            await anyio.sleep(3)
            continue
        raise SystemExit(f"{tool} failed: {message}")


async def run(server: Path, question: str, wait_seconds: float, log) -> None:
    parameters = StdioServerParameters(command=sys.executable, args=[str(server)], env=dict(os.environ))
    async with Client(stdio_client(parameters, errlog=log), read_timeout_seconds=300) as client:
        print("tools:", ", ".join(tool.name for tool in (await client.list_tools()).tools))

        documents = (await call(client, "list_documents", {}, wait_seconds))["documents"]
        print(f"== list_documents: {len(documents)} documents")
        for doc in documents:
            print(f"  {doc['doc_id']} | {','.join(doc['languages'])} | {doc['passages']} passages")

        passages = (await call(client, "search_legal_texts", {"query": question, "k": 3}))["passages"]
        print(f"== search_legal_texts: {len(passages)} passages")
        if not passages:
            raise SystemExit("search_legal_texts returned nothing")
        for p in passages:
            print(f"  {p['doc_title']}, p. {p['page']} | score {p['score']}")

        top = passages[0]
        page = await call(client, "get_document_page", {"doc_id": top["doc_id"], "page": top["page"]})
        print(f"== get_document_page: {page['doc_title']}, p. {page['page']}: {len(page['passages'])} passage(s)")
        print("  " + page["passages"][0][:200].replace("\n", " "))

        answer = await call(client, "ask_legal_question", {"question": question})
        print(f"== ask_legal_question: {answer['status']} | confidence {answer['confidence']}")
        print("  " + answer["answer"])
        for c in answer["citations"]:
            print(f"  [{c['number']}] {c['doc_title']}, p. {c['page']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Call every LegalEase MCP tool over stdio")
    parser.add_argument("--question", default="Quelle est la durée maximale du label Startup ?")
    parser.add_argument("--server", type=Path, default=SERVER, help="server script to start")
    parser.add_argument("--wait", type=float, default=300, help="seconds to wait for the model")
    parser.add_argument("--server-log", type=Path, help="keep the server's log (stderr) in this file")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # titles are French and Arabic, also when piped on Windows
    log_path = args.server_log or Path(tempfile.gettempdir()) / "legalease-mcp-client.log"
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            anyio.run(run, args.server, args.question, args.wait, log)
        except SystemExit as exc:
            raise SystemExit(f"{exc}\nServer log: {log_path}") from None
    print("all four tools answered")


if __name__ == "__main__":
    main()
