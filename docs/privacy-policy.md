# maasv Privacy Policy

**Last updated:** February 23, 2026

## Overview

maasv is a self-hosted cognition layer. All data stays on your infrastructure — maasv does not transmit, collect, or store data on any external server operated by maasv or its authors.

## What maasv stores

When you use maasv, the following data is stored **locally in your SQLite database**:

- **Memories**: Text content you explicitly store, with category, subject, source, and confidence metadata.
- **Entities**: Named entities (people, places, projects, etc.) in the knowledge graph.
- **Relationships**: Connections between entities with temporal validity.
- **Wisdom entries**: Decision logs, outcomes, and feedback scores for experiential learning.
- **Embeddings**: Vector representations of stored memories for semantic search.

## What maasv does NOT do

- **No telemetry.** maasv does not phone home, report usage statistics, or send crash reports.
- **No cloud storage.** Your database is a local SQLite file. maasv has no server infrastructure.
- **No data sharing.** maasv does not share data with third parties.
- **No account required.** There are no user accounts, sign-ups, or authentication against any maasv service.

## Third-party services

maasv connects to third-party services **only when you explicitly configure them**:

| Service | When used | What is sent |
|---------|-----------|-------------|
| **Ollama** (local) | Default embedding provider | Text for embedding (local network only) |
| **Anthropic API** | If configured as LLM provider | Text for entity extraction |
| **OpenAI API** | If configured as LLM/embed provider | Text for embedding or extraction |
| **Voyage AI** | If configured as embed provider | Text for embedding |

These connections are governed by each provider's own privacy policy. maasv does not add tracking or metadata to these requests beyond what is required by their APIs.

## MCP server

When running as an MCP server (via `maasv-mcp` or `python -m maasv.mcp_server`):

- **STDIO transport** (default): Communication happens over stdin/stdout with the local MCP client. No network connections are made by the MCP server itself.
- **HTTP transport**: Listens on the configured host/port. Requires an API token (`MAASV_AUTH_TOKEN`). All authentication uses constant-time comparison to prevent timing attacks.

## Data deletion

All maasv data is stored in a single SQLite file (default: `maasv.db`). To delete all data, delete this file. Individual memories can be deleted via the API or MCP `maasv_memory_forget` tool.

## Changes to this policy

Updates to this policy will be noted in the repository changelog. Since maasv is self-hosted and has no telemetry, policy changes cannot retroactively affect previously stored data.

## Contact

For questions about this privacy policy: [GitHub Issues](https://github.com/ascottbell/maasv/issues)
