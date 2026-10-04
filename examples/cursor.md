# Cursor reads .cursor/mcp.json in the project, or ~/.cursor/mcp.json for every project.
# Put this in .cursor/mcp.json:

{
  "mcpServers": {
    "mmjb": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/foklepoint/multi-modal-jev-browser",
        "mmjb-mcp"
      ],
      "env": {
        "OPENROUTER_API_KEY": "your_openrouter_key"
      }
    }
  }
}

# Codex reads ~/.codex/config.toml:

# [mcp_servers.mmjb]
# command = "uvx"
# args = ["--from", "git+https://github.com/foklepoint/multi-modal-jev-browser", "mmjb-mcp"]
# env = { OPENROUTER_API_KEY = "your_openrouter_key" }