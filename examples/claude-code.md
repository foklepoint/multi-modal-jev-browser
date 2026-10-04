# Claude Code. Add the server, then set the keys in the shell Claude Code runs in.
claude mcp add mmjb --env CLOUDFLARE_API_TOKEN=your_cloudflare_token --env CLOUDFLARE_ACCOUNT_ID=your_account_id -- uvx --from git+https://github.com/foklepoint/multi-modal-jev-browser mmjb-mcp

# With no cloudflare credentials. Jev is the cheapest decision model that still reads a page.
claude mcp add mmjb --env OPENROUTER_API_KEY=your_openrouter_key -- uvx --from git+https://github.com/foklepoint/multi-modal-jev-browser mmjb-mcp

# Using your installed Google Chrome instead of Playwright's Chromium.
claude mcp add mmjb --env MMJB_CHROME=1 --env OPENROUTER_API_KEY=your_openrouter_key -- uvx --from git+https://github.com/foklepoint/multi-modal-jev-browser mmjb-mcp

# Attach to a Chrome you started with --remote-debugging-port=9222.
claude mcp add mmjb --env MMJB_CDP_URL=http://127.0.0.1:9222 --env OPENROUTER_API_KEY=your_openrouter_key -- uvx --from git+https://github.com/foklepoint/multi-modal-jev-browser mmjb-mcp

# Check the setup at any time:
#   uvx --from git+https://github.com/foklepoint/multi-modal-jev-browser mmjb doctor