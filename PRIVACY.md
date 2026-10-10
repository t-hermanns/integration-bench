# Privacy policy

integration-bench is a developer tool for testing and evaluating MCP servers
against sandbox accounts. It is not a hosted service.

- **What it accesses:** With the Google Drive scope `drive.file`, only files
  that this app created or that you opened with it. Equivalent sandbox-only
  access applies to the Jira, GitHub and Slack integrations.
- **Where data goes:** Everything runs on your own machine. OAuth tokens are
  stored in your operating system's keyring. No data is sent to the authors
  of this project.
- **Model providers:** When you run live evals, tool results from your
  sandbox accounts are sent to the model provider you configured (for example
  Google Gemini, Groq or Cohere) under that provider's terms. Use sandbox
  data only.
- **Revoking access:** Remove the app at
  https://myaccount.google.com/permissions and delete the stored token from
  your keyring.

Questions: open an issue at
https://github.com/t-hermanns/integration-bench/issues.
