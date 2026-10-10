# Setup: test accounts

Unit tests run against fakes and recorded cassettes and need no accounts. The live parts
(`bench doctor`, `bench smoke`, `bench record`, `bench eval`, the nightly run) talk to real
test accounts. This guide sets them up with free plans only; no payment method anywhere.

Secrets never go into the repository: locally they live in `.env` in the repo root
(gitignored), in CI they are repository secrets. User sign-ins (`bench auth login`) are kept
in the OS keyring, or in JSON files under `BENCH_TOKEN_DIR` if that is set.

Provider UIs change often; the paths below are the ones that worked in October 2026.

Sections: [Jira](#jira-atlassian) · [GitHub](#github) · [Google Drive](#google-drive) ·
[Slack](#slack) · [Model keys](#model-keys) · [Check and seed](#check-and-seed) ·
[CI secrets](#ci-secrets)

## Jira (Atlassian)

1. **Site:** open <https://go.atlassian.com/cloud-dev>, sign up with a dedicated test
   account, pick a site name (e.g. `bench-sandbox` → `bench-sandbox.atlassian.net`), choose
   Jira. The onboarding forces you to create a first space (Jira now calls projects
   "spaces"); pick **Scrum**, team-managed is fine.
2. **Space key:** space settings → **Details**: name `Bench`, key **`BENCH`**. Leave it
   empty; `bench sandbox seed jira` fills it.
3. **Language:** profile → language **English**, otherwise status names are localized and
   the tasks do not match.
4. **API token** (CI, seeding, recording): <https://id.atlassian.com/manage-profile/security/api-tokens>
   → **Create API token** (the plain one, *not* "with scopes": scoped tokens only work via
   `api.atlassian.com`).
5. **OAuth app** (interactive login): <https://developer.atlassian.com/console/myapps/> →
   **Create → OAuth 2.0 integration**. Access type **Resource level**. Permissions → Jira
   API → scopes `read:jira-work`, `write:jira-work`, `read:jira-user`. Authorization →
   callback `http://localhost:8765/callback`. Settings → Client ID and Secret. The app stays
   private.

```dotenv
JIRA_SITE=bench-sandbox        # the part before .atlassian.net
JIRA_EMAIL=you@example.com     # owner of the API token
JIRA_API_TOKEN=
JIRA_CLIENT_ID=
JIRA_CLIENT_SECRET=
```

## GitHub

1. **Org:** create a free organization (e.g. `bench-sandbox`) and an empty **private**
   repo `widgets`. (Creating the org once showed "access temporarily restricted"; waiting
   helped.)
2. **GitHub App:** org settings → Developer settings → GitHub Apps → **New GitHub App**.
   - Homepage: the repo URL. **Redirect URI** `http://localhost:8765/callback`. Webhook off.
   - Repository permissions: **Issues**, **Pull requests**, **Contents** read and write
     (Metadata read is added automatically).
   - Under **Credentials**: generate a client secret and a private key ("key pair", downloads
     a `.pem`). Store the `.pem` under the gitignored `.bench/`.
3. **Install** the app on the org, **only** on `widgets`. The installation ID is the number
   at the end of the URL under org settings → GitHub Apps → Configure.
4. The account that signs in with `bench auth login github` must be an org member; the
   seed uses it as issue author and assignee.

```dotenv
GITHUB_ORG=bench-sandbox
GITHUB_REPO=widgets
GITHUB_APP_ID=
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
GITHUB_INSTALLATION_ID=
GITHUB_PRIVATE_KEY_PATH=.bench/github-app.pem
```

## Google Drive

1. Use a private Google account. In <https://console.cloud.google.com> create a project
   (no billing) and enable the **Google Drive API**.
2. **APIs & Services → OAuth consent screen** → Google Auth Platform **Get started**: app
   name, support e-mail, audience **External**, contact.
3. **Data access:** only `https://www.googleapis.com/auth/drive.file` (non-sensitive, so no
   verification; the app only sees files it created).
4. **Clients → Create client → Desktop app** (no redirect URI needed). The secret is shown
   only once.
5. **Branding:** home page (repo URL), privacy policy (`PRIVACY.md` in the repo), authorized
   domain `github.com`, no logo (a logo triggers a review). Then **Audience → Publish app**
   ("In production"); in testing mode refresh tokens expire after seven days.

```dotenv
GDRIVE_CLIENT_ID=
GDRIVE_CLIENT_SECRET=
```

## Slack

1. Create a free workspace and a public channel **`#bench-sandbox`**.
2. Two apps, both via <https://api.slack.com/apps> → **Create New App → From a manifest**:
   - **`bench-ci`** (CI, seeding, recording): bot scopes `channels:read`,
     `channels:history`, `chat:write`; no token rotation, no distribution. Install it to the
     workspace, copy the **Bot User OAuth Token** (`xoxb-…`), and run `/invite @bench-ci` in
     `#bench-sandbox`.
   - **`bench-dev`** (interactive login): the same three scopes as **user** scopes, redirect
     URL `http://localhost:8765/callback`, PKCE enabled, token rotation on. Do not install
     it; `bench auth login slack` does that. Slack only allows `http://localhost` redirects
     with PKCE, and such apps get user tokens only (refresh token valid for 30 days, no
     client secret).

```yaml
# bench-ci manifest (bench-dev: user scopes, redirect_urls, pkce_enabled, rotation on)
display_information:
  name: bench-ci
features:
  bot_user:
    display_name: bench-ci
oauth_config:
  scopes:
    bot: [channels:read, channels:history, chat:write]
settings:
  token_rotation_enabled: false
```

```dotenv
SLACK_BOT_TOKEN=
SLACK_CLIENT_ID=               # bench-dev → Basic Information
```

## Model keys

| Provider | Where | Free limits (Oct 2026) |
|---|---|---|
| Cohere (Command A) | <https://dashboard.cohere.com> → API keys → trial key | 20 chat calls/min, 1,000 calls/month |
| Google Gemini | <https://aistudio.google.com> → Get API key | Flash Lite: 15 RPM, 500 requests/day |
| Groq (Qwen) | <https://console.groq.com> → API Keys | 8K tokens/min, 200K tokens/day, 1,000 requests/day |

```dotenv
COHERE_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
```

Model IDs are pinned in `bench_evals.models.providers`; `COHERE_MODEL`, `GEMINI_MODEL`,
`GROQ_MODEL` override them. The Cohere trial key has no data opt-out; only sandbox data is
sent. `bench budget` tracks the monthly Cohere calls (stops at 900).

## Check and seed

```bash
uv run bench auth login jira       # likewise github, gdrive, slack
uv run bench doctor                # every line should be ok
uv run bench sandbox seed jira     # likewise github, gdrive, slack; idempotent
uv run bench smoke jira github gdrive slack
```

Seeding expects an empty Jira space, an empty `widgets` repo, no `bench-sandbox` folder in
Drive and no messages in `#bench-sandbox`; it refuses to touch foreign content.

## CI secrets

Set them with `gh secret set NAME` (values from `.env`). CI uses the API token, the app key
and the bot token; only Drive needs a stored user refresh token.

| Secret | Value |
|---|---|
| `JIRA_SITE`, `JIRA_EMAIL`, `JIRA_API_TOKEN` | as in `.env` |
| `GH_APP_ID`, `GH_INSTALLATION_ID` | `GITHUB_APP_ID`, `GITHUB_INSTALLATION_ID` (names may not start with `GITHUB_`) |
| `GH_PRIVATE_KEY` | content of the `.pem` file |
| `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET` | as in `.env` |
| `GDRIVE_REFRESH_TOKEN` | sign in once with `BENCH_TOKEN_DIR` set and copy `refresh_token` from the `gdrive` file there (Drive refresh tokens do not rotate) |
| `SLACK_BOT_TOKEN` | as in `.env` |
| `COHERE_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY` | as in `.env` |

Workflows: `live-smoke.yml` (read-only check), `eval.yml` (manual eval run), `nightly.yml`
(Gemini regression run; the schedule only runs when the repo variable
`NIGHTLY_ENABLED=true` is set).
