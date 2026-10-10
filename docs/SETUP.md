# Setup: test accounts

The integrations are developed against fakes and recorded cassettes, but the evals and the
nightly run talk to real test accounts. This guide sets them up. Everything here uses free
plans; no payment method is needed anywhere.

You create the accounts and sign in yourself. Secrets never go into the repository: locally
they live in a `.env` file in the repo root (gitignored), in CI they become repository
secrets (see "CI secrets" at the end).

Sections: [Atlassian (Jira)](#atlassian-jira) · GitHub · Cohere · Mistral · Google Drive ·
Slack (follow later).

## Atlassian (Jira)

You end up with a free Jira Cloud site with a project `BENCH`, an OAuth 2.0 (3LO) app for
`bench auth login jira`, and an API token for CI.

### 1. Developer site

1. Open <https://go.atlassian.com/cloud-dev> and sign up (or sign in) with the account
   that will own the test site. A dedicated test account is cleaner than your work account.
2. Pick a site name, e.g. `bench-sandbox` → `https://bench-sandbox.atlassian.net`. If it is
   taken, any name works; it only goes into `JIRA_SITE`.
3. Choose **Jira** as the product. Skip invitations and onboarding questions.

The developer site is free and does not expire while it is used.

### 2. Project `BENCH`

1. In Jira: **Projects → Create project → Software development → Scrum**.
2. Choose **Company-managed** (not team-managed): its default workflow has exactly the
   statuses the fakes use (`To Do`, `In Progress`, `Done`, every transition allowed) and the
   issue types `Task` and `Bug`.
3. Name `Bench`, key **`BENCH`**. Create it, leave it empty; `bench sandbox seed jira`
   fills it later.

### 3. API token (CI and the stdio server)

1. Open <https://id.atlassian.com/manage-profile/security/api-tokens>, signed in as the
   site owner.
2. **Create API token** (the plain one, *not* "Create API token with scopes": scoped
   tokens only work through `api.atlassian.com`, the bench calls the site URL directly).
3. Label `integration-bench`, longest expiry offered (at most one year). Copy the token,
   it is shown only once.

### 4. OAuth 2.0 (3LO) app (interactive login)

1. Open <https://developer.atlassian.com/console/myapps/> → **Create → OAuth 2.0
   integration**, name `integration-bench`, accept the developer terms.
2. **Permissions → Jira API → Add**, then **Configure → Edit scopes** (classic scopes):
   - `read:jira-work`
   - `write:jira-work`
   - `read:jira-user`

   `offline_access` is not listed there; the bench requests it at login to get a refresh
   token.
3. **Authorization → OAuth 2.0 (3LO) → Configure**, callback URL:

   ```text
   http://localhost:8765/callback
   ```

   It must match exactly (the login command listens on this port).
4. **Settings**: copy **Client ID** and **Secret**.
5. The app stays private ("Not shared"); that is enough for your own site.

### 5. Local `.env`

Add to `.env` in the repo root:

```dotenv
JIRA_SITE=bench-sandbox          # the part before .atlassian.net
JIRA_EMAIL=you@example.com       # login e-mail of the token owner
JIRA_API_TOKEN=...
JIRA_CLIENT_ID=...
JIRA_CLIENT_SECRET=...
```

Check (with the variables exported in a POSIX shell): the API token works when this prints
your account as JSON.

```bash
curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" "https://$JIRA_SITE.atlassian.net/rest/api/3/myself"
```

Later `bench doctor` checks both the token and the OAuth app.

## CI secrets

Set later, once all accounts exist. For Jira: `JIRA_SITE`, `JIRA_EMAIL`, `JIRA_API_TOKEN`
(CI uses the API token only; refresh tokens rotate and cannot live in a static secret).
