# SYNTHETIC GraphQL responses

Hand-built, **not recorded from GitHub**: no `GITHUB_TOKEN` was available when these were written.
The shapes follow GitHub's published schema (checked against `schema.docs.graphql`), but every
value is invented. They exist so CI can exercise `mine` and the parsers without a network.

When you have a token, real responses can be captured with
`GitHubClient(record_dir=...)` and saved under `tests/fixtures/recorded/` (a separate directory, so
real and synthetic data are never mixed). Replace these synthetic files with recorded ones then.

- `repo.json`: the repository metadata query.
- `search_p1.json`: first search page: a good PR (#101), a bot PR (#102), and an empty node (a
  non-PR search hit). `hasNextPage` is true.
- `search_p2.json`: second page: a PR into a non-default branch (#103). Last page.
