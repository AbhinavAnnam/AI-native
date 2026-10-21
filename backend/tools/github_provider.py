"""Small GitHub REST adapter. Credentials and raw provider errors never leave it."""
import os

import httpx

from backend.models.repository import validate_repository_url
from backend.stages.documentation import DeliveryError


def repository_slug(url):
    validate_repository_url(url)
    return url.removeprefix("https://github.com/").removesuffix(".git")


def github_token():
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        raise DeliveryError("Set GITHUB_TOKEN (or GH_TOKEN) in .env with Contents and Pull requests write access to this repository, then restart the backend.")
    return token


class GitHubProvider:
    def __init__(self, token, *, transport=None):
        self.token = token
        self.transport = transport

    def _request(self, method, path, **kwargs):
        try:
            with httpx.Client(base_url="https://api.github.com", timeout=30, follow_redirects=False,
                              transport=self.transport, headers={"Authorization": f"Bearer {self.token}",
                              "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}) as client:
                response = client.request(method, path, **kwargs)
            if response.is_error or response.is_redirect:
                reason = {401: "Check the GitHub token.", 403: "Check repository permissions or GitHub rate limits.",
                          404: "Check repository access.", 422: "Check the branch and pull request details."}.get(
                              response.status_code, "GitHub is temporarily unavailable; retry this action.")
                raise DeliveryError(f"GitHub request failed ({response.status_code}). {reason}")
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DeliveryError("GitHub could not be reached or returned an invalid response. Saved work is preserved; retry publishing.") from exc

    def preflight(self, slug):
        self._request("GET", f"/repos/{slug}")

    def find_pull_request(self, slug, head, base):
        owner = slug.split("/")[0]
        values = self._request("GET", f"/repos/{slug}/pulls",
                               params={"state": "open", "head": f"{owner}:{head}", "base": base})
        if not isinstance(values, list):
            raise DeliveryError("GitHub returned an invalid pull request list.")
        return values[0] if values else None

    def create_pull_request(self, slug, head, base, title, body):
        return self._request("POST", f"/repos/{slug}/pulls",
                             json={"head": head, "base": base, "title": title, "body": body, "draft": True})

    @staticmethod
    def validate_pull_request(value, slug, head, base, commit):
        def field(*keys):
            item = value
            for key in keys:
                item = item.get(key) if isinstance(item, dict) else None
            return item
        if (not isinstance(value, dict) or field("draft") is not True or
            field("head", "ref") != head or field("base", "ref") != base or
            field("head", "sha") != commit or
            str(field("head", "repo", "full_name")).casefold() != slug.casefold() or
            str(field("base", "repo", "full_name")).casefold() != slug.casefold() or
            not isinstance(value.get("number"), int) or isinstance(value.get("number"), bool) or
            value["number"] <= 0 or value.get("html_url") != f"https://github.com/{slug}/pull/{value['number']}"):
            raise DeliveryError("An existing or returned PR does not match this draft branch and commit. Inspect it on GitHub before retrying.")
        return value
