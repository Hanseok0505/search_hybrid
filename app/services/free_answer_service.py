from __future__ import annotations

import re
import urllib.parse
from typing import Optional, Tuple

import httpx


class FreeAnswerService:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=6.0,
            headers={
                "User-Agent": "hybrid-search-platform/0.1 (+https://localhost)",
                "Accept": "application/json,text/plain,*/*",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def answer(self, query: str) -> dict:
        q = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1"
        try:
            r = await self._client.get(url)
            r.raise_for_status()
            data = r.json()

            answer = (data.get("AbstractText") or data.get("Answer") or "").strip()
            source = data.get("AbstractURL") or None
            if answer:
                return {
                    "provider": "duckduckgo_instant_answer",
                    "query": query,
                    "answer": answer,
                    "source_url": source,
                }

            related_answer, related_source = self._extract_related_topics(data)
            if related_answer:
                return {
                    "provider": "duckduckgo_related_topics",
                    "query": query,
                    "answer": related_answer,
                    "source_url": related_source,
                }

            html_answer, html_source = await self._duckduckgo_html_fallback(query)
            if html_answer:
                return {
                    "provider": "duckduckgo_html_fallback",
                    "query": query,
                    "answer": html_answer,
                    "source_url": html_source,
                }

            wiki_answer, wiki_source = await self._wikipedia_fallback(query)
            if wiki_answer:
                return {
                    "provider": "wikipedia_fallback",
                    "query": query,
                    "answer": wiki_answer,
                    "source_url": wiki_source,
                }

            return {
                "provider": "duckduckgo_instant_answer",
                "query": query,
                "answer": "No direct answer found.",
                "source_url": None,
            }
        except Exception:
            return {
                "provider": "duckduckgo_instant_answer",
                "query": query,
                "answer": "Free API call failed. Please try again later.",
                "source_url": None,
            }

    def _extract_related_topics(self, data: dict) -> Tuple[Optional[str], Optional[str]]:
        topics = data.get("RelatedTopics", [])
        snippets: list[str] = []
        source_url = None

        def walk(items: list) -> None:
            nonlocal source_url
            for item in items:
                if isinstance(item, dict) and "Topics" in item:
                    walk(item.get("Topics", []))
                    continue
                if not isinstance(item, dict):
                    continue
                text = (item.get("Text") or "").strip()
                if not text:
                    continue
                snippets.append(text)
                if source_url is None:
                    source_url = item.get("FirstURL")
                if len(snippets) >= 3:
                    return

        if isinstance(topics, list):
            walk(topics)
        if not snippets:
            return None, None
        return " / ".join(snippets[:3]), source_url

    async def _duckduckgo_html_fallback(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            q = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={q}"
            r = await self._client.get(url)
            r.raise_for_status()
            html = r.text
            snippets = self._extract_html_snippets(html)
            first_url = self._extract_first_result_url(html)
            if snippets:
                return " / ".join(snippets[:3]), first_url
        except Exception:
            pass
        return None, None

    @staticmethod
    def _extract_html_snippets(html: str) -> list[str]:
        out: list[str] = []
        for m in re.finditer(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, flags=re.S):
            txt = m.group(1)
            txt = re.sub(r"<[^>]+>", " ", txt)
            txt = re.sub(r"\s+", " ", txt).strip()
            if txt:
                out.append(txt)
            if len(out) >= 3:
                break
        return out

    @staticmethod
    def _extract_first_result_url(html: str) -> Optional[str]:
        m = re.search(r'<a rel="nofollow" class="result__a" href="([^"]+)"', html)
        if not m:
            return None
        href = m.group(1)
        if "uddg=" in href:
            try:
                parsed = urllib.parse.urlparse(href)
                qs = urllib.parse.parse_qs(parsed.query)
                uddg = qs.get("uddg", [None])[0]
                if uddg:
                    return urllib.parse.unquote(uddg)
            except Exception:
                pass
        return href

    async def _wikipedia_fallback(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        queries = [query]
        simplified = self._simplify_query(query)
        if simplified and simplified not in queries:
            queries.append(simplified)

        for q_raw in queries:
            for lang in ["ko", "en"]:
                try:
                    q = urllib.parse.quote(q_raw)
                    search_url = (
                        f"https://{lang}.wikipedia.org/w/api.php?"
                        f"action=opensearch&search={q}&limit=1&namespace=0&format=json"
                    )
                    r = await self._client.get(search_url)
                    r.raise_for_status()
                    arr = r.json()
                    if not isinstance(arr, list) or len(arr) < 4:
                        continue
                    titles = arr[1] if isinstance(arr[1], list) else []
                    descs = arr[2] if isinstance(arr[2], list) else []
                    links = arr[3] if isinstance(arr[3], list) else []
                    if not titles:
                        continue
                    title = titles[0]
                    desc = descs[0] if descs else ""
                    link = links[0] if links else None
                    if desc:
                        return f"{title}: {desc}", link

                    title_q = urllib.parse.quote(title)
                    summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title_q}"
                    s = await self._client.get(summary_url)
                    if s.status_code == 200:
                        js = s.json()
                        extract = (js.get("extract") or "").strip()
                        page = js.get("content_urls", {}).get("desktop", {}).get("page") or link
                        if extract:
                            return extract, page
                except Exception:
                    continue
        return None, None

    @staticmethod
    def _simplify_query(query: str) -> str:
        q = query.strip()
        for token in ["알려줘", "요약해줘", "뭐야", "내용", "해주세요", "해줘", "what is", "tell me"]:
            q = q.replace(token, " ")
        return " ".join(q.split())
