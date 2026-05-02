"""
Vault Research Bridge — pulls trading knowledge from Obsidian vault into agent memory.
Reads concepts/ files and makes them searchable by trading agents.
"""
import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

VAULT_PATH = os.path.expanduser("~/Documents/Obsidian Vault")
CONCEPTS_DIR = os.path.join(VAULT_PATH, "second-brain", "concepts")

TRADING_FILES = [
    "trading-strategies-master.md",
    "trading-strategies-addendum-1.md",
    "trading-strategies-addendum-2.md",
    "deep-research-trading-2026.md",
]

class VaultResearchBridge:
    def __init__(self):
        self.knowledge = {}
        self._load()

    def _load(self):
        for fname in TRADING_FILES:
            path = os.path.join(CONCEPTS_DIR, fname)
            if os.path.exists(path):
                with open(path) as f:
                    text = f.read()
                sections = self._parse_sections(text)
                self.knowledge[fname] = {
                    "text": text,
                    "sections": sections,
                    "size": len(text),
                }

    def _parse_sections(self, text):
        sections = {}
        current = None
        for line in text.split("\n"):
            if line.startswith("## "):
                current = line[3:].strip()
                sections[current] = []
            elif current and line.strip():
                sections[current].append(line)
        return {k: " ".join(v) for k, v in sections.items()}

    def search(self, query, max_results=3):
        results = []
        for fname, data in self.knowledge.items():
            if query.lower() in data["text"].lower():
                # Find relevant sentence
                for line in data["text"].split("\n"):
                    if query.lower() in line.lower() and line.strip():
                        results.append({"file": fname, "match": line.strip(), "context": data["text"][:200]})
                        if len(results) >= max_results:
                            break
            if len(results) >= max_results:
                break
        return results

    def get_concepts_for_agent(self, agent_type):
        map = {
            "mean_reversion": ["rsi", "oversold", "bollinger", "mean rever", "deviation"],
            "momentum": ["breakout", "momentum", "trend", "volume", "macd"],
            "news": ["sentiment", "catalyst", "news", "headline"],
            "volatility": ["vix", "volatility", "hedge", "spike", "contango"],
            "crypto": ["btc", "crypto", "bitcoin", "ethereum", "altcoin"],
        }
        keywords = map.get(agent_type, [])
        all_matches = []
        for kw in keywords:
            all_matches.extend(self.search(kw, max_results=1))
        return all_matches[:5]

    def inject_research_context(self, agent_name, agent_type):
        insights = self.get_concepts_for_agent(agent_type)
        if not insights:
            return "No vault research found for this agent type."
        lines = [f"## Vault Research Context for {agent_name} (#{agent_type})"]
        for i, ins in enumerate(insights, 1):
            lines.append(f"{i}. [{ins['file']}] {ins['match'][:200]}")
        return "\n".join(lines)

    def list_loaded_files(self):
        return list(self.knowledge.keys())


# Quick test
if __name__ == "__main__":
    bridge = VaultResearchBridge()
    print(f"Loaded {len(bridge.knowledge)} vault trading files")
    for name, data in bridge.knowledge.items():
        print(f"  {name}: {data['size']} chars, {len(data['sections'])} sections")

    print("\n--- Mean reversion concepts ---")
    for r in bridge.get_concepts_for_agent("mean_reversion"):
        print(f"  {r['file']}: {r['match'][:120]}")

    print("\n--- News concepts ---")
    for r in bridge.get_concepts_for_agent("news"):
        print(f"  {r['file']}: {r['match'][:120]}")
