"""Public-input-only mechanical transport. Never used for live system metrics."""

import json
import re

from ..agent.router import fixed_request_payload
from ..pipeline import FixedRequest, REQUEST_KEYS
from ..testing import FakeReply


class PublicFixtureTransport:
    is_live = False

    def chat(self, messages, **_options):
        system = messages[0]["content"]
        raw = messages[1]["content"].removeprefix("Public request: ")
        raw = raw.split(". Previous execution failed: ")[0]
        value = json.loads(raw)
        if "schema_version" in value:
            request = FixedRequest.parse({k: v for k, v in value.items() if k in REQUEST_KEYS})
        else:
            text = value["message"]
            count = re.search(r"\b(\d+)\s+(?:titles|recommendations|alternatives|options|catalog movies|choices|items|movies|suggestions)", text)
            seed = re.search(r"item(?: ID)?\s+(\d+)", text)
            genres = re.findall(r"\b(Action|Adventure|Animation|Children's|Comedy|Crime|Documentary|Drama|Fantasy|Film-Noir|Horror|Musical|Mystery|Romance|Sci-Fi|Thriller|War|Western)\b", text)
            constraints = {"k": int(count[1]) if count else 5}
            if genres:
                constraints["include_genres"] = list(dict.fromkeys(genres))
                if "forbid" in text or "exclude every" in text:
                    constraints["exclude_genres"] = constraints["include_genres"]
            if "duration" in text:
                constraints["required_fields"] = ["duration"]
            request = FixedRequest.parse({"schema_version": 1,
                "user_id": value.get("user_id"), "history_authorized": value.get("history_authorized", False),
                "liked_item_ids": [int(seed[1])] if seed else [], "constraints": constraints})
        payload = fixed_request_payload(request)
        if "InteRecAgent rebuilt" in system:
            response = {"request": payload, "plan": [
                {"tool_name": "Movie Candidates Ranking Tool", "input": '{"schema":"model_scores"}'},
                {"tool_name": "Mapping Tool", "input": str(request.constraints.k)}]}
        elif 'key "request"' in system:
            response = {"request": payload}
        else:
            response = {"plan": [{"step_id": "s1", "tool_name": "recommend", "arguments": {"request": payload}}]}
        return FakeReply(json.dumps(response, sort_keys=True, separators=(",", ":")))
