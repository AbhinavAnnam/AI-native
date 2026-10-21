"""Local code-module planning; preserve extracted facts instead of summarizing them."""
import copy
import json
import re


def _domain_tokens(feature):
    ignored = {"and", "the", "of", "management", "manager", "service", "module", "settings", "configuration", "feature"}
    module = feature["target_module"].removesuffix(".py").rsplit(".", 1)[-1]
    words = re.findall(r"[a-z0-9]+", (module + " " + feature.get("feature_name", "")).lower())
    return frozenset(word[:-3] + "y" if word.endswith("ies") else
        word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word
        for word in words if word not in ignored)


def _fact_key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _facts(spec, destinations):
    return {(destinations[f["target_module"]], field, _fact_key(value))
        for f in spec["features"] for field in ("business_rules", "functions", "exceptions")
        for value in f.get(field, [])}


def plan_code_modules(specification):
    """Merge equal or unambiguously nested domain names; keep uncertain domains apart.

    Package namespaces must agree. Shared generic words and references in rule text
    are insufficient evidence to merge domains. Every source feature stays recorded.
    """
    features = specification.get("features", [])
    if not isinstance(features, list) or not 1 <= len(features) <= 500:
        raise ValueError("Module planning requires between one and 500 extracted features.")
    if any(not isinstance(f, dict) or not isinstance(f.get("target_module"), str) or not f["target_module"] for f in features):
        raise ValueError("Each specification feature needs a target_module.")
    names = [f["target_module"] for f in features]
    if len(set(names)) != len(names):
        raise ValueError("Each specification feature needs a unique target_module.")
    tokens = [_domain_tokens(f) for f in features]
    namespaces = [name.removesuffix(".py").rpartition(".")[0] for name in names]
    destinations = {}
    for index, feature in enumerate(features):
        parents = [j for j in range(len(features)) if tokens[index] and tokens[index] <= tokens[j]
                   and namespaces[index] == namespaces[j]]
        maximal = [j for j in parents if not any(tokens[j] < tokens[k] for k in parents)]
        if maximal and len({tokens[j] for j in maximal}) == 1:
            parent = max(maximal, key=lambda j: (len(features[j].get("functions", [])),
                         -int("settings" in names[j] or "configuration" in names[j]), -j))
        else:
            parent = index
        destinations[feature["target_module"]] = names[parent]
    groups = {}
    for name in names:
        groups.setdefault(destinations[name], []).append(name)
    return merge_module_groups(specification, groups)


def merge_module_groups(specification, groups):
    """Apply a complete partition of feature IDs; the model never rewrites facts."""
    features = specification["features"]
    names = [f["target_module"] for f in features]
    sources = [name for members in groups.values() for name in members]
    if len(set(names)) != len(names) or len(sources) != len(names) or set(sources) != set(names):
        raise ValueError("Module groups must contain every original feature exactly once.")
    for destination, members in groups.items():
        if not members or destination not in members:
            raise ValueError("Each module group needs an existing destination feature.")
        if len({name.removesuffix(".py").rpartition(".")[0] for name in members}) != 1:
            raise ValueError("Module groups cannot cross package namespaces.")
    destinations = {name: destination for destination, members in groups.items() for name in members}
    planned = copy.deepcopy(specification)
    by_name = {f["target_module"]: f for f in features}
    grouped = []
    for destination, sources in groups.items():
        merged = copy.deepcopy(by_name[destination])
        if len(sources) > 1:
            merged["feature_name"] = "; ".join(dict.fromkeys(by_name[name].get("feature_name") or name for name in sources))
            for field in ("business_rules", "functions", "exceptions"):
                values = {_fact_key(value): copy.deepcopy(value) for name in sources for value in by_name[name].get(field, [])}
                merged[field] = list(values.values())
        grouped.append(merged)
    planned["features"] = grouped
    if _facts(specification, destinations) != _facts(planned, {name: name for name in groups}):
        raise ValueError("Module grouping lost extracted requirements; generation stopped.")
    return {"specification": planned, "groups": groups}
