"""Lossless consolidation of validated chunk specifications; no inference or I/O."""
import copy
import json


def _key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def validate_fragment(data, schema):
    result = schema.model_validate(data).model_dump()
    strings = [result["project_title"]]
    for feature in result["features"]:
        strings.extend([feature["feature_name"], feature["target_module"], *feature["business_rules"], *feature["exceptions"]])
        strings.extend(value for function in feature["functions"] for value in (function["signature"], function["purpose"]))
        if not feature["target_module"].strip() or not any(feature[field] for field in ("business_rules", "functions", "exceptions")):
            raise ValueError("An extracted module has no identifier or requirements.")
    for category in result["cross_cutting_constraints"]:
        strings.extend([category["category_name"], *category["rules"]])
    if any(not value.strip() for value in strings):
        raise ValueError("Extraction fields and requirement entries must not be blank.")
    if not result["features"] and not any(category["rules"] for category in result["cross_cutting_constraints"]):
        raise ValueError("The extraction contains no requirements.")
    return result


def _facts(spec):
    facts = set()
    for feature in spec["features"]:
        module = feature["target_module"].strip()
        facts.add((module, "module", module))
        for field in ("business_rules", "functions", "exceptions"):
            facts.update((module, field, _key(value)) for value in feature[field])
    for category in spec["cross_cutting_constraints"]:
        facts.update((category["category_name"].strip(), "constraint", _key(rule)) for rule in category["rules"])
    return facts


def _extend_unique(target, values):
    seen = {_key(value) for value in target}
    for value in values:
        key = _key(value)
        if key not in seen:
            target.append(copy.deepcopy(value))
            seen.add(key)


def merge_extractions(extractions, schema):
    """Merge identical module/category IDs and exact duplicates, preserving every fact.

    Different function purposes or contradictory rules remain visible. This does
    not guess semantic equivalence or force a fixed number of domain modules.
    """
    validated = [validate_fragment(data, schema) for data in extractions]
    if not validated:
        raise ValueError("No chunk extractions are available.")
    modules, constraints = {}, {}
    expected = set()
    for data in validated:
        expected.update(_facts(data))
        for feature in data["features"]:
            module = feature["target_module"].strip()
            if module not in modules:
                modules[module] = {**copy.deepcopy(feature), "target_module": module,
                                   "business_rules": [], "functions": [], "exceptions": []}
            for field in ("business_rules", "functions", "exceptions"):
                _extend_unique(modules[module][field], feature[field])
        for category in data["cross_cutting_constraints"]:
            name = category["category_name"].strip()
            if name not in constraints:
                constraints[name] = {"category_name": name, "rules": []}
            _extend_unique(constraints[name]["rules"], category["rules"])
    merged = schema.model_validate({"project_title": validated[0]["project_title"],
        "features": list(modules.values()), "cross_cutting_constraints": list(constraints.values())}).model_dump()
    if not merged["features"]:
        raise ValueError("No functional modules were extracted. Review the document before generating code.")
    if _facts(merged) != expected:
        raise ValueError("Consolidation lost extracted requirements; code generation must not proceed.")
    return merged


def specification_counts(spec):
    features = spec["features"]
    return {"modules": len(features), "business_rules": sum(len(f["business_rules"]) for f in features),
        "functions": sum(len(f["functions"]) for f in features),
        "exceptions": sum(len(f["exceptions"]) for f in features),
        "constraints": sum(len(c["rules"]) for c in spec["cross_cutting_constraints"])}
