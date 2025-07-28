import networkx as nx
from typing import List, Set, Tuple

# Sample relationships you can expand or automate later
PREREQUISITE_MAP = {
    "machine learning": ["python", "statistics"],
    "deep learning": ["machine learning"],
    "nlp": ["machine learning"],
    "power bi": ["data visualization", "excel"],
    "sql": ["databases"],
    "data warehousing": ["sql"],
    "etl": ["data warehousing"]
}


def normalize(skill: str) -> str:
    return skill.lower().strip()


def extract_unique_skills(documents: List[dict], field: str = "skills") -> Set[str]:
    skills = set()
    for doc in documents:
        if field in doc and isinstance(doc[field], list):
            for skill in doc[field]:
                skills.add(normalize(skill))
    return skills


def build_skill_graph(curriculum_skills: Set[str], job_skills: Set[str]) -> nx.DiGraph:
    G = nx.DiGraph()

    # Add all unique skills
    all_skills = curriculum_skills.union(job_skills)
    for skill in all_skills:
        G.add_node(skill, in_curriculum=skill in curriculum_skills, in_job=skill in job_skills)

    # Add prerequisite edges
    for skill, prerequisites in PREREQUISITE_MAP.items():
        for pre in prerequisites:
            if skill in G.nodes and pre in G.nodes:
                G.add_edge(pre, skill, relation="prerequisite")

    return G


def analyze_skill_gap(G: nx.DiGraph) -> Tuple[Set[str], List[Tuple[str, str]]]:
    missing_skills = set()
    dependency_paths = []

    for node, data in G.nodes(data=True):
        if data.get("in_job") and not data.get("in_curriculum"):
            missing_skills.add(node)

            # Get related curriculum skills via incoming edges (prereqs)
            for pred in G.predecessors(node):
                if G.nodes[pred].get("in_curriculum"):
                    dependency_paths.append((pred, node))

    return missing_skills, dependency_paths


def explain_gap(missing_skills: Set[str], paths: List[Tuple[str, str]]) -> str:
    lines = []
    if not missing_skills:
        return "✅ No skill gaps detected between curriculum and job listings."

    lines.append(f"❌ Missing Skills Required by Jobs:\n- " + "\n- ".join(sorted(missing_skills)))

    if paths:
        lines.append("\n🔁 Curriculum Covers These Related Skills:")
        for pre, missing in paths:
            lines.append(f"  ✅ {pre} → ❌ {missing} (prerequisite or related)")

    return "\n".join(lines)
