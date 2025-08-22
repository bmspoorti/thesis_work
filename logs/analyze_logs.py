import re
from collections import defaultdict

log_path = "logs/workflow_logs.txt"

# Metrics
fallback_count = 0
total_queries = 0
agent_latency = defaultdict(list)
agent_counts = defaultdict(int)

# Routing accuracy tracking
routing_total = 0
routing_correct = 0

# Regex patterns
agent_pattern = re.compile(r"📌 Routed Agent: (.+)")
expected_pattern = re.compile(r"🎯 Expected Agent: (.+)")
latency_pattern = re.compile(r"⏱️ Latency: ([\d.]+) seconds")
fallback_pattern = re.compile(r"🛡️ Fallback Used: (Yes|No)")

with open(log_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_agent = None
expect_agent = None

for line in lines:
    agent_match = agent_pattern.search(line)
    expected_match = expected_pattern.search(line)
    latency_match = latency_pattern.search(line)
    fallback_match = fallback_pattern.search(line)

    # Routed agent
    if agent_match:
        current_agent = agent_match.group(1).strip()
        agent_counts[current_agent] += 1
        total_queries += 1

    # Expected agent (optional annotation)
    if expected_match:
        expect_agent = expected_match.group(1).strip()
        routing_total += 1
        if current_agent and expect_agent and current_agent == expect_agent:
            routing_correct += 1

    # Latency
    if latency_match and current_agent:
        latency = float(latency_match.group(1))
        agent_latency[current_agent].append(latency)

    # Fallback check
    if fallback_match:
        if fallback_match.group(1) == "Yes":
            fallback_count += 1

# ------------------------- OUTPUT -------------------------

print("📊 Chatbot Evaluation Summary\n")
print(f"✅ Total Queries Logged: {total_queries}")
print(f"🛡️ Fallbacks Triggered: {fallback_count} ({(fallback_count / total_queries * 100):.2f}%)\n")

print("📌 Average Latency per Agent:")
for agent, latencies in agent_latency.items():
    avg = sum(latencies) / len(latencies)
    print(f"   - {agent}: {avg:.2f} seconds")

print("\n📌 Query Count per Agent:")
for agent, count in agent_counts.items():
    print(f"   - {agent}: {count} queries")

# Routing Accuracy
if routing_total > 0:
    print(f"\n🎯 Routing Accuracy: {routing_correct}/{routing_total} = {(routing_correct / routing_total * 100):.2f}%")
else:
    print("\n🎯 Routing Accuracy: No annotated entries found.")
