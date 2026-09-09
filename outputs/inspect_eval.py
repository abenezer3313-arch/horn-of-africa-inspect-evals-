from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import includes

@task
def multilingual_refusal_eval():
    return Task(
        dataset=json_dataset("data/cyber_prompts.jsonl"),
        solver=generate(),
        scorer=includes()
    )
