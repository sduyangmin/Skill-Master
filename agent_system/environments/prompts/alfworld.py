# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- ALFWorld --------------------- #
ALFWORLD_STATE_CONSISTENCY_RULES = """
## State Consistency Rules

For every step, use the latest observation as the source of truth.

- Treat an object as in hand only if the latest observation or inventory clearly shows that you are holding it.
- Seeing an object is not the same as holding it.
- Treat a state change as successful only if the latest environment feedback confirms it.
- Reaching a fridge, sink, lamp, table, drawer, or other receptacle does not mean you have used it successfully.
- Before cooling, heating, cleaning, examining, or placing an object, check that the required preconditions are satisfied in the latest observation.
- If the latest observation contradicts your previous plan, update the plan immediately.
- Do not say the task is complete unless the latest observation confirms the goal is satisfied.

When reasoning, distinguish clearly between:
- what you currently observe,
- what you are currently holding,
- what has already been successfully done,
- what still needs to be done next.
"""

ALFWORLD_TEMPLATE_NO_HIS = (
    """
You are an expert agent operating in the ALFRED Embodied Environment.

"""
    + ALFWORLD_STATE_CONSISTENCY_RULES
    + """

Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
First think briefly inside <think>...</think> using 1-3 short sentences.
Reason only about the latest observation, what you are holding, and the single best next admissible action.
Do not mention instructions, tags, formatting, or what you are supposed to output.
Then output exactly one admissible environment action inside <action>...</action>.
"""
)

ALFWORLD_TEMPLATE = (
    """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

"""
    + ALFWORLD_STATE_CONSISTENCY_RULES
    + """

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
First think briefly inside <think>...</think> using 1-3 short sentences.
Reason only about the latest observation, what you are holding, and the single best next admissible action.
Do not mention instructions, tags, formatting, or what you are supposed to output.
Then output exactly one admissible environment action inside <action>...</action>.
"""
)

ALFWORLD_TEMPLATE_WITH_MEMORY = (
    """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

## Retrieved Relevant Experience

{retrieved_memories}

"""
    + ALFWORLD_STATE_CONSISTENCY_RULES
    + """

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
First think briefly inside <think>...</think> using 1-3 short sentences.
Reason only about the latest observation, what you are holding, and the single best next admissible action.
Do not mention instructions, tags, formatting, or what you are supposed to output.
Then output exactly one admissible environment action inside <action>...</action>.
"""
)
