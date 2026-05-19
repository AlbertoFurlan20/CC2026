""" This file defines the basic agent class that can be used to implement different agents. """

import json
import sys
import os
import re
import glob
import copy
from argparse import Namespace

import MLAgentBench.high_level_actions as high_level_actions
from MLAgentBench.schema import Action, EnhancedJSONEncoder
from MLAgentBench.LLM import complete_text, complete_text_fast

initial_prompt = """You are a helpful research assistant. You have access to the following tools:
{tools_prompt}

Research Problem: {task_description}

Always respond in this format exactly:
{format_prompt}
Observation:
```
the result of the action
```

"""

format_prompt_dict = {
    "Thought": "What you are currently doing, what actions to perform and why",
    "Action": "the action to take, should be one of the names of the tools",
    "Action Input": "the input to the action as a valid JSON string",
}


class Agent:
    """ Base class for agents. """

    def __init__(self, args, env):
        self.args = args
        self.valid_format_entires = ["Action", "Action Input"]
        self.log_dir = os.path.join(args.log_dir, "agent_log")
        self.fast_model = getattr(args, "fast_llm_name", args.llm_name)

        self.action_infos = env.action_infos
        tool_names = list(env.action_infos.keys())
        self.all_tool_names = copy.deepcopy(tool_names)
        actions_remove_from_prompt = ["Read File", "Write File", "Append File", "Retrieval from Research Log", "Append Summary to Research Log", "Python REPL", "Edit Script Segment (AI)"]
        actions_remove_from_prompt.extend(args.actions_remove_from_prompt)
        for t in actions_remove_from_prompt:
            try:
                tool_names.remove(t)
            except:
                pass
        for t in args.actions_add_to_prompt:
            try:
                tool_names.append(t)
            except:
                pass
        self.prompt_tool_names = tool_names
        high_level_actions.EDIT_SCRIPT_MODEL = args.edit_script_llm_name
        high_level_actions.EDIT_SCRIPT_MAX_TOKENS = args.edit_script_llm_max_tokens
        self.tools_prompt = self.construct_tools_prompt(tool_names, env.action_infos)

        self.initial_prompt = initial_prompt.format(
            tools_prompt=self.tools_prompt,
            tool_names=self.prompt_tool_names,
            task_description=env.research_problem,
            format_prompt="\n".join([f"{k}: {format_prompt_dict[k]}" for k in self.valid_format_entires]),
        )

        self.history_steps = []

        self.initialize_logging()

        if self.args.resume:
            list_of_files = glob.glob(os.path.join(self.args.resume, f"agent_log/agent_{self.args.resume_step}_*.json"))
            latest_file = max(list_of_files, key=os.path.getctime)
            print("Restoring agent from {}".format(latest_file))
            self.restore(latest_file)


    def run(self, env):
        env.execute(Action("Execute Script", {"script_name": "train.py"}))
        env.execute(Action("Final Answer", "Done!"))


    def initialize_logging(self):
        if os.path.exists(self.log_dir):
            print("Log dir {} already exists. Overwriting.".format(self.log_dir))
        else:
            os.makedirs(self.log_dir)

        with open(os.path.join(self.log_dir, "main_log"), "w", 1) as f:
            f.write("Enabled Tools in Prompt:" + str(self.prompt_tool_names) + "\n")
            f.write("================================Start=============================\n")

        print("Agent is up! See progress in {}".format(os.path.join(self.log_dir, "main_log")))


    def save(self, file_path):
        with open(file_path, "w") as f:
            try:
                json.dump(self.__dict__, f, indent=4, cls=EnhancedJSONEncoder)
            except:
                print("save agent state failed")
                pass


    def restore(self, file_path):
        with open(file_path, "r") as f:
            agent_state = json.load(f)
        agent_state["args"] = Namespace(**agent_state["args"])
        for key, value in agent_state.items():
            if key == "log_dir":
                continue
            if key == "action_infos":
                continue
            setattr(self, key, value)


    def summarize_observation(self, action, observation, log_file, bs=10000):
        """Summarize long observation using a sliding window of size bs."""
        blocks = [observation[i:i+bs] for i in range(0, len(observation), bs)]
        descriptions = []
        for idx, b in enumerate(blocks):
            start_line_number = bs * idx + 1
            end_line_number = bs * idx + 1 + len(b)
            prompt = f"""
{action}

The full observation is too long. Given this (partial) observation from character {start_line_number} to character {end_line_number}:
```
{b}
```
Summarize the observation concisely in this format:
[Observation]: Summarize all relevant details in the observation objectively

Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""
            completion = complete_text_fast(prompt, log_file=log_file + f"_{idx}", model=self.fast_model)
            descriptions.append(completion)
        if len(descriptions) == 1:
            completion = descriptions[0]
        else:
            descriptions = "\n\n".join(["Segment {idx}: \n\n" + s for s in descriptions])
            prompt = f"""
{action}

The full observation is too long.
Given summaries for each segments of the whole observation, summarize to get a cohesive description of the entire observation.
{descriptions}

Summarize the observation concisely in this format:
[Observation]: Summarize all relevant details in the observation objectively

Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""
            completion = complete_text_fast(prompt, log_file=log_file, model=self.fast_model)
        try:
            return completion.split("[Observation]:")[1]
        except:
            return completion


    ############# Helper Functions ################

    @staticmethod
    def construct_tool_prompt(tool_name, action_info):
        tool = action_info
        usage = ",\n            ".join([f"\"{k}\": [{v}]" for k, v in tool.usage.items()])

        tools_prompt = f"""{tool.description}
        Usage:
        ```
        Action: {tool_name}
        Action Input: {{
            {usage}
        }}
        Observation: [{tool.return_value}]
        ```
            """.strip() + "\n\n"
        return tools_prompt

    @classmethod
    def construct_tools_prompt(cls, tool_names, action_infos):
        tools_prompt = ""
        for tool_name in tool_names:
            tools_prompt += f"""- {tool_name}:
        """
            tools_prompt += cls.construct_tool_prompt(tool_name, action_infos[tool_name])
        return tools_prompt

    @staticmethod
    def sanitize_json_string(s):
        s = s.strip("```json").strip("```").strip()
        s = s.replace('\\', '\\\\')
        s = s.replace('/', '\\/')
        s = s.replace('\b', '\\b')
        s = s.replace('\f', '\\f')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return re.sub(r'"([^"]*)"', lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\"', '\\"') + '"', s)

    @classmethod
    def parse_action_input(cls, s, action_info):
        try:
            try:
                d = json.loads(s)
            except:
                s = cls.sanitize_json_string(s)
                d = json.loads(s)
            if set(d.keys()) != set(action_info.usage.keys()):
                raise Exception("Argument mismatch")
            return d
        except Exception as e:
            try:
                return cls.parse_action_input_by_matching(s, action_info)
            except:
                raise e

    @staticmethod
    def parse_action_input_by_matching(s, action_info):
        entries = list(action_info.usage.keys())
        index = s.find('{')
        s = s[index + 1:]
        index = s.rfind('}')
        s = s[:index]
        pattern = ""
        for e in entries:
            pattern += f'"{e}":([\s\S]*),\s*'
        pattern = pattern[:-4]
        result = re.search(pattern, s, re.MULTILINE)

        if result is None:
            raise Exception("Invalid Format")
        result = {e: r.strip().strip('\"') for e, r in zip(entries, result.groups())}
        return result

    @staticmethod
    def print_action(entries, valid_format_entires):
        return "".join([k + ": " + entries[k] for k in valid_format_entires])

    @staticmethod
    def parse_entries(s, entries):
        entries = [e.strip() for e in entries]
        pattern = ""
        for e in entries:
            e = e.replace("[", "\[").replace("]", "\]")
            pattern += f"{e}:([\s\S]*)"
        result = re.search(pattern, s, re.MULTILINE)
        if result is None:
            raise Exception("Invalid: " + s)

        parsed = [r for r in result.groups()]
        return {e: parsed[idx] for idx, e in enumerate(entries)}
