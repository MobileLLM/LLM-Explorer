import logging
import collections
import copy
import logging
import random
import time
import math
import os
import requests
import json
import re
import yaml
from openai import OpenAI
import pdb
import networkx as nx

import numpy as np
import pandas as pd

from .input_event import *
from .input_policy import UtgBasedInputPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
DEBUG = True
ACTION_INEFFECTIVE = 'no effect'
DUMMY_INPUT = 'dummy_user_input'

RANDOM_EXPLORE_PROB = 0.0

MAX_NUM_STEPS_OUTSIDE = 3
# MAX_NAV_STEPS = 10
# MAX_NAV_STEPS = 20
MAX_NAV_STEPS = 9999
MAX_START_APP_RETRY = 4
MAX_NUM_STEPS_OUTSIDE_ACTIVITY_STACK = 5
ONLY_EXPLORE_IN_APP = True

MAX_NUM_DIFF_ELEMENTS_IN_SIMILAR_STATES = 2
MIN_SIZE_SAME_FUNCTION_ELEMENT_GROUP = 5
# MIN_SIZE_SAME_FUNCTION_ELEMENT_GROUP = 2
SKIP_SIMILAR_ACTION_THRESHOLD = 4
DUMP_MEMORY_NUM_STEPS = 3
MAX_EXPLORE_CURRENT_STATE_TIME = 10
MAX_EXPLORED_ACTIVITIES_NOT_INCREASE_TIME = 100
MAX_NAVIGATE_NUM_AT_ONE_TIME = 10

# EXPLORE_WITH_LLM = False
EXPLORE_WITH_LLM = True

'''below is for manual mode'''
ADDTEXT = True

Manual_mode = os.environ['manual'] == 'True'
GOBACK_element = {
                'allowed_actions': ['press'],
                'status':[],
                'desc': '<button bound_box=0,0,0,0>go back</button>',
                'event_type': 'press',
                'bound_box': '0,0,0,0',
                'class': 'android.widget.ImageView',
                'content_free_signature': 'android.widget.ImageView',
                'size': 0,
                'semantic_element_title': '<button bound_box=0,0,0,0>go back</button>'
            }
RESTART_element = {
                'allowed_actions': ['restart'],
                'status':[],
                'desc': '<button bound_box=1,1,1,1>restart</button>',
                'event_type': 'restart',
                'bound_box': '1,1,1,1',
                'class': 'android.widget.ImageView',
                'content_free_signature': 'android.widget.ImageView',
                'size': 0,
                'semantic_element_title': '<button bound_box=1,1,1,1>restart</button>'
            }
def _save2yaml(file_name, state_prompt, idx, inputs=None, action_type='touch', state_str=None, structure_str=None, tag=None, width=None, height=None):
    if not os.path.exists(file_name):
        tmp_data = {
        'step_num': 0,
        'records': []
        }
        with open(file_name, 'w', encoding='utf-8') as f:
            yaml.dump(tmp_data, f)

    with open(file_name, 'r', encoding='utf-8') as f:
        old_yaml_data = yaml.safe_load(f)
    
    new_records = old_yaml_data['records']
    new_records.append(
            {'State': state_prompt,
            'Choice': idx,
            'Action': action_type,
            'Input': inputs,
            'state_str': state_str,
            'structure_str': structure_str,
            'tag':tag,
            'width':width,
            'height':height}
        )
    data = {
        'step_num': len(list(old_yaml_data['records'])),
        'records': new_records
    }
    with open(file_name, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)
'''end for manual mode'''
class GPT:
    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.history = collections.OrderedDict()
        self.initial_time = time.time()
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_list": [],
            "completion_tokens_list": [],
            "total_tokens_list": [],
            "time_cost": [],
            "time": [],
            "prompt": [],
            "response": []
        }


    @staticmethod
    def query(prompt, usage_path, model='gpt-3.5-turbo-1106', url=os.environ['GPT_API_URL'], api_key=os.environ['GPT_API_KEY'], temperature=0.7, verbose=True):
    # def query(prompt, usage_path, model='gpt-4-1106-preview', url=os.environ['GPT_API_URL'], api_key=os.environ['GPT_API_KEY'], temperature=0.7, verbose=True):
        client = OpenAI(
            base_url=url,
            api_key=api_key
        )
        start_time = time.time()
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            timeout=15
        )

        gpt_inst.usage["prompt_tokens"] += completion.usage.prompt_tokens
        gpt_inst.usage["completion_tokens"] += completion.usage.completion_tokens
        gpt_inst.usage["total_tokens"] += completion.usage.total_tokens
        gpt_inst.usage["prompt_tokens_list"].append(completion.usage.prompt_tokens)
        gpt_inst.usage["completion_tokens_list"].append(completion.usage.completion_tokens)
        gpt_inst.usage["total_tokens_list"].append(completion.usage.total_tokens)
        gpt_inst.usage["time_cost"].append(time.time() - start_time)
        gpt_inst.usage["time"].append(time.time() - gpt_inst.initial_time)
        if verbose:
            print(f'-------- GPT query ---------\n{prompt}')
        res = completion.choices[0].message.content
        if verbose:
            print(f'-------- GPT response ---------\n{res}')
            gpt_inst.usage["prompt"].append(prompt)
            gpt_inst.usage["response"].append(res)
        # print(f'-------- GPT usage ---------\n{gpt_inst.usage}')
        gpt_path = os.path.join(usage_path, "gpt_usage.json")
        with open(gpt_path, "w", encoding='utf-8') as gpt_usage_json:
            json.dump(gpt_inst.usage, gpt_usage_json, indent=4)
        return res

gpt_inst = GPT()


class Utils:
    @staticmethod
    def get_action_type(action):
        action_type = action.event_type
        if action_type == KEY_KeyEvent:
            return KEY_KeyEvent
        if action_type == KEY_RestartAppEvent:
            return KEY_RestartAppEvent
        allowed_actions = action.view['allowed_actions']
        status = action.view['status']
        if action_type == KEY_TouchEvent and 'select' in allowed_actions:
            if 'selected' in status:
                return 'unselect'
            else:
                return 'select'
        if isinstance(action, ScrollEvent):
            return f'{action_type} {action.direction}'
        return action_type
    
    @staticmethod
    def pack_action(app, action_type, target_element, input_text):
        action_dict = {'event_type': action_type, 'view': target_element}
        if action_type == KEY_SetTextEvent:
            action_dict['text'] = input_text
        elif 'scroll' in action_type:
            action_dict['event_type'] = KEY_ScrollEvent
            action_dict['direction'] = action_type.split(' ')[-1]
        elif action_type == 'press':
            return KeyEvent(name='BACK')
        elif action_type == "restart":
            return RestartAppEvent(app=app)
        return InputEvent.from_dict(action_dict)
    
    @staticmethod
    def action_desc(action):
        action_type = action.event_type
        desc = action_type
        if action_type in [KEY_IntentEvent]:
            desc += f' {action.intent}'
        if action_type in [KEY_ScrollEvent]:
            desc += f' {action.direction}'
        if action_type in [KEY_KeyEvent]:
            desc += f' {action.name}'
        if action_type in [KEY_TouchEvent, KEY_LongTouchEvent, KEY_SelectEvent, KEY_UnselectEvent, KEY_ScrollEvent, KEY_SetTextEvent]:
            element = action.view
            view_desc = element['desc'] if 'desc' in element else f"<{element['class']}, bound_box={element['bound_box']}>"
            desc += f' {view_desc}'
        if action_type in [KEY_SetTextEvent]:
            desc += f' {action.text}'
        return desc


class Memory:
    """
        known_states[state.state_str] = state_info
        self.know_states = {
            "cfa46d": {
                state: <droidbot.device_state.DeviceState object at 0x00000175B2064D30>,
                activity: MainActivity,
                app_foreground_depth: 0,
                page_description: Note management and creation page,
                elements_description: Search, Open note, Create new note, More options, scrollbar, General note, Meeting notes, Insert text,
                elements: [{'package': 'com.simplemobiletools.notes.pro', 'visible': True, 'checkable': False, ...}, ...]
                same_function_element_groups: [],
                semantic_state_title: Note management and creation page. Elements: Search, Open note, Create new note, More options, scrollbar, General note, Meeting notes, Insert text
            },
        }
        state.foreground_activity: com.android.documentsui/.picker.PickActivity
        state.activity_stack: ['com.android.documentsui/.picker.PickActivity', 'com.simplemobiletools.notes.pro/.activities.MainActivity']

    """
    def __init__(self, utg, app, device):
        self.utg = utg
        self.app = app
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self.known_states = collections.OrderedDict()
        self.semantic_states = collections.OrderedDict()
        self.known_transitions = collections.OrderedDict()
        self.known_structures = collections.OrderedDict()
        self.action_history = pd.DataFrame()
        self.action_effects = pd.DataFrame()
        self.nav_failed_actions = []
        self.last_random_action_state = None
        self.app_document = []
        # GPT.query('hello!', verbose=True) # GPT check
    
    def to_string(self, with_similarity_info=True, with_target_info=True, with_action_effects_info=True):
        memory_str = f'## All pages of app "{self.app.app_name}":\n'
        semantic_states = self.semantic_states
        for si, semantic_state_title in enumerate(semantic_states.keys()):
            state_desc = self.get_semantic_state_desc(semantic_state_title, with_similarity_info, with_target_info)
            memory_str += f'\n{state_desc}'
        if with_action_effects_info:
            memory_str += f'\n\n## Action effects:\n{self.get_action_effects_desc()}\n'
        # print(memory_str)
        return memory_str
    
    def get_action_effects_desc(self, with_element_info=False):
        action_effects_desc = ''
        if len(self.action_effects) == 0:
            return action_effects_desc
        if with_element_info:
            action_effects_desc += self.action_effects.to_string()
        else:
            # action_effects_desc += self.action_effects[['from_page', 'to_page', 'action_type', 'elemend_id', 'element_desc', 'text']].to_string()
            action_effects_desc += self.action_effects[['from_page', 'to_page', 'action_type', 'element_id', 'element_desc', 'text']].to_string()
        return action_effects_desc

    def get_semantic_state_desc(self, semantic_state_title, with_similarity_info=False, with_target_info=True):
        semantic_states = self.semantic_states
        state_desc = f' page {list(semantic_states.keys()).index(semantic_state_title)}: {semantic_state_title}\n'
        semantic_elements = semantic_states[semantic_state_title]['semantic_elements']
        same_function_element_groups = []
        # print(semantic_elements)
        for ei, semantic_element_title in enumerate(semantic_elements.keys()):
            action_targets = semantic_elements[semantic_element_title]['action_targets']
            action_effect_info = []
            # print('action_targets', action_targets)
            if with_target_info:
                for action_type in action_targets:
                    target_state_infos = action_targets[action_type]
                    
                    '''below to add the inputted text into the memory string. You can disable this by setting ADDTEXT = False'''
                    target_state_strs, input_texts = [], []
                    for state_id, target_state_info in enumerate(target_state_infos):
                        input_text = ''
                        if isinstance(target_state_info, list):
                            target_state_strs.append(target_state_info[0])
                            input_text = target_state_info[1]
                        else:
                            target_state_strs.append(target_state_info)
                        input_texts.append(input_text)
                    '''end'''
                    
                    target_semantic_state_titles = self._get_target_semantic_states(target_state_strs)
                    action_effects = []
                    
                    input_text_id = 0
                    for target_semantic_state_title, _ in target_semantic_state_titles:
                        if target_semantic_state_title == ACTION_INEFFECTIVE:
                            action_effects.append(ACTION_INEFFECTIVE)
                            continue
                        # if target_semantic_state_title == semantic_element_title:
                        #     continue
                        target_semantic_state_id = list(semantic_states.keys()).index(target_semantic_state_title)
                        if ADDTEXT and action_type == 'set_text':
                            action_effects.append(f'on set_text(\'{input_texts[input_text_id]}\'), go to page {str(target_semantic_state_id)}')
                        else:
                            action_effects.append(f'go to page {str(target_semantic_state_id)}')
                        input_text_id += 1
                        
                    if not action_effects:
                        continue
                    
                    if ADDTEXT and action_type == 'set_text':
                        action_effect_info.append(", ".join(action_effects))
                    else:
                        action_effect_info.append(f'on {action_type}, {", ".join(action_effects)}')
                    
            if with_similarity_info:
                similar_semantic_elements = semantic_elements[semantic_element_title]['similar_semantic_elements']
                similar_ele_ids = []
                for similar_ele, count in similar_semantic_elements.items():
                    if count > 0:
                        similar_ele_ids.append(list(semantic_elements.keys()).index(similar_ele))
                if len(similar_ele_ids) > 0:
                    same_function_element_group = '{' + ','.join([str(ele_id) for ele_id in sorted(set(similar_ele_ids + [ei]))]) + '}'
                    if same_function_element_group not in same_function_element_groups:
                        same_function_element_groups.append(same_function_element_group)
                    # similar_ele_ids = ','.join([str(ele_id) for ele_id in similar_ele_ids])
                    # action_effect_info.append(f'similar to elements {similar_ele_ids}')
            action_effect_comment = f'// {"; ".join(action_effect_info)}' if action_effect_info else ''
            state_desc += f'  element {ei}: {semantic_element_title} {action_effect_comment}\n'
        if with_similarity_info:
            if len(same_function_element_groups) > 0:
                state_desc += f' same-function elements: {", ".join(same_function_element_groups)}\n'
        return state_desc
    
    def all_states(self, in_app_only=True):
        states = []
        for state_str, state_info in self.known_states.items():
            if in_app_only and state_info['app_foreground_depth'] != 0:
                continue
            states.append(state_info['state'])
        return states

    # 调用 GPT 生成一个 UI 界面 state 的语义信息 state_info，包括：
    """
        self.known_states[state.state_str] = state_info
        state_info
        {
            'state': state,
            'activity': state.activity_short_name,
            'app_foreground_depth': state.get_app_activity_depth(self.app),
            'page_description': state.structure_str,
            'elements_description': '',
            'elements': elements,
            'same_function_element_groups': same_function_element_groups
        }
        {
            'state': droidbot.device_state.DeviceState object,
            'activity': MainActivity,
            'app_foreground_depth': 0,
            'page_description': '87d6f9',
            'elements_description': '',
            'elements': [{'package': 'com.simplemobiletools.dialer', 'visible': True, ...}, {}],
            'same_function_element_groups': [(), ()]
        }
    """
    def _gen_state_semantic_info(self, state, with_llm=EXPLORE_WITH_LLM):
        state_desc, elements = state.text_representation
        state_html = state.text_representation_frame

        # 检查是否有结构相同的 state，
        # 如果有，就直接将当前 state 添加到对应的 abstract state 中
        # 直接使用其 page_description、elements_description、same_function_element_groups 就不 query gpt 了
        for state_str_i in self.known_states:
            state_i = self.known_states[state_str_i]['state']
            state_i_html = state_i.text_representation_frame
            if state_html == state_i_html:
                state_info = {
                    'state': state,
                    'activity': state.activity_short_name,
                    'app_foreground_depth': state.get_app_activity_depth(self.app),
                    'page_description': state.structure_str,
                    'elements_description': self.known_states[state_str_i]['elements_description'],
                    'elements': elements,
                    'same_function_element_groups': self.known_states[state_str_i]['same_function_element_groups']
                }
                print("############## same structure abstract state ##############")
                print(state_desc)
                print("###########################################################")
                return state_info

        if not with_llm:
            state_info = {
                'state': state,
                'activity': state.activity_short_name,
                'app_foreground_depth': state.get_app_activity_depth(self.app),
                'page_description': state.structure_str,
                'elements_description': '',
                'elements': elements,
                'same_function_element_groups': []
            }
            return state_info

        
        # 通过 query gpt 去获取界面相关信息
        prompt = f'Now suppose you are analyzing an app named "{self.app.app_name}", ' + \
        f'the current GUI page shows following elements:\n{state_desc}\n' + \
        'Please think step by step and respond in the following format:\n' + \
        ' Page description: <short (less than 20 words) description of the function of current page>\n' + \
        ' Elements description: <short (less than 20 words) summary of main control elements in current page, comma separated>\n' + \
        ' Same-function elements: <groups of element ids, each group contains multiple elements that lead to the same function or share common characteristics, ' + \
        'comma separated. Example: [0,1],[2,3,4,5,6],[7,8]> The elements with different layouts and redirect targets are less likely to have the same function.\n' + \
        '**If there are various file categories, group them together. If there are checkboxes with similar functionalities, group them together. If there are redundant date labels, group them together.**\n' + \
        'You should respond in the following format:\n' + \
        '{\n' + \
        "    \"Page description\": \"\",\n" + \
        "    \"Element description\": \"\",\n" + \
        "    \"Same-function elements\": [\n" + \
        "        {\n" + \
        "            \"elements\": [],\n" + \
        "            \"function\": \"\"\n" + \
        "        }\n" + \
        "    ]\n" + \
        "}\n"
        response = GPT.query(prompt, self.device.output_dir)
        response_json = json.loads(response)

        # Here to get the shortest path of current state
        previous_element_semantic = ""
        previous_action = ""
        state_path = nx.shortest_path(self.utg.G, source=self.utg.first_state.state_str, target=state.state_str)
        current_state = state_path[-1]
        previous_state = state_path[-2]
        edge = self.utg.G[previous_state][current_state]
        edge_event_strs = list(edge["events"].keys())
        start_state = self.utg.G.nodes[previous_state]['state']
        event = edge["events"][edge_event_strs[0]]["event"]

        event = event.to_dict()
        if "view" in event.keys():
            if 'semantic_element_title' in event['view'].keys():    
                previous_element_semantic = event['view']['semantic_element_title']
            else:
                previous_element_semantic = event['view']['desc']
            previous_action = event['event_type']

        page_description = response_json['Page description']
        elements_description =  response_json['Element description']
        same_function_elements_list =  response_json['Same-function elements']
        same_function_element_groups = []
        for i in range(len(same_function_elements_list)):
            same_function_elements = same_function_elements_list[i]
            element_ids = same_function_elements['elements']
            element_function = same_function_elements['function']
            semantic_state_title = f"{page_description}. Elements: {elements_description}"
            element_document = {
                "elements": [elements[element_id]['desc'] for element_id in element_ids],
                "semantic_state_titles": [semantic_state_title] * len(element_ids),
                "function": element_function,
                "previous_element": previous_element_semantic,
                "previous_action": previous_action
            }
            self.app_document.append(element_document)
            if len(elements) < MIN_SIZE_SAME_FUNCTION_ELEMENT_GROUP:
                continue
            same_function_element_groups.append(set(element_ids))



        # app_doc_path = os.path.join(self.device.output_dir, f'app_document_{self.app.app_name}.json')
        app_doc_path = os.path.join(self.device.output_dir, f'app_document.json')
        with open(app_doc_path, 'w', encoding='utf-8') as app_doc_file:
            json.dump(self.app_document, app_doc_file, ensure_ascii=False, indent=4)
        
        state_info = {
            'state': state,
            'activity': state.activity_short_name,
            'app_foreground_depth': state.get_app_activity_depth(self.app),
            'page_description': page_description,
            'elements_description': elements_description,
            'elements': elements,
            'same_function_element_groups': same_function_element_groups
        }
        return state_info
    
    # 通过调用 GPT，找到在 之前探索过的界面中 和 现在界面 功能一样的界面。
    # state_title: state_info["page_description"]. Elements: state_info["elements_description"]
    # state_title: 87d6f9. Elements: 
    # history_state_title: 5b5634. Elements:
    # 返回：matched_state_title, state_title
    # 如果 memory 中没有匹配的，则返回的 matched_state_title 为 Null
    def _classify_state(self, state_info, semantic_states, group_same_structure=True, filter_same_activity=True, filter_similar_elements=True, with_llm=EXPLORE_WITH_LLM):
        state_title = f'{state_info["page_description"]}. Elements: {state_info["elements_description"]}'
        history_states = {}
        history_states_desc = {}

        # 获取该 state 的界面 HTML 描述
        state_info_page_desc, _ = state_info['state'].text_representation
        state_info_page_desc_frame = state_info['state'].text_representation_frame
        for i, history_state_title in enumerate(semantic_states.keys()):
            # 获取 history_state 的界面 HTML 描述
            history_state_str_i = semantic_states[history_state_title]['states']
            history_state_i = self.known_states[history_state_str_i[0]]['state']
            history_state_page_desc, _ = history_state_i.text_representation
            history_state_page_desc_frame = history_state_i.text_representation_frame


            # 如果该 semantic_states 中的 history_state 与当前 state 的 title 相同, 就直接返回了
            if state_title == history_state_title:
                print("########## state_title == history_state_title ##########")
                print("########################################################")
                return history_state_title, state_title
            # 如果该 semantic_states 中的 history_state 与当前 state 的 page_html_desc 相同, 就直接返回了
            if state_info_page_desc == history_state_page_desc:
                print("########## state_info_page_desc == history_state_page_desc ##########")
                print("########################################################")
                return history_state_title, state_title

            # 否则需要进一步操作
            history_state_info = semantic_states[history_state_title]
            if group_same_structure:
                if state_info['state'].structure_str in history_state_info['states_structures']:
                    return history_state_title, state_title
            if filter_same_activity:
                if state_info['activity'] != history_state_info['activity']:
                    continue
            if filter_similar_elements:
                state_ele_sigs = set([e['content_free_signature'] for e in state_info['elements']])
                history_state_ele_sigs = history_state_info['element_sigs']
                different_ele_sigs = state_ele_sigs.symmetric_difference(history_state_ele_sigs)
                if len(different_ele_sigs) > MAX_NUM_DIFF_ELEMENTS_IN_SIMILAR_STATES:
                    continue
            history_state_id = list(self.semantic_states.keys()).index(history_state_title)
            history_states[history_state_id] = history_state_title
            history_states_desc[history_state_id] = f'page {history_state_id}: {history_state_title}'
            # history_states_desc[i] = self.get_semantic_state_desc(history_state_title, with_similarity_info=False, with_target_info=False)
        print("########## NO Match ##########")
        print("##############################")
        if len(history_states) == 0 or not with_llm:
            return None, state_title
        history_states_desc = '\n'.join(history_states_desc.values())
        current_state_desc, _ = state_info['state'].text_representation
        current_state_desc = f'{state_title}\n{current_state_desc}'

        state_id = None
        matched_state_title = None
        if state_id is not None and state_id in history_states:
            matched_state_title = history_states[state_id]

        return matched_state_title, state_title
    
    # 在 semantic_elements 中查找是否有和 element 相同的组件，并返回
    # return element_match_title, element_title
    # element_title: '<button bound_box=0,0,0,0>go back</button>'
    # 匹配规则：两个 element 的 element_title 只有 'go back' 可以不同，其余均相同。
    def _classify_element(self, element, semantic_elements):
        # element_title: '<button bound_box=0,0,0,0>go back</button>'
        element_title = element['desc']

        # 如果有组件描述完全相同的，就直接返回
        if element_title in semantic_elements:
            return element_title, element_title
        
        # 否则，遍历查找 element_tag 和 element_bound 均相同的组件，如果有就返回
        # element_tag: button
        # element_bound: 0,0,0,0
        element_tag = re.search(r'<(\S+) ', element_title).group(1)
        element_bound = re.search(r"bound_box=(\d+,\d+,\d+,\d+)", element_title).group(1)
        for element_i_title in semantic_elements:
            element_i_tag = re.search(r'<(\S+) ', element_i_title).group(1)
            element_i_bound = re.search(r"bound_box=(\d+,\d+,\d+,\d+)", element_i_title).group(1)
            if element_i_tag == element_tag and element_i_bound == element_bound:
                return element_i_title, element_title
        # if element_title == '<input bound_box=143,858,1036,974>bob</input>':
            # import pdb;pdb.set_trace()
        
        # 否则，返回 None 和 匹配组件的描述。
        return None, element_title

    # 在 memory 中记录传入的 state，以及该 state 上的所有 element
    # 同时检查是否有语义上相同的 state，形成该 state 的 semantic_state_title（相当于一组相同的 state 的共同的名称）
    # 同时检查是否有语义上相同的 element，形成该 element 的 semantic_element_title（相当于一组相同的 element 的共同的名称）
    # state.state_str: state 的 id，如：58676e
    # semantic_state_title: 和 state_title 一样，如 "e004cd. Elements: "
    # element_title: <button bound_box=124,1964,235,2030>Contacts</button>
    # semantic_element_title: 和 element_title 一样，没有时就是 Null
    """
        self.semantic_states[state_title] = {
            'states': [],  # state.state_str
            'states_structures': [],  # state.structure_str
            'semantic_elements': collections.OrderedDict(),
            'activity': state_info['activity'],
            'app_foreground_depth': state_info['app_foreground_depth'],
            'element_sigs': set()
        }
        self.semantic_states[state_title] = {
            'states': [58676e],
            'states_structures': [e004cd],
            'semantic_elements': collections.OrderedDict(),
            'activity': state_info['activity'],
            'app_foreground_depth': state_info['app_foreground_depth'],
            'element_sigs': {'android.widget.ImageView'}
        }
        self.semantic_states[semantic_state_title]['semantic_elements'][semantic_element_title] = {
            'elements': [(state.state_str, i)], 
            'action_targets': { action: [] }, 
            'similar_semantic_elements': { }
        }
        self.semantic_states[semantic_state_title]['semantic_elements'][semantic_element_title] = {
            'elements': [(58676e, 6), (87d6f9, 2)], 
            'action_targets': {
                'torch': [],
                'long_torch': []
            }, 
            'similar_semantic_elements': {
                '<button bound_box=124,1964,235,2030>Contacts</button>': 1,
                '<button alt='Dialpad' bound_box=871,1656,1025,1810></button>': -1,
                '<scrollbar bound_box=0,252,1080,1854></scrollbar>': -1
            }
        }
    """
    def _memorize_state(self, state):

        # 如果 known_states 中已经记录过该 state, 则直接返回在 known_states 中记录的其 state_info
        if state.state_str in self.known_states:
            return self.known_states[state.state_str]
        
        # 没有记录，则先为该 state 创建一个 state_info
        state_info = self._gen_state_semantic_info(state)


        self.known_states[state.state_str] = state_info
        semantic_state_title, state_title = self._classify_state(state_info, self.semantic_states)

        # 如果 memory 中没有匹配的界面，则需要先在 semantic_states 中为该 state 创建一条。
        if not semantic_state_title:
            semantic_state_title = state_title
            self.semantic_states[state_title] = {
                'states': [],
                'states_structures': [],
                'semantic_elements': collections.OrderedDict(),
                'activity': state_info['activity'],
                'app_foreground_depth': state_info['app_foreground_depth'],
                'element_sigs': set()
            }

        state_info['semantic_state_title'] = semantic_state_title
        self.semantic_states[semantic_state_title]['states'].append(state.state_str)
        self.semantic_states[semantic_state_title]['states_structures'].append(state.structure_str)

        # 对当前 state 中的每一个 element 做处理
        # 通过调用 _classify_element，得到其 semantic_element_title（一组相同组件的统一的一个名称）
        # 然后在 self.semantic_states[semantic_state_title]['semantic_elements'][semantic_element_title] 中记录组件的相关信息
        semantic_elements = self.semantic_states[semantic_state_title]['semantic_elements']
        idx_semantic_element_titles = []
        for i, element in enumerate(state_info['elements']):
            # element['content_free_signature']: 和 class 一样
            # element['content_free_signature']: android.widget.ImageView
            self.semantic_states[semantic_state_title]['element_sigs'].add(element['content_free_signature'])
            semantic_element_title, element_title = self._classify_element(element, semantic_elements)
            # print(element, semantic_element_title)
                    
            # 如果没有匹配的 element，就为其在 semantic_elements 中创建一个条目
            if not semantic_element_title:
                semantic_element_title = element_title
                semantic_elements[semantic_element_title] = {'elements': [], 'action_targets': {}, 'similar_semantic_elements': {}}
            
            # 在 semantic_elements[semantic_element_title] 中间记录组件的相关信息，如允许的 action
            element['semantic_element_title'] = semantic_element_title
            semantic_elements[semantic_element_title]['elements'].append((state.state_str, i))
            idx_semantic_element_titles.append(semantic_element_title)
            for action in element['allowed_actions']:
                if action not in semantic_elements[semantic_element_title]['action_targets']:
                    semantic_elements[semantic_element_title]['action_targets'][action] = []
        
        # 对任意两个 element 的 semantic_element_title 做匹配，看其是否在同一个 same_function_element_groups 中
        # 如果匹配到了，那就将：
        # semantic_elements = self.semantic_states[semantic_state_title]['semantic_elements']
        # semantic_elements[ele_i_title]['similar_semantic_elements'][ele_j_title] 设为 1，否则设为 -1
        same_function_element_groups = state_info['same_function_element_groups']
        for ele_i, ele_i_title in enumerate(idx_semantic_element_titles):
            for ele_j, ele_j_title in enumerate(idx_semantic_element_titles):
                if ele_i == ele_j:
                    continue
                ele_ij_similar = False
                for ele_group in same_function_element_groups:
                    if ele_i in ele_group and ele_j in ele_group:
                        ele_ij_similar = True
                if ele_j_title not in semantic_elements[ele_i_title]['similar_semantic_elements']:
                    semantic_elements[ele_i_title]['similar_semantic_elements'][ele_j_title] = 0
                semantic_elements[ele_i_title]['similar_semantic_elements'][ele_j_title] += (1 if ele_ij_similar else -1)
        return state_info
    
    def save_transition(self, action, from_state, to_state):
        if not from_state or not to_state:
            return
        action_record = {
            'timestamp': pd.Timestamp.now(),
            'from_state': from_state.state_str,
            'to_state': to_state.state_str,
            'action': Utils.action_desc(action)
        }
        self.action_history = pd.concat([self.action_history, pd.DataFrame([action_record])], ignore_index=True)
        if not isinstance(action, UIEvent):
            if not Manual_mode or isinstance(action, IntentEvent):
                return
            
        from_state_info = self._memorize_state(from_state)
        to_state_info = self._memorize_state(to_state)
        if not Manual_mode and action.view is None:
            return
        action_str = action.get_event_str(state=from_state)
        if action_str in self.known_transitions and self.known_transitions[action_str]['to_state'] == to_state:
            return
        if from_state_info is None:
            return
        
        if isinstance(action, RestartAppEvent):
            element = RESTART_element
        elif isinstance(action, UIEvent):
            element = action.view
        else:
            element = GOBACK_element
        action_target = ACTION_INEFFECTIVE \
            if from_state.state_str == to_state.state_str \
            else to_state.state_str
        # TODO decide how to represent the effect of an action
        # action_effect = f'{from_state.structure_str}->{action_target}'
        action_effect = action_target
        # 此处老报错，加一个 try except 试一试
        try:
            self.known_transitions[action_str] = {
                'from_state': from_state,
                'to_state': to_state,
                'action': action,
                'action_effect': action_effect
            }
        except:
            pass
        self.update_action_effects(from_state, to_state, action)  

        try:
            from_semantic_state = from_state_info['semantic_state_title']
            to_semantic_state = to_state_info['semantic_state_title']
            semantic_element_title = element['semantic_element_title'] if 'semantic_element_title' in element else element['desc']
            # 这个地方 semantic_elements 可能不在当前界面当中，需要去 semantic_states 中遍历
            # action_targets = self.semantic_states[from_semantic_state]['semantic_elements'][semantic_element_title]['action_targets']
            # if semantic_element_title in self.semantic_states[from_semantic_state]['semantic_elements']:
            action_targets = self.semantic_states[from_semantic_state]['semantic_elements'][semantic_element_title]['action_targets']
            # else:
            #     # for 
            #     action_targets = self
            action_type = Utils.get_action_type(action)
            if action_type not in action_targets:
                self.logger.warn(f'save_transition: action_type {action_type} not available')
            else:
                if ADDTEXT and action_type == 'set_text':
                    action_targets[action_type].append([action_target, action.text])
                else:
                    action_targets[action_type].append(action_target)
        except Exception as e:
            pass
            # pdb.set_trace()
            # print()
            
    def update_action_effects(self, from_state, to_state, action):
        if not isinstance(action, UIEvent) and not Manual_mode:  
            return None
        if isinstance(action, RestartAppEvent):
            element = RESTART_element
        elif isinstance(action, UIEvent):
            element = action.view
        else:
            element = GOBACK_element
        is_effective = from_state.state_str != to_state.state_str
        from_state_title = self.known_states[from_state.state_str]['semantic_state_title']
        from_state_id = list(self.semantic_states.keys()).index(from_state_title)
        to_state_title = self.known_states[to_state.state_str]['semantic_state_title']
        to_state_id = list(self.semantic_states.keys()).index(to_state_title)
        action_type = Utils.get_action_type(action)
        element_desc = element['desc']
        element_status = ','.join(element['status'])
        semantic_element_title = element['semantic_element_title'] if 'semantic_element_title' in element else element['desc']
        
        try:
            element_id = list(self.semantic_states[from_state_title]['semantic_elements'].keys()).index(semantic_element_title)
            element_class = element['class']
            element_size = element['size']
            new_effect = {
                'from_page': from_state_id,
                'to_page': to_state_id,
                'action_type': action_type,
                'element_id': element_id,
                'element_desc': element_desc,
                'element_class': element_class,
                'element_size': element_size,
                'element_status': element_status,
                'text': action.text if hasattr(action, 'text') else None,
                'effective': is_effective
            }
            self.action_effects = pd.concat([self.action_effects, pd.DataFrame([new_effect])], ignore_index=True)
        except Exception as e:
            print("update_action_effects failed")
            new_effect = {}
        return new_effect
    
    def _get_target_semantic_states(self, target_state_strs):
        semantic_states = []
        for target_state_str in target_state_strs:
            if target_state_str in self.known_states:
                state_info = self.known_states[target_state_str]
                semantic_states.append(state_info['semantic_state_title'])
            elif target_state_str == ACTION_INEFFECTIVE:
                semantic_states.append(target_state_str)
            else:
                self.logger.warn(f'_get_target_semantic_states unknown state_str: {target_state_str}')
        if not semantic_states:
            return []
        semantic_states_ordered = []
        for state, count in collections.Counter(semantic_states).most_common():
            semantic_states_ordered.append((state, count))
        return semantic_states_ordered

    def save_structure(self, state):
        structure_str = state.structure_str
        is_new_structure = False
        if structure_str not in self.known_structures:
            self.known_structures[structure_str] = []
            is_new_structure = True
        self.known_structures[structure_str].append(state)
        return is_new_structure
    
    # 遍历探索过的页面，对于每个页面，遍历其组件，如果在该组件上有过动作，
    # 就将 semantic_state_title, semantic_element_title, action_type 添加到 explored_semantic_actions 中，
    # 最后返回 explored_semantic_actions
    """
    explored_semantic_actions
    {
        ('e004cd. Elements: ', '<input bound_box=154,98,606,230>Search</input>', 'set_text'), 
        ('e004cd. Elements: ', "<button alt='Clear call history' bound_box=782,98,914,230></button>", 'touch')
    }
    semantic_element_title: 组件转换为 HTML 的描述，包括其边框属性
        <p bound_box=829,1964,971,2030>Call History</p>
    semantic_elements[semantic_element_title]: element 的 id、可以执行的动作、同一界面上的其他元素
    {
        'elements': [('11fb6c', 10)], 
        'action_targets': {'touch': []}, 
        'similar_semantic_elements': {"<button alt='Search' bound_box=44,98,154,230></button>": -1, '<input bound_box=154,98,606,230>Search</input>': -1, "<button alt='Filter' bound_box=650,98,782,230></button>": -1, "<button alt='Clear call history' bound_box=782,98,914,230></button>": -1, "<button alt='More options' bound_box=914,98,1025,230></button>": -1, '<scrollbar bound_box=0,252,1080,1854></scrollbar>': -1, '<p bound_box=0,252,1080,355>No previous calls have been found</p>': -1, "<button alt='Dialpad' bound_box=871,1656,1025,1810></button>": -1, '<button bound_box=124,1964,235,2030>Contacts</button>': -1, '<button bound_box=483,1964,597,2030>Favorites</button>': -1}
    }
    """
    def get_explored_semantic_actions(self):
        explored_semantic_actions = set()
        # semantic_state_title：界面的 id、组件 
        # "e004cd. Elements:"
        for semantic_state_title in self.semantic_states:
            # 得到该界面的 structure_str
            state_structure_strs = self.semantic_states[semantic_state_title]['states_structures']
            # 得到该界面的 structure HTML 描述
            state_strs = self.semantic_states[semantic_state_title]['states']
            state_structure_HTMLs = []
            for state_str in state_strs:
                state_structure_HTMLs.append(self.known_states[state_str]['state'].text_representation_frame)
            k = 0
            semantic_elements = self.semantic_states[semantic_state_title]['semantic_elements']
            for semantic_element_title in semantic_elements:
                # pdb.set_trace()
                k += 1
                # semantic_element_title = semantic_elements[i]
                action_targets = semantic_elements[semantic_element_title]['action_targets']
                similar_semantic_elements = semantic_elements[semantic_element_title]['similar_semantic_elements']
                for action_type in action_targets:
                    target_state_strs = action_targets[action_type]
                    if not target_state_strs:
                        continue
                    explored_semantic_actions.add((semantic_state_title, semantic_element_title, action_type))
                    # also mark the similar elements as explored
                    for similar_semantic_element in similar_semantic_elements:
                        if similar_semantic_elements[similar_semantic_element] > 0:
                            explored_semantic_actions.add((semantic_state_title, similar_semantic_element, action_type))
                    
                    # 将相同 state frame 上的 element 也标注为 explored
                    for semantic_state_title_i in self.semantic_states:
                        if semantic_state_title_i == semantic_state_title:
                            continue
                        # semantic_state_structure_strs_i = self.semantic_states[semantic_state_title_i]['states_structures']
                        semantic_state_strs_i = self.semantic_states[semantic_state_title_i]['states']
                        # 遍历该 semantic_state, 如果有 frame 相同的 state 就将其 element 也置为 explored
                        for semantic_state_str_i in semantic_state_strs_i:
                            state_structure_HTML_i = self.known_states[semantic_state_str_i]['state'].text_representation_frame
                            if state_structure_HTML_i in state_structure_HTMLs:
                        # for semantic_state_structure_str_i in semantic_state_structure_strs_i:
                        #     if semantic_state_structure_str_i in state_structure_strs:
                                j = 0
                                semantic_elements_i = self.semantic_states[semantic_state_title_i]['semantic_elements']
                                for semantic_element_title_j in semantic_elements_i:
                                    j += 1
                                    if j == k:
                                        explored_semantic_actions.add((semantic_state_title_i, semantic_element_title_j, action_type))
                    
                    # 获取 similar_semantic_element 的 id，将相同 state frame 上对应的 element 也标注为 explored
                    for similar_semantic_element in similar_semantic_elements:
                        if similar_semantic_elements[similar_semantic_element] > 0:
                            similar_semantic_element_id = 0
                            for similar_semantic_element_title in semantic_elements:
                                if similar_semantic_element_title == similar_semantic_element:
                                    break
                                similar_semantic_element_id += 1
                            # 将相同 state frame 上对应的 element 也标注为 explored
                            for semantic_state_title_i in self.semantic_states:
                                if semantic_state_title_i == semantic_state_title:
                                    continue
                                semantic_state_strs_i = self.semantic_states[semantic_state_title_i]['states']
                                # 遍历该 semantic_state, 如果有 frame 相同的 state 就将其 element 也置为 explored
                                for semantic_state_str_i in semantic_state_strs_i:
                                    state_structure_HTML_i = self.known_states[semantic_state_str_i]['state'].text_representation_frame
                                    if state_structure_HTML_i in state_structure_HTMLs:
                                        j = 0
                                        semantic_elements_i = self.semantic_states[semantic_state_title_i]['semantic_elements']
                                        for semantic_element_title_j in semantic_elements_i:
                                            j += 1
                                            if j == similar_semantic_element_id:
                                                # print("############### same function same structure set as explored ###############")
                                                # print("############################################################################")
                                                explored_semantic_actions.add((semantic_state_title_i, semantic_element_title_j, action_type))
                    
                            # explored_semantic_actions.add((semantic_state_title, similar_semantic_element, action_type))
        return explored_semantic_actions

    # 遍历 所有的界面 上的 所有的组件 上的 所有支持的动作，如果该动作没有被触发过，那就形成一个元组 (state, element, action_type), 其加入 unexplored_actions 中
    # unexplored_actions 是一个 list, 其中元素格式为(state, element, action_type)
    # 最后返回 unexplored_actions
    def get_unexplored_actions(self, find_in_states=[], skip_similar=True, prefer_unique=False):
    # def get_unexplored_actions(self, find_in_states=[], skip_similar=True, prefer_unique=True):
        unexplored_actions = []
        if not find_in_states:
            return unexplored_actions
        unique_actions = []
        explored_semantic_actions = self.get_explored_semantic_actions()
        for state in find_in_states:
            # 首先应该判断该 state 是否在当前 APP 的 activity 中，如果不在，就跳过
            package_name = state.foreground_activity.split("/")[0]  # com.simplemobiletools.dialer
            activity_name = state.foreground_activity.split("/")[1]  # .activities.MainActivity
            activity_name_full = package_name + activity_name  # com.simplemobiletools.dialer.activities.MainActivity

            state_activity_in_app_activities = False
            if activity_name_full in self.app.activities or activity_name in self.app.activities:
                state_activity_in_app_activities = True
            if not state_activity_in_app_activities:
                # print("############### not state_activity_in_app_activities ##############")
                # print("###################################################################")
                continue

            # 使用 _memorize_state 函数获取界面的 state_info, 
            # 如果 memory 中没有记录该 state, 则会为其创建一个 state_info, 并添加到 known_states 中。
            # 所以应该修改 _memorize_state 中的 page merge 的逻辑。
            state_info = self._memorize_state(state)
            semantic_state_title = state_info['semantic_state_title']

            # 获取 same_function_element_groups 用于后续 element 的匹配
            same_function_element_groups = state_info['same_function_element_groups']
            # print("########## same_function_element_groups #########")
            # print(same_function_element_groups)
            # print("#################################################")

            state_frame = state.text_representation_frame

            for ei, element in enumerate(state_info['elements']):
                # print("########## semantic_element_title #########")
                # print(state_info['elements'][ei]['semantic_element_title'])
                # print("#################################################")
                semantic_element_title = element['semantic_element_title']
                semantic_element_tag = re.search(r'<(\S+) ', semantic_element_title).group(1)
                semantic_element_bound = re.search(r"bound_box=(\d+,\d+,\d+,\d+)", semantic_element_title).group(1)

                # 遇到 p 标签就不要做动作了
                if "</p>" in semantic_element_title:
                    continue

                # action_targets = semantic_elements[semantic_element_title]['action_targets']
                for action_type in element['allowed_actions']:
                    semantic_action = (semantic_state_title, semantic_element_title, action_type)
                    
                    # 如果该 action 之前 navigate fail 过，那就直接跳过
                    if semantic_action in self.nav_failed_actions:
                        continue

                    # 如果是对 input 标签做 long_touch, 那就直接跳过
                    if "</input>" in semantic_element_title and action_type == "long_touch":
                        continue
                    
                    # 如果该 element 上的该 action 被触发过，就直接跳过
                    if semantic_action in explored_semantic_actions:
                        continue
                    
                    # TODO: Test
                    # 如果有和该 state frame 相同的组件被触发过，那就直接跳过
                    same_state_frame_element_explored = False
                    for known_state in self.known_states.keys():
                        # if known_state == state.state_str:
                            # continue
                        # try:
                        known_state_info = self._memorize_state(self.known_states[known_state]['state'])
                        # except:
                        #     pdb.set_trace()
                        known_state_frame = self.known_states[known_state]['state'].text_representation_frame
                        if  known_state_frame == state_frame:
                            for known_state_ei, known_state_element in enumerate(known_state_info['elements']):
                                known_state_semantic_element_title = known_state_element['semantic_element_title']
                                known_state_semantic_element_tag = re.search(r'<(\S+) ', known_state_semantic_element_title).group(1)
                                known_state_semantic_element_bound = re.search(r"bound_box=(\d+,\d+,\d+,\d+)", known_state_semantic_element_title).group(1)
                                # print("############### test_state_frame_element_explored ###############")
                                # semantic_known_state_title = self.known_states[known_state]['semantic_state_title']
                                # semantic_known_state_action = (semantic_known_state_title, known_state_semantic_element_title, action_type)
                                # print(explored_semantic_actions)
                                # print(semantic_known_state_action)
                                # print("#################################################################")
                                if known_state_semantic_element_tag == semantic_element_tag and known_state_semantic_element_bound == semantic_element_bound:
                                    
                                    # if action_type in 
                                    # self.semantic_states[semantic_state_title]['semantic_elements'][semantic_element_title]
                                    semantic_known_state_title = self.known_states[known_state]['semantic_state_title']
                                    semantic_known_state_action = (semantic_known_state_title, known_state_semantic_element_title, action_type)
                                    if semantic_known_state_action in explored_semantic_actions:
                                        # pdb.set_trace()
                                        same_state_frame_element_explored = True
                                        break
                        if same_state_frame_element_explored:
                            break
                    if same_state_frame_element_explored:
                        # print("############### same_state_frame_element_explored ###############")
                        # print("#################################################################")
                        # pdb.set_trace()
                        continue

                    # 如果该 element 的同一个 state 上有 same_function_element 的元素执行过该动作，那就跳过
                    same_function_element_explored = False
                    if same_function_element_groups:
                        for same_function_element_group in same_function_element_groups:
                            if ei in same_function_element_group:
                                # print("########## same_function_element_group ##########")
                                # print(same_function_element_group)
                                # print("#################################################")
                                for same_function_element_id in same_function_element_group:
                                    same_function_element_title = state_info['elements'][same_function_element_id]['semantic_element_title']
                                    same_function_element_semantic_action = (semantic_state_title, same_function_element_title, action_type)
                                    if same_function_element_semantic_action in explored_semantic_actions:
                                        same_function_element_explored = True
                                        # print("########## explored_semantic_action == semantic_action ##########")
                                        # print(same_function_element_semantic_action)
                                        # print(semantic_action)
                                        # print("#################################################################")
                                        break
                    if same_function_element_explored:
                        # print("########## same_function_element_explored ##########")
                        # print("####################################################")
                        continue

                    # 如果该 element 上的该 action 没有被触发过
                    # 如果该动作是 long_touch 然而该 element 的 touch 还没有被触发过，就先不处理 long_touch
                    if action_type == "long_touch" and "touch" in element['allowed_actions']:
                        semantic_action_touch = (semantic_state_title, semantic_element_title, "touch")
                        if semantic_action_touch not in explored_semantic_actions:
                            continue
                    
                    from_state_id = list(self.semantic_states.keys()).index(semantic_state_title)
                    element_status = ','.join(element['status'])
                    element_class = element['class']
                    element_size = element['size']
                    element_desc = element['desc']
                    df = self.action_effects
                    if skip_similar and len(self.action_effects) > SKIP_SIMILAR_ACTION_THRESHOLD:
                        # same element across different states
                        df1 = df[(df['element_desc']==element_desc) & (df['element_status']==element_status) & (df['action_type']==action_type)] \
                            [['to_page', 'effective']]
                        if len(df1) > SKIP_SIMILAR_ACTION_THRESHOLD and len(df1.drop_duplicates()) == 1:
                            continue
                        # similar elements in the same state
                        df2 = df[(df['from_page']==from_state_id) & (df['element_class']==element_class) & (df['element_size']==element_size) & \
                            (df['element_status']==element_status) & (df['action_type']==action_type)] \
                            [['to_page', 'effective']]
                        if len(df2) > SKIP_SIMILAR_ACTION_THRESHOLD and len(df2.drop_duplicates()) == 1:
                            continue
                    if prefer_unique and len(self.action_effects) > 1:
                        df3 = df[(df['element_class']==element_class) & (df['element_size']==element_size) & \
                            (df['element_status']==element_status) & (df['action_type']==action_type)] \
                            [['to_page', 'effective']]
                        if len(df3) == 0:
                            unique_actions.append((state, element, action_type))
                    unexplored_actions.append((state, element, action_type))
        if prefer_unique and len(unique_actions) > 0:
            return unique_actions
        return unexplored_actions
    
    def gen_input_text(self, state_desc, target_element, with_llm=EXPLORE_WITH_LLM):
        """
        return a text string that can be the input text for the target element
        """
        if not with_llm:
            return DUMMY_INPUT

        prompt = f'Now suppose you are analyzing a GUI page with following elements:\n{state_desc}\n' + \
            f'For the input tag id={target_element["local_id"]}, give an example of possible inputs; the input example you generate should be short and precise, based on the elements on the current interface.\n' + \
            'Please respond in the following format:\n' + \
            ' Input text: "<the generated input text>"'
        response = GPT.query(prompt, self.device.output_dir)
        input_text = re.search(r'Input text: "(.+)"', response)
        input_text = input_text.group(1).strip() if input_text else DUMMY_INPUT
        return input_text
    
    # 从 state 中随机选择一个元素，并从该 element 的 allowed_actions 中随机选择一个动作
    # 如果 state 为空，则随机选择一个界面
    def get_executable_action(self, state=None, element=None, action_type=None, input_text=None):
        # 如果上次的 state 和这次的不一样，那就 press back
        # if state != self.last_random_action_state:
        #     self.last_random_action_state = state
        #     return state, "BACK"
        if state is None:
            if len(self.known_states) < 3:
                state_str = random.choice(self.known_states.keys())
                state = self.known_states[state_str]['state']
            else:
                return state, "BACK"
            # 此处本来是从已知的界面中随机选择一个，如下所示
            # state_str = random.choice(self.known_states.keys())
            # state = self.known_states[state_str]['state']
        state_desc, elements = state.text_representation
        element_without_p = []
        for i in range(len(elements)):
            element_desc = elements[i]["desc"]
            if "</p>" in element_desc:
                continue
            element_without_p.append(elements[i])
        if len(element_without_p) == 0:
            return state, "BACK"
        if element is None:
            element = random.choice(element_without_p)
        if action_type is None:
            allowed_actions = [action for action in element['allowed_actions']]
            if len(allowed_actions) > 1 and "long_touch" in allowed_actions:
                allowed_actions.remove("long_touch")
            # action_type = random.choice(element['allowed_actions'])
            action_type = random.choice(allowed_actions)
        if action_type == KEY_SetTextEvent and input_text is None:
            input_text = self.gen_input_text(state_desc, element) if action_type == KEY_SetTextEvent else None
        return state, Utils.pack_action(self.app, action_type, element, input_text)

    def add_nav_failed_actions(self, current_nav_target_action):
        target_state, target_element, target_action_type = current_nav_target_action
        if target_state != None:
            state_info = self._memorize_state(target_state)
            semantic_state_title = state_info['semantic_state_title']
            semantic_element_title = target_element['semantic_element_title']
            nav_failed_action = (semantic_state_title, semantic_element_title, target_action_type)
            self.nav_failed_actions.append(nav_failed_action)
            print("############### failed_nav_actions ###############")
            print(nav_failed_action)
            print("##################################################")


class LLM_Guided_Policy(UtgBasedInputPolicy):
    def __init__(self, device, app, random_input):
        super(LLM_Guided_Policy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.memory = Memory(utg=self.utg, app=self.app, device=self.device)
        self.previous_actions = []
        self._nav_steps = []
        self._num_steps_outside = 0
        self._num_steps_not_in_activity_stack = 0
        self._explore_start = 1
        self._num_app_not_in_activity_stack = 0
        self.explored_activities = set()
        self.start_time = time.time()
        self.time_coverage_rate = {
            'time': [],
            'coverage_rate': []
        }
        self.explored_activities_not_increase_time = 0
        self.current_nav_target_action = (None, None, None)
        self.explored_states_in_order = []
        self.navigate_num = 0
        # # for manually generating UTG
        # self.manual = Manual_mode

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        def returned_action(state, action):
            action_desc = Utils.action_desc(action)
            self.logger.info(f'>> executing action in state {state.state_str}: {action_desc}')
            self.previous_actions.append(action)
            return action


        explore_new_activity = self.calculate_activity_coverage_rate()

        # 如果已经探索到的 activity 数量太长时间没有增加，那就重启 APP
        current_state = self.current_state
        if explore_new_activity:
            self.explored_activities_not_increase_time = 0
        else:
            self.explored_activities_not_increase_time += 1
        print(f"Explored activities not increase num: {self.explored_activities_not_increase_time}")
        if self.explored_activities_not_increase_time > MAX_EXPLORED_ACTIVITIES_NOT_INCREASE_TIME:
            self.logger.info("explored activities not increase for too long, restarting app")
            self.explored_activities_not_increase_time = 0
            return returned_action(current_state, RestartAppEvent(app=self.app))


        # 记录当前界面
        current_state = self.current_state
        try:
            self.memory.save_transition(self.last_event, self.last_state, current_state)
        except Exception as e:
            self.logger.warning(f'failed to save transition: {e}')
            import traceback
            traceback.print_exc()

        # 人工模式
        if Manual_mode and self.last_event is not None:
            executable_action = self.get_manual_action(current_state)
            self.logger.debug("current state: %s" % current_state.state_str)
            self._dump_memory()
            return returned_action(current_state, executable_action)

        if self.last_event is not None:
            self.last_event.log_lines = self.parse_log_lines()
        # interested_apis = self.monitor.get_interested_api()
        # self.monitor.check_env()
        self.logger.debug("current state: %s" % current_state.state_str)
        self._dump_memory()

        # navigate 模式
        # 在此处添加一个逻辑，如果上一步也是从这儿 navigate 出去，就计数，超过一定阈值，就记为 navigate 失败。
        nav_action, n_steps = self.navigate(current_state)
        if nav_action:
            self.navigate_num += 1
            self.logger.info(f'navigate continue for {self.navigate_num} steps')
            self.logger.info(f'navigating, {n_steps} steps left')
            return returned_action(current_state, nav_action)
        self._nav_steps = []  # if navigation fails, stop navigating

        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            print("app is not in the activity stack")


            # 修改原始逻辑为：跳转到别的 APP 之后，先点击一次 back，若不行，再点击一次 back，若还不行，就重启
            # time to start explore, need to start the APP
            if self._explore_start == 1:
                start_app_intent = self.app.get_start_intent()
                start_app_action = IntentEvent(intent=start_app_intent)
                print("#############################################################")
                print("time to start explore, start the APP")
                print("#############################################################")
                self.logger.info("starting app")
                self._explore_start = 0
                return returned_action(current_state, start_app_action)
            # exploration has started, close the current activity, then restart
            elif self._num_app_not_in_activity_stack < 2:
                self._num_app_not_in_activity_stack += 1
                self.logger.info("touch back")
                return returned_action(current_state, KeyEvent(name="BACK"))
            else:
                self._num_app_not_in_activity_stack = 0
                self.logger.info("restarting app")
                return returned_action(current_state, RestartAppEvent(app=self.app))


        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            print("app is in activity stack but is not in foreground")
            self._num_app_not_in_activity_stack = 0
            self._num_steps_outside += 1
            print("the app has not been in foreground for:" + str(self._num_steps_outside))
            if self._num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                print("the app has not been in foreground for too long:" + str(self._num_steps_outside))
                if self._num_steps_outside > MAX_NUM_STEPS_OUTSIDE + 1:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    # 此处逻辑由启动应用的 intent, 修改为 restart 关闭再重启的 intent。
                    # start_app_intent = self.app.get_start_intent()
                    # go_back_event = IntentEvent(intent=start_app_intent)
                    go_back_event = RestartAppEvent(app=self.app)
                self.logger.info("going back to the app")
                return returned_action(current_state, go_back_event)
        else:
            # If the app is in foreground
            print("the app is in foreground")
            self._num_app_not_in_activity_stack = 0
            self._num_steps_outside = 0
            self._num_steps_not_in_activity_stack = 0

        # 探索次数太长，那就重启
        steps_since_last_kill = 0
        for previous_action in reversed(self.previous_actions):
            if isinstance(previous_action, RestartAppEvent):
                break
            steps_since_last_kill += 1
        if steps_since_last_kill > MAX_NAV_STEPS:
            self.logger.info(f"exploring too long, kill and restart")

            # 此处从单纯的 kill APP，改为关闭并重启
            # return returned_action(current_state, KillAppEvent(app=self.app))
            return returned_action(current_state, RestartAppEvent(app=self.app))
        

        # 添加逻辑：如果在一个 state 上点击了多次都没有变化，那就点击 back。
        current_state_explore_num = 0
        self.explored_states_in_order.append(current_state.text_representation_frame)
        print("########### current state text_representation_frame ###########")
        print(current_state.text_representation_frame)
        print("###############################################################")
        for i in range(len(self.explored_states_in_order)-1, -1, -1):
            if self.explored_states_in_order[i] == current_state.text_representation_frame:
                current_state_explore_num += 1
            else:
                break
        print("Explore current state frame num:", current_state_explore_num)
        if min(current_state_explore_num, steps_since_last_kill) > MAX_EXPLORE_CURRENT_STATE_TIME:
            # self.explored_states_in_order.append(current_state)
            print("########## explore the same state frame for too long ##########")
            print("###############################################################")
            return returned_action(current_state, KeyEvent(name="BACK"))

        # 重启 APP 失败次数太多，那就将 APP 卸了重装
        num_start_app_retry = 0
        for previous_action in reversed(self.previous_actions):
            if isinstance(previous_action, IntentEvent) and previous_action.intent == self.app.get_start_intent():
                num_start_app_retry += 1
            else:
                break
        print("num_start_app_retry: " + str(num_start_app_retry))
        if num_start_app_retry > MAX_START_APP_RETRY:
            self.logger.info(f"starting app failed for {num_start_app_retry} times, reinstalling the app")
            self.device.uninstall_app(self.app)
            self.device.install_app(self.app)
            self.previous_actions = []
            start_app_intent = self.app.get_start_intent()
            start_app_action = IntentEvent(intent=start_app_intent)
            return returned_action(current_state, start_app_action)


        # 选择一个没有探索过的 (element, action) 进行随机探索
        if len(self._nav_steps) == 0 and np.random.uniform() > RANDOM_EXPLORE_PROB:
            # 在当前界面上选
            target_state, target_action = self.pick_target(current_state)
            if target_state:
                # perform target action
                self.logger.info(f"exploring current state")
                return returned_action(current_state, target_action)
            # 当前界面上没有没探索过的，就去全部界面上选
            target_state, target_action, nav_steps = self.pick_navigate_target(current_state)
            self.navigate_num = 0
            if target_state:
                # navigate to target action
                self.logger.info(f"exploring state {target_state.state_str}, action: {Utils.action_desc(target_action)}")
                self._nav_steps = nav_steps
                nav_action, n_steps = self.navigate(current_state)
                if nav_action:
                    self.logger.info(f'navigate continue for {self.navigate_num} steps')
                    self.logger.info(f'navigating, {n_steps} steps left')
                    self.navigate_num += 1
                    return returned_action(current_state, nav_action)
        self._nav_steps = []  # if navigation fails, stop navigating


        # 此处本来是在当前界面随机选一个元素，但是容易在该界面卡死（如果该界面只有<p>的话），代码如下所示        
        self.logger.info("trying random action")
        # possible_events = current_state.get_possible_input()
        # possible_events.append(KeyEvent(name="BACK"))
        # random.shuffle(possible_events)
        # action = possible_events[0]
        # if isinstance(action, UIEvent) and 'desc' not in action.view:
        #     print('invalid action: ', action.view)
        _, random_action = self.memory.get_executable_action(state=current_state)
        if random_action == "BACK":
            press_back_event = KeyEvent(name='BACK')
            return returned_action(current_state, press_back_event)
        return returned_action(current_state, random_action)

        # 现在修改为点击 back
        # press_back_event = KeyEvent(name='BACK')
        # return returned_action(current_state, press_back_event)


    def pick_target(self, current_state):
        unexplored_actions = self.memory.get_unexplored_actions(find_in_states=[current_state])
        if not unexplored_actions:
            return None, None

        unexplored_actions_without_long_touch = []
        for unexplored_action in unexplored_actions:
            (state, element, action_type) = unexplored_action
            if action_type != 'long_touch':
                unexplored_actions_without_long_touch.append(unexplored_action)
        
        # 从当前界面除了 long_touch 的动作中选一个，把 long_touch 留到 pick_navigate_target 中去。
        # 因为 long_touch 一般没有什么效果，这样可以提高探索效率
        if len(unexplored_actions_without_long_touch) > 0:
            (state, element, action_type) = random.choice(unexplored_actions_without_long_touch)
            _, action = self.memory.get_executable_action(state, element, action_type)
        else:
            state = None
            action = None


        # 从当前界面界面能进行的所有 action 中随机选择一个
        # (state, element, action_type) = random.choice(unexplored_actions)
        # _, action = self.memory.get_executable_action(state, element, action_type)

        return state, action
    
    def pick_navigate_target(self, current_state, randomly=True, shortest=True):
        unexplored_actions = self.memory.get_unexplored_actions(find_in_states=self.memory.all_states(in_app_only=ONLY_EXPLORE_IN_APP))
        if randomly:
            random.shuffle(unexplored_actions)
        target_state, target_element, target_action_type, nav_steps = None, None, None, None
        for state_, element_, action_type_ in unexplored_actions:
            nav_steps_ = self.get_shortest_nav_steps(current_state, state_)
            if nav_steps_ is None:
                continue
            if nav_steps is None or len(nav_steps_) < len(nav_steps):
                target_state, target_element, target_action_type, nav_steps = state_, element_, action_type_, nav_steps_
                if not shortest:   # no need to return shortest, return now
                    break
        if target_state is None:
            return None, None, None
        _, target_action = self.memory.get_executable_action(target_state, target_element, target_action_type)
        self.current_nav_target_action = (target_state, target_element, target_action_type)
        nav_steps = nav_steps + [(target_state, target_action)]
        return target_state, target_action, nav_steps

    def navigate(self, current_state):
        if self._nav_steps and len(self._nav_steps) > 0:
            # 如果 navigate 次数太多，将该 navigate target 设置为 fail
            if self.navigate_num > MAX_NAVIGATE_NUM_AT_ONE_TIME:
                self.memory.add_nav_failed_actions(self.current_nav_target_action)
                self.logger.warning("navigate for too long at one time, stop navigate")
                return None, 0

            nav_state, nav_action = self._nav_steps[0]
            self._nav_steps = self._nav_steps[1:]
            # nav_action_ = self._get_nav_action(current_state, nav_state, nav_action)
            nav_action_, nav_enable = self._get_nav_action(current_state, nav_state, nav_action)
            # if nav_action_:
            if nav_enable:
                return nav_action_, len(self._nav_steps)
            else:
                # 如果 navigate 失败了，那就将这个不能 navigate 的 action 添加到 memory 当中。
                self.memory.add_nav_failed_actions(self.current_nav_target_action)
                # self.memory.failed_nav_actions.append(self.current_nav_target_action)
                self.logger.warning(f"navigate: failed in state {current_state.state_str}")
                # self.utg.remove_transition(self.last_event, self.last_state, nav_state)  # FIXME how to punish the failed navigation
        return None, 0

    def _get_nav_action(self, current_state, nav_state, nav_action):
        # get the action similar to nav_action in current state
        try:
            # if current_state.structure_str != nav_state.structure_str:
            #     return None
            if not isinstance(nav_action, UIEvent):  # 处理 kill_app_event 这种 action
                return nav_action, False
            nav_view = nav_action.view
            nav_view_desc = nav_view['desc']
            new_state_views = current_state.text_representation[-1]
            new_view_idx = [view['desc'] for view in new_state_views].index(nav_view_desc)
            new_view = new_state_views[new_view_idx]
            input_text = nav_action.text if hasattr(nav_action, 'text') else None
            new_action = Utils.pack_action(self.app, action_type=Utils.get_action_type(nav_action), target_element=new_view, input_text=input_text)
            # new_action = copy.deepcopy(nav_action)
            # new_action.view = new_view
            return new_action, True
        except Exception as e:
            self.logger.warning(f'exception during _get_nav_action: {e}')
            return nav_action, False

    def parse_log_lines(self):
        log_lines = self.device.logcat.get_recent_lines()
        filtered_lines = []
        app_pid = self.device.get_app_pid(self.app)
        # print(f'current app_pid: {app_pid}')
        for line in log_lines:
            try:
                seps = line.split()
                if int(seps[2]) == app_pid:
                    filtered_lines.append(line)
            except:
                pass
        return filtered_lines

    def get_shortest_nav_steps(self, current_state, target_state):
        normal_nav_steps = self.utg.get_G2_nav_steps(current_state, target_state)
        restart_nav_steps = self.utg.get_G2_nav_steps(self.utg.first_state, target_state)
        normal_nav_steps_len = len(normal_nav_steps) if normal_nav_steps else MAX_NAV_STEPS
        restart_nav_steps_len = len(restart_nav_steps) + 1 if restart_nav_steps else MAX_NAV_STEPS
        if normal_nav_steps_len >= MAX_NAV_STEPS and restart_nav_steps_len >= MAX_NAV_STEPS:
            self.logger.warning(f'get_shortest_nav_steps: cannot find a path to {target_state.structure_str} {target_state.foreground_activity}')

            return None
        elif normal_nav_steps_len > restart_nav_steps_len:  # prefer shortest path
        # elif normal_nav_steps_len >= MAX_NAV_STEPS:  # prefer normal navigation
            nav_steps = [(current_state, KillAppEvent(app=self.app))] + restart_nav_steps
        else:
            nav_steps = normal_nav_steps
        return nav_steps
    
    def _dump_memory(self):
        """
        Output current memory to text files
        """
        if not self.device.output_dir:
            return
        if self.action_count % DUMP_MEMORY_NUM_STEPS != 1 and not Manual_mode:
            return
        self.memory.action_history.to_csv(os.path.join(self.device.output_dir, "actions.csv"), encoding="utf-8")
        memory_path = os.path.join(self.device.output_dir, "memory.txt")
        memory_str = self.memory.to_string()
        # with open(memory_path, "w") as memory_file:
        with open(memory_path, "w", encoding='utf-8') as memory_file:
            memory_file.write(memory_str)
            
    def get_manual_action(self, state):
        
        def debug_action_extract(actions):
            # TODO: add an exit and restart action
            ele_set, action_set, input_set = False, False, False
            element_id, action_choice, input_text_value = None, None, None
            while not ele_set:
                try:
                    response = input(f"Please input element id:")
                    element_id = int(response)
                    ele_set = True
                    break
                except KeyboardInterrupt:
                    raise KeyboardInterrupt()
                except:
                    print('warning, wrong format, please input again')
                    continue
                
            while not action_set:
                try:
                    actions_desc = [f'({i}) {actions[element_id][i]}' for i in range(len(actions[element_id]))]
                    print('You can choose from: ', '; '.join(actions_desc))
                    response = input(f"Please input action id:")
                    action_choice = int(response)
                    action_set = True
                    break
                except KeyboardInterrupt:
                    raise KeyboardInterrupt()
                except:
                    print('warning, wrong format, please input again')
                    continue
                
            if actions[element_id][action_choice] == 'set_text':
                while not input_set:
                    try:
                        input_text_value = input(f"Please input the text:")
                        input_set = True
                        break
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt()
                    except:
                        print('warning, wrong format, please input again')
                        continue
            return element_id, action_choice, input_text_value
        
        element_descs, actiontypes, all_elements = self.parse_all_executable_actions(state)
        element_descs_without_bbox = [re.sub(r'\s*bound_box=\d+,\d+,\d+,\d+', '', desc) for desc in element_descs]
        state_desc = "\n".join(element_descs_without_bbox)
        print('='*80, f'\n{state_desc}\n', '='*80)
        
        id, action_id, input_text = debug_action_extract(actiontypes)
        selected_action_type, selected_element = actiontypes[id][action_id], all_elements[id]
        
        file_path = os.path.join(self.device.output_dir, 'log.yaml')
        _save2yaml(file_path, state_desc, id, input_text, selected_action_type, state.state_str, state.structure_str, state.tag, state.width, state.height)
        return Utils.pack_action(self.app, selected_action_type, selected_element, input_text)
                   
    def parse_all_executable_actions(self, state):
        state_info = self.memory._memorize_state(state)
        elements = state_info['elements']
        
        element_descs, actiontypes, all_elements = [], [], []  # an element may have different action types, so len(all_elements)>len(elements)

        for element_id, element in enumerate(elements):
            element_desc = f"element {element_id}: {element['full_desc']}"
            all_elements.append(element)
            actiontypes.append(element['allowed_actions'])
            element_descs.append(element_desc)
        state_dict_path = os.path.join(self.device.output_dir, 'StateDicts')
        if not os.path.exists(state_dict_path):
            os.mkdir(state_dict_path)
        state_dict_file = os.path.join(self.device.output_dir, f'StateDicts/{state.tag}.json')
        with open(state_dict_file, 'w') as f:
            json.dump(all_elements, f)
        return element_descs, actiontypes, all_elements
    

    def calculate_activity_coverage_rate(self):
        app_activities = self.app.activities

        activity_num = len(app_activities)
        reached_activities_num = len(self.utg.reached_activities)

        activities_coverage_rate = reached_activities_num / activity_num
        current_time = time.time()
        cost_time = current_time - self.start_time
        print("#################### app.activities.coverage.rate ####################")
        print(str(activities_coverage_rate * 100) + "%")
        print("########################### cost_time ################################")
        print(cost_time)
        print("######################################################################")
        self.time_coverage_rate['time'].append(cost_time)
        self.time_coverage_rate['coverage_rate'].append(activities_coverage_rate)
        # 保存为Excel文件
        time_coverage_rate_df = pd.DataFrame(self.time_coverage_rate)
        time_coverage_rate_path = os.path.join(self.device.output_dir, 'time_coverage_rate.csv')
        time_coverage_rate_df.to_csv(time_coverage_rate_path, index=False)
        # 返回是否有探索到了新的 activity
        if len(self.time_coverage_rate['coverage_rate']) > 1:
            return not (self.time_coverage_rate['coverage_rate'][-1] == self.time_coverage_rate['coverage_rate'][-2])
        else:
            return False


if __name__ == '__main__':
    r = GPT.query('hello!')
    print(r)

