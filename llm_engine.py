from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
from utils import generate_scene_graph

defalt_obj_list = ['cap', 'hook', 'table', 'chair', 'person', 'sofa', 'stool', 'wood block']

prompt_env = Environment(loader=FileSystemLoader('prompt_templates/'))
with open('config/joint_descriptions.txt', 'r') as f:
  joint_descriptions = f.read()

with open('config/base_description.txt', 'r') as f:
  base_description = f.read()

with open('config/hat_state_machine.txt', 'r') as f:
    state_machine = f.read()

with open('config/constraints_summary.txt', 'r') as f:
    constraints = f.read()

class NarrationEngine:
    def __init__(self, model='gpt-4-turbo'):
        self._narrator = OpenAI(
                            api_key='input_key',
                        )
        
        self.narration_history = {}
        self.model = model

    def _get_llm_response(self, prompt=None, query=None):
      response = self._narrator.chat.completions.create(
         messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}
            ],
            model=self.model,
      )
      message = response.choices[0].message.content
      return message

    def add_trajectory(self, trajectory_name):
       self.narration_history[trajectory_name] = []
       
    def add_frame_narration(self, trajectory_name, timestamp, narration):
      self.narration_history[trajectory_name].append((timestamp, narration))
  
    def summarize_env_rgbd(self, key_frame, vocab_list=defalt_obj_list, confidence=0.7):
      sg = generate_scene_graph(key_frame, vocab_list=vocab_list, confidence=confidence)
      env_sys_template = prompt_env.get_template('env_sys_prompt.txt')
      env_sys_prompt = env_sys_template.render(
          env_description = 'living room in a house',
          task_desc = 'putting the dirty cup into the sink in the kitchen'
      )
      env_template = prompt_env.get_template('env_summary.txt')
      env_prompt = env_template.render(
          scene_graph = sg
      )
      env_rgb_summary = self._get_llm_response(prompt=env_sys_prompt, query=env_prompt)
      return env_rgb_summary

    def summarize_internal(self, keyframe, personal_pronoun='first'):
      internal_template = prompt_env.get_template('internal_sys_prompt.txt')
      internal_sys_prompt = internal_template.render(
          personal_pronoun = personal_pronoun,
          joint_descriptions = joint_descriptions,
          base_description = base_description,
          
      )
      internal_template = prompt_env.get_template('internal_summary.txt')
      internal_prompt = internal_template.render(
          joints = keyframe['joint_state'],
          odom = keyframe['odom'],
      )
      
      internal_summary = self._get_llm_response(prompt=internal_sys_prompt, query=internal_prompt)
      return internal_summary
    
    def summarize_planning(self, keyframe, task_name=None, task_desc=None):
      planning_sys_template = prompt_env.get_template('planning_sys_prompt.txt')
      planning_sys_prompt = planning_sys_template.render(
        task = task_name,
        task_desc = task_desc,
        state_machine = state_machine,
      )
      planning_template = prompt_env.get_template('planning_summary.txt')
      planning_prompt = planning_template.render(
        current_state = keyframe['state'],
        state_history = keyframe['state_history']
      )
      planning_summary = self._get_llm_response(prompt=planning_sys_prompt, query=planning_prompt)
      return planning_summary
    
    def narrate_frame(self, task, env_summary=None, internal_summary=None, planning_summary=None, constrains=None, narration_history=[], mode='alert'):
      if mode == 'debug':
        narration_template = prompt_env.get_template('frame_narr_sys_prompt_debug.txt')
      elif mode == 'info':
        narration_template = prompt_env.get_template('frame_narr_sys_prompt_info.txt')
      elif mode == 'alert':
        narration_template = prompt_env.get_template('frame_narr_sys_prompt_alert.txt')

      narration_sys_prompt = narration_template.render(
          task = task
      )

      narration_template = prompt_env.get_template('frame_summary.txt')
      narration_prompt = narration_template.render(
          env_summary = env_summary,
          robot_summary = internal_summary,
          state_summary = planning_summary,
          constraints_summary = constraints,
          narration_history = narration_history
      )
      frame_narration = self._get_llm_response(prompt=narration_sys_prompt, query=narration_prompt)
      return frame_narration
      
       
    # TODO: Narrate about the trajectory
    def summarize_trajectory(self):
       pass
    
    # TODO: Narrate about the experiment
    def summarize_expriment(self):
       pass
    
    # TODO: Narrate about the experiment
    def analyze_risk(self):
       pass
    
    # TODO: Narrate about the experiment
    def analyze_failure(self):
       pass
    
    # TODO: Narrate about the experiment
    def assist_recovery(self):
       pass

