import os
import json
import time
import pandas as pd
from datetime import datetime
from config import PreAttackConfig
from concurrent.futures import ThreadPoolExecutor
from utils import gpt_call, read_prompt_from_file, parse_json, check_file, get_client, gpt_call_append


def preprocess_response(response_str):
    # Strip leading/trailing whitespace
    response_str = response_str.strip()
    
    # Check if the string starts and ends with triple backticks and remove them
    if response_str.startswith("```") and response_str.endswith("```"):
        # Split into lines and remove the first and last line (which are the fences)
        lines = response_str.splitlines()
        # Sometimes the first line might include language info (e.g., ```json)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        response_str = "\n".join(lines)
    
    return response_str


class PreAttack:
    def __init__(self, config: PreAttackConfig, starter = 1):
        self.pre_attack_path = config.pre_attack_path
        
        self.behavior_csv = check_file(config.behavior_csv)
        self.harmful_behav_csv = check_file(config.harmful_behav_csv)

        self.extract_prompt = read_prompt_from_file(config.extract_prompt)
        self.network_prompt = read_prompt_from_file(config.network_prompt)
        self.actor_prompt = read_prompt_from_file(config.actor_prompt)
        self.query_prompt = read_prompt_from_file(config.query_prompt)
        
        # data1
        df = pd.read_csv(self.behavior_csv)
        self.org_data = df['Goal'].tolist()

        # data2
        df2= pd.read_csv(self.harmful_behav_csv)
        #self.org_data = df2['goal'].tolist() # new dataset (AdvBench) for doing exprements
        
        # model
        self.model_name = config.model_name
        self.client = get_client(config.model_name)
        self.config = config

        self.starter = starter
        self.topic_number_threshold = 5 # *** topic hyper-parameter for optimization ***

    def get_gpu_info():
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader'], capture_output=True, text=True)
        output = result.stdout.strip().split('\n')
        for line in output:
            name, total, used, free = line.split(',')
            print(f"GPU: {name}, Total Memory: {total}, Used Memory: {used}, Free Memory: {free}")
            
    def retrieve_query_goal_and_type(self, org_query):
        prompt = self.extract_prompt.format(org_query=org_query)
        for _ in range(5):
            try:
                res = gpt_call(self.client, prompt, model_name=self.model_name)
                data = parse_json(res)
                return data['main_goal'], data['output_type']
            except Exception as e:
                print("Error in retrieve_query_goal_and_type:", e)
                continue
        return {}, {}
    
    def get_entites(self, main_goal, org_query, query_number = -1): 
        
        topic_list = []
        topic_details = []
        topic_timer = 10
        except_counter = 0
        except_counter_threshold = 3
        while True: 
            try:
                system_prompt = self.network_prompt.format(topic_list=", ".join(topic_list), main_goal=main_goal)
                #print('\n\n network_prompt: ',system_prompt,'\n\n')
                resp, dialog_hist = gpt_call_append(self.client, self.model_name, [], system_prompt)
                #print('\n\n topic_list: ',resp,'\n\n')
                #print('\n\n dialog_hist: ',dialog_hist,'\n\n')
    
                clean_resp = preprocess_response(resp)
                data = json.loads(clean_resp)
                topic_details += data
                topics = [entry["topic"] for entry in data]
                topic_list += topics
                if len(topic_list) >= self.topic_number_threshold:
                    print("\nquery_number111: ", query_number)
                    break
            except Exception as e:
                except_counter += 1
                #print('\n\n org_query: ',org_query)
                #print('\n\n main_goal: ',main_goal,'\n\n')
                #print("Error in getting entities:", e) 
                if except_counter > except_counter_threshold:
                    #print("\ntopic_list: ", len(topic_list))
                    return topic_details, {}, dialog_hist  
                time.sleep(topic_timer)
                
            
        #print('\n\n topic_list: ',topic_list,'\n\n')  
        
        while True: 
            try:
                entity_prompt = self.actor_prompt.format(topic_list=", ".join(topic_list), main_goal=main_goal)
                #print('\n\n entity_prompt: ',entity_prompt,'\n\n')
                sample_dict, dialog_hist = gpt_call_append(self.client, self.model_name, dialog_hist, entity_prompt)
                sample_dict = preprocess_response(sample_dict)
                #print('\n\n sample_dict: ',sample_dict,'\n\n') 
                sample_data = json.loads(sample_dict)
                print("\nsample_number2222: ", query_number)
                break
            except Exception as e:
                print("Error in getting samples:", e)
                time.sleep(topic_timer)
                
        return topic_details, sample_data, dialog_hist
    
    def get_init_thoughts(self, main_goal, sample_set, number):
        #print('\n\n sample_set before: ',sample_set,'\n\n')

        #print('\n\n sample_set: ',sample_set,'\n\n')
        query_prompt = self.query_prompt.format(main_goal=main_goal, sample_set=sample_set)
        updated_sample_set = {"topics":[]}
        for _ in range(3):
            try:
                new_sample_set, dialog_hist = gpt_call_append(self.client, self.model_name, [], query_prompt)
        
                #print('\n\n new_sample_set: ',new_sample_set,'\n\n')
                clean_sample_set = preprocess_response(new_sample_set)
                updated_sample_set = json.loads(clean_sample_set)
                break
            except Exception as e:
                print("\nError in get_init_thoughts:", e)
                #print("\nmain_goal1111: ", number, main_goal, sample_set)
                time.sleep(2)
        '''
        with open(self.pre_attack_path+'/PreOutput-'+str(number)+'.json', 'w') as f:
            json.dump(updated_sample_set, f, indent=4)
        '''
        
        #print('\n\n sample_set after: ',json.dumps(updated_sample_set, indent=4),'\n\n')
        
        return updated_sample_set
        
    def split_list(self, lst, n):
        """Split list lst into n nearly equal parts."""
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    
    def retrieve_sample_sets(self, sample_data):
        topics_list = sample_data["topics"]
        
        total_topics = len(topics_list)
        num_partitions = 5
        partition_size = total_topics // num_partitions
        partitions = self.split_list(topics_list, num_partitions)
        
        json_parts = []
        for i, part in enumerate(partitions, start=1):
            part_data = {"topics": part}
            json_parts.append(part_data)

        return json_parts 

    
    def infer_single(self, org_query: str, query_number: int):   
        main_goal, output_type = self.retrieve_query_goal_and_type(org_query)
        
        #print('main_goal: ',main_goal)
        #print('output_type: ',output_type)
        #print("\nquery_number000: ", query_number)
        topic_details, sample_data, dialog_hist = self.get_entites(main_goal, org_query, query_number = query_number)
        
        if sample_data:
            #print("\n\n sample_data:", sample_data)
            sample_sets = self.retrieve_sample_sets(sample_data)
            
            sample_updated_sets = []
            counter = 0
            for _set_ in sample_sets:
                #try:
                counter += 1
                sample_set = self.get_init_thoughts(main_goal, _set_, counter)
                sample_updated_sets += sample_set["topics"]
    
                '''except Exception as e:
                    print(f"Error in infer_single: {_set_}\n {e}")
                    continue'''

            
            output_file = self.pre_attack_path+"/output_sample_set"+str(query_number)+".json"
            merged_json = {"org_query":org_query,"topics": sample_updated_sets}
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(merged_json, f, indent=4, ensure_ascii=False)

        
    def action(self, num=-1):
        start_time = time.time()
        if not os.path.exists(self.pre_attack_path):
            os.makedirs(self.pre_attack_path)
        
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = list(executor.map(self.infer_single, self.org_data[self.starter-1:num+self.starter-1], list(range(self.starter, num + self.starter))))
            
        end_time = time.time()
        elapsed_time = end_time - start_time  # Time in seconds
        elapsed_minutes = elapsed_time / 60   # Time in minutes
        print(f"Execution Time: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes)")
        return self.pre_attack_path
            
if __name__ == '__main__':
    config = PreAttackConfig()
    attacker = PreAttack(config)
    attacker.infer(num = 2)