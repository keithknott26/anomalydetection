import re
import time
import json
from os.path import dirname
from dateutil.parser import parse
from datetime import timedelta
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

class LogParser:
    def __init__(self):
        # Compiling regular expressions for different log types
        self.log_type_patterns = {
            'ERROR': re.compile(r'\bERROR\b'),
            'WARN': re.compile(r'\bWARN\b'),
            'DEBUG': re.compile(r'\bDEBUG\b'),
            'INFO': re.compile(r'\bINFO\b'),
        }
        config = TemplateMinerConfig()
        self.persistence = FilePersistence("./models/drain3_state.bin")
        self.drain3_config = config.load(f"{dirname(__file__)}/drain3.ini")
        self.template_miner = TemplateMiner(self.persistence, self.drain3_config)
        self.batch_size = 100000 # process and parse logs in batches of X

        # Pattern for replacing whitespace characters
        self.whitespace_pattern = re.compile(r'\t+|\r+|\n+|\r\n+|\s\s+|\r\n|\t')
        
        # List of common timestamp patterns in various log files, expected at the start of the line
        self.timestamp_patterns = [
            r'^[\[\(]?\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[\]\)]?', # Format with hyphens
            r'^[\[\(]?\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}[\]\)]?', # Format with slashes
            r'^[\[\(]?\d{2}/\d{2}/\d{4}:\d{2}:\d{2}:\d{2}[\]\)]?', # Format with colon
            r'^[\[\(]?\w{3} \d{2} \d{2}:\d{2}:\d{2}[\]\)]?', # Format with three-letter month abbreviation
            r'^[\[\(]?\w{3} \d{1,2}, \d{4} \d{1,2}:\d{1,2}:\d{1,2}[\]\)]?', # Format with comma
            r'^[\[\(]?\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2},\d{2}[\]\)]?', # Format with dots and comma
            r'^[\[\(]?\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}[\]\)]?',    # Short date with slashes
            r'^[\[\(]?\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}[\]\)]?',     # Short date with hyphens
            r'^[\[\(]?\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}[\]\)]?',   # Short date with dots
            r'^[\[\(]?\d{4}\d{2}\d{2}T\d{2}:\d{2}:\d{2}[\]\)]?',       # ISO 8601 without separators
            r'^[\[\(]?\d{8} \d{2}:\d{2}:\d{2}[\]\)]?',                # Date without separators
            r'^[\[\(]?[A-Z]\w{2,8} \d{1,2} \d{4} \d{2}:\d{2}:\d{2}[\]\)]?', # Full month name
            r'^[\[\(]?\d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2}[\]\)]?',     # Two-digit day with three-letter month
            r'^[\[\(]?\d{2}:\d{2}:\d{2} \d{2}/\d{2}/\d{4}[\]\)]?',     # Time first, date with slashes
            r'^[\[\(]?[A-Za-z]{3,4} \d{2} \d{2}:\d{2}:\d{2} \d{4}[\]\)]?', # Syslog format
            r'^[\[\(]?[A-Za-z]{3,4}, \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2}[\]\)]?' # HTTP log format (RFC 1123)
            r'^[\[\(]?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?[\]\)]?',   # ISO 8601 with dashes
            r'^[\[\(]?\d{2}\w{3}\d{2} \d{2}:\d{2}:\d{2}[\]\)]?',        # Apache Log with abbreviation
            r'^[\[\(]?\d{2}/\d{2}/\d{4}:\d{2}:\d{2}:\d{2}[\]\)]?',      # Apache Combined Log Format
            r'^[\[\(]?\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{2,3}[\]\)]?', # Java Util Logging
            r'^[\[\(]?[A-Za-z]{3,4} \d{2}, \d{4} \d{2}:\d{2}:\d{2} \w{3}[\]\)]?', # Syslog with timezone
            r'^[\[\(]?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3,6}[\]\)]?', # ISO 8601 with microseconds
            r'^[\[\(]?\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}[\]\)]?',       # Date with slashes and hyphen
            r'^[\[\(]?\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}[\]\)]?',    # Date with dots
            r'^[\[\(]?[A-Za-z]{3,4} \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2}\.\d{3} \w{3}[\]\)]?', # Full Syslog with millis
            r'^[\[\(]?W\d{2}\w{3}\d{2} \d{2}:\d{2}:\d{2} \d{4}[\]\)]?',  # Week number with abbreviation
            r'^[\[\(]?[A-Za-z]{3,4} \w{3} \d{2} \d{2}:\d{2}:\d{2}\.\d{6} \d{4}[\]\)]?', # Full Syslog with microseconds
            r'^[\[\(]?[A-Za-z]{3,4} \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2} \w{3}\.\d{3,6}[\]\)]?', # Timezone with microseconds
            r'^[\[\(]?[A-Za-z]{3,4}, \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2} \w{3}\.\d{3,6}[\]\)]?', # HTTP log format with millis
            r'^[\[\(]?\d{4}\d{2}\d{2} \d{2}:\d{2}:\d{2}\.\d{3}[\]\)]?',  # Date without separators, with milliseconds
            r'^[\[\(]?[A-Za-z]{3,4} \w{3} \d{2} \d{2}:\d{2}:\d{2}\.\d{3} \d{4} \w{3}[\]\)]?' # Full Syslog with timezone and millis
        ]
    
    def parse_log_lines(self, filepath, lines):
        # Performance stats
        line_count = 0
        start_time = time.time()
        batch_start_time = start_time
        batch_size = self.batch_size
        structured_logs = []
        print(f"[{filepath}] --> Line count before condensing & deduplicating: {len(lines)}")
        condensed_lines = self.condense_lines(filepath, lines)
        print(f"[{filepath}] ---> Line count after condensing & deduplicating {len(condensed_lines)}")
        for line in condensed_lines:
            line = line.strip()
            #print(f"Line: {line}")
            result = self.template_miner.add_log_message(line)
            params = self.template_miner.extract_parameters(result['template_mined'], line)
            original_line_content = line
            line_count += 1
            if line_count % batch_size == 0:
                time_took = time.time() - batch_start_time
                rate = batch_size / time_took
                print(f"[{filepath}] --> Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                            f"{len(self.template_miner.drain.clusters)} clusters so far.")
                batch_start_time = time.time()
            if result["change_type"] != "none":
                result_json = json.dumps(result)
                #print(f"Input ({line_count}): {line}")
                #print(f"Result: {result_json}")
            if result["template_mined"] != "none":
                #print(f"Parameters being added: {str(params)}")
                cluster = self.template_miner.match(line)
                if cluster is None:
                    print(f"[{filepath}] --> No cluster match found for line: {line}")
                else:
                    template = cluster.get_template()
                    parameters = self.template_miner.get_parameter_list(template, line)
                    #print(f"Matched template #{cluster.cluster_id}: {template}")
                    #print(f"Parameters: {parameters}")
                structured_logs.append({
                    'template': template,
                    'parameters': parameters,
                    'content': original_line_content,
                })
        time_took = time.time() - start_time
        rate = line_count / time_took
        print(f"[{filepath}] ---> Done mining file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
                    f"{len(self.template_miner.drain.clusters)} clusters")
        sorted_clusters = sorted(self.template_miner.drain.clusters, key=lambda it: it.size, reverse=True)

        print(f"\n\n--------------------------------------------------")
        print(f"[{filepath}] --> Clusters:")
        print(f"--------------------------------------------------")

        for cluster in sorted_clusters:
            print(cluster)
        print(f"\n\n--------------------------------------------------")
        print(f"[{filepath}] --> Prefix Tree:")
        print(f"--------------------------------------------------")
        self.template_miner.drain.print_tree()
        print("\n\n")
        #self.template_miner.profiler.report(0)

        #print(f"Result from add_log_message: {result")
        #print(result)

        return structured_logs

    def condense_lines(self, filepath, lines): 
        output =  []
        in_non_timestamp_block = False
        current_log_entry = None
        skipped_empty_line_count = 0
        # Compile the regular expressions
        carriage_return_pattern = re.compile(r'\r\n+')
        whitespace_pattern1 = re.compile(r'\>\s*\n?\s*\<')
        whitespace_pattern2 = re.compile(r'\>\s\s+\<')
        newline_pattern = re.compile(r'\>\n\<')

        #keep a count of skipped empty lines
        skipped_empty_line_count = sum(1 for line in lines if not line.strip())

        # Strip more than one carriage return
        condensed_lines = [carriage_return_pattern.sub('\r\n', line.strip()) for line in lines]

        #Remove empty values or empty strings
        condensed_lines = [line.strip() for line in condensed_lines if line.strip() != '']
        # Iterate through the condensed_lines list and apply the other regular expressions
        for i in range(len(condensed_lines)):
            condensed_lines[i] = whitespace_pattern1.sub('><', condensed_lines[i].strip())
            condensed_lines[i] = whitespace_pattern2.sub('><', condensed_lines[i].strip())
            condensed_lines[i] = newline_pattern.sub('><', condensed_lines[i].strip())
            condensed_lines[i] = self.whitespace_pattern.sub(' ', condensed_lines[i].strip()).strip()

        for i in range(len(condensed_lines)):
            line = condensed_lines[i].strip()
            # Skip empty lines
            if not line.strip():
                skipped_empty_line_count = skipped_empty_line_count + 1
                continue
            # Search for lines starting with a timestamp
            has_timestamp = self.try_parse_timestamp(line.strip())
            #print(f'Processing line number: {index + 1}')
            #print(f'Line content: {line}')
            #print(f'Has timestamp: {has_timestamp}')
            #print(f'In non-timestamp block: {in_non_timestamp_block}')
            #print(f'Current Log Entry: {current_log_entry}')
            
            # Care for malformed XML
            if line.strip().startswith('<'):
                in_non_timestamp_block = True
                has_timestamp = False
            if has_timestamp:
                in_non_timestamp_block = False
                if current_log_entry:
                    output.append(line.strip())
                    current_log_entry = line.strip()
                elif current_log_entry is None:
                    current_log_entry = line.strip()
                else:
                    if current_log_entry:
                        output.append(current_log_entry.strip())
                    else:
                        output.append(current_log_entry)
            else:
                if in_non_timestamp_block and current_log_entry:
                    current_log_entry += '\n' + line.strip()
                else:
                    if current_log_entry:
                        output.append(current_log_entry.strip()) 
                    current_log_entry = line.strip()
                    in_non_timestamp_block = True
        
        #Remove duplicate lines after the condense processing
        output = list(set(output))

        # Add the last log entry if exists
        if current_log_entry:
            output.append(current_log_entry.strip()) 
        # else:
        #     print(f"Current log entry is none for line {line}!")
        #     print(f"in-non_timestamp_block = {in_non_timestamp_block}")
        #     print(f"has_timestamp = {has_timestamp}")

        print(f"[{filepath}] ---> Condense lines: { len(output)} valid lines, ({skipped_empty_line_count}) empty lines skipped. ")
        return output

    def unique_structured_logs(self, structured_logs_master):
        # Step 1: Create a dictionary to hold unique content
        unique_content = {}

        # Step 2: Iterate through lines
        for line in structured_logs_master:  # Replace with your actual lines iteration
            original_line_content = line['content']  # Adjust as needed
            template = line['template']
            parameters = line['parameters']

            # Check if the content is already in the dictionary
            if original_line_content not in unique_content:
                unique_content[original_line_content] = {
                    'template': template,
                    'parameters': parameters,
                }

        # Step 3: Convert the dictionary into structured_logs
        structured_logs = []
        for content, value in unique_content.items():
            structured_logs.append({
                'template': value['template'],
                'parameters': value['parameters'],
                'content': content,
            })
            
        return structured_logs

    def try_parse_timestamp(self, line):
        try:
            if self.extract_timestamp(line) is not None:
                return True
            else:
                return False
        except ValueError:
            return False
        
    def parse_log_line(self, log_line):
        print("log_line before calling extract_timestamp:", log_line) # Add this line
        log_type = self.identify_log_type(log_line)
        if len(log_line['content']) > 0:
            timestamp = self.extract_timestamp(log_line['content'])
        else:
             timestamp = self.extract_timestamp(log_line)
        if timestamp is not None:
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
        else:
            timestamp_str = "None"
        return {'type': log_type, 'timestamp': timestamp_str, 'content': log_line }

    def identify_log_type(self, log_line):
        if not isinstance(log_line, str):
            return 'INFO'  # Or you can handle this case differently, if needed

        for log_type, pattern in self.log_type_patterns.items():
            if pattern.search(log_line):
                return log_type
        return 'INFO'


    def extract_timestamp(self, log_line):
        timestamp_pattern = re.compile('|'.join(self.timestamp_patterns))

        if log_line.startswith('<'):
            return None
        match = re.search(timestamp_pattern, log_line)
        if match:
            timestamp_str = match.group(0).strip('[]()')  # Remove enclosing brackets if present
            try:
                timestamp = parse(timestamp_str)
                return timestamp
            except Exception as e:
                print(f"Failed to parse timestamp from log line: {log_line}. Error: {e}")
        #print(f"No matching timestamp found in log line: {log_line}.")
        return None
